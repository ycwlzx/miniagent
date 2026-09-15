"""LLM 客户端: 把 OpenAI 兼容接口包成一个"只收发纯数据"的薄层。

为什么要单独一层 (而不是 loop 里直接调 SDK):

1. **隔离 SDK 类型**。上层 (loop / context) 只处理 dict, 不依赖 openai 的对象类型。
   将来换服务商、换 SDK 版本, 只改这一个文件。
   这也是为什么 LLMResponse 是 dataclass 而不是直接返回 SDK 的响应对象。

2. **必须拿回真实 usage**。我们的预算是*估算*出来的, 估算必然有误差。
   每次响应里服务商返回的真实 prompt_tokens 是**校准误差的唯一依据** ——
   没有它, 你永远不知道自己估得准不准, 也就没法解释"缓冲值为什么定这个数"。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class LLMResponse:
    """一次模型调用的结果。纯数据, 方便序列化和测试。"""

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    # 真实 token 用量: 用于校准本地估算 (见 context/budget.py 的误差统计)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        """模型是否请求调用工具。这是 loop 的继续/终止判据之一。"""
        return bool(self.tool_calls)


class LLMClient:
    """OpenAI 兼容客户端。

    默认读环境变量, 便于切换服务商:
        LLM_API_KEY   —— 必填
        LLM_BASE_URL  —— 默认智谱 (OpenAI 兼容协议)
        LLM_MODEL     —— 默认 glm-4.5-air
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL", "glm-4.5-air")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.base_url = base_url or os.getenv(
            "LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
        )
        # temperature 默认 0: 编码智能体要的是可复现, 不是创意。
        # 同一个任务两次跑出完全不同的行为, 会让"验证"这件事失去意义。
        self.temperature = temperature
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError(
                "缺少 LLM_API_KEY。请设置环境变量, 或在项目根目录的 .env 里配置后加载。"
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
        )

        # 累计用量: verify.py 会用它汇总整轮任务的成本
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """发一次请求。

        注意 messages 是**调用方构造好的副本**, 这里不做任何修改 ——
        上下文治理 (裁剪/压缩) 发生在调用之前, 由 context 层负责。
        这种分工让 LLM 层保持无状态, 容易被测试和替换。
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        raw = self._client.chat.completions.create(**kwargs)
        self.call_count += 1

        choice = raw.choices[0]
        msg = choice.message

        # 把 SDK 对象转成纯 dict —— 上层不关心 SDK 的类型
        tool_calls: list[dict[str, Any]] = []
        for tc in (msg.tool_calls or []):
            tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
            )

        usage = getattr(raw, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, int]:
        """累计统计。用于验证脚本输出成本数据。"""
        return {
            "calls": self.call_count,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
        }
