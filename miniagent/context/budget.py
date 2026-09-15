"""上下文预算与水位检测。

**核心公式**:

    预算 = 模型窗口 − 最大输出 − 安全缓冲

三项各自的理由:

- **模型窗口**: 输入 + 输出的总上限, 服务商给的硬约束。
- **最大输出**: 你自己要留出的生成空间。如果不留, 输入占满窗口后模型没有空间输出,
  表现为"回答被截断"或"只吐两个字就结束", 而且这种失败很难归因到上下文管理上。
- **安全缓冲**: 吸收估算误差。**这一项的存在理由是必须能讲清的** ——
  我们的 token 数是**估算**出来的, 必然有偏差; 不留缓冲, 一次低估就直接超限。

**为什么估算而不是精确计数**:

精确计数需要接入对应模型的 tokenizer (不同模型的分词方式不同, 中文差异尤其明显),
会引入额外依赖并且需要跟版本。估算的误差可以通过**运行时用真实 usage 校准**来收敛 ——
这比一次性引入重依赖更划算。

但这是取舍, 不是偷懒: 如果场景要求精确 (比如按 token 计费的商业化产品),
就应该老老实实接 tokenizer。**这条边界要讲清楚, 不要假装估算没有代价。**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 每条消息的固定结构开销 (角色标记、分隔符等)
# 这个值不需要精确 —— 它是经验值, 误差会被安全缓冲吸收
MESSAGE_OVERHEAD_TOKENS = 4

# 中文按 1 token/字 估, 其他字符按 ~3.5 字符/token 估。
# 刻意**偏保守 (估高)**: 低估的代价是请求失败, 高估的代价只是少用一点窗口。
_CJK_PER_TOKEN = 1.0
_OTHER_CHARS_PER_TOKEN = 3.5


def estimate_text(text: str) -> int:
    """估算一段文本的 token 数。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk / _CJK_PER_TOKEN + other / _OTHER_CHARS_PER_TOKEN) + 1


def estimate_message(message: dict[str, Any]) -> int:
    """估算单条消息的 token 数。

    注意 tool_calls 也要算 —— 模型发出的工具调用同样占据输入空间,
    而且它的参数 (比如要写入的文件内容) 可能非常大。
    漏算这一项会导致"明明消息不多, 怎么又超了"的困惑。
    """
    total = MESSAGE_OVERHEAD_TOKENS

    content = message.get("content")
    if isinstance(content, str):
        total += estimate_text(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                total += estimate_text(part["text"])

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        total += estimate_text(str(fn.get("name", "")))
        total += estimate_text(str(fn.get("arguments", "")))
        total += MESSAGE_OVERHEAD_TOKENS

    return total


def estimate_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """估算一次完整请求的输入 token。

    **工具定义必须算进来**。它是每次请求都要付的固定开销,
    而且不可裁剪 —— 工具越多、schema 描述越详细, 这个基线就越高。
    忽略它的后果是: 历史已经裁到很短了, 请求还是超限, 而你会一直盯着历史找原因。
    """
    total = sum(estimate_message(m) for m in messages)
    if tools:
        for t in tools:
            fn = t.get("function") or {}
            total += estimate_text(str(fn.get("name", "")))
            total += estimate_text(str(fn.get("description", "")))
            total += estimate_text(str(fn.get("parameters", {})))
            total += MESSAGE_OVERHEAD_TOKENS
    return total


@dataclass
class TokenBudget:
    """预算与水位。

    这个对象**不修改消息** —— 它只回答"还能装多少""现在多危险"。
    真正动消息的是 truncate.py / repair.py, 这样职责清晰。
    """

    window: int = 128_000
    max_output: int = 4_096
    safety_buffer: int = 1_024

    # 用于计算水位与统计
    last_estimated: int = 0
    last_actual: int = 0
    # (估算, 真实) 样本对, 用于校准误差
    samples: list[tuple[int, int]] = field(default_factory=list)
    snipped_times: int = 0

    @property
    def budget(self) -> int:
        """本次请求可用的输入上限。"""
        return max(0, self.window - self.max_output - self.safety_buffer)

    def pressure(self, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None) -> float:
        """水位: 0.0 (空) ~ 1.0 (正好到预算)。

        > 1.0 表示已经超预算, 必须处理。
        用比值而不是绝对值, 是因为阈值策略需要跨模型通用 ——
        128K 窗口的 70% 和 8K 窗口的 70% 是不同的绝对量, 但同样危险。
        """
        used = estimate_request(messages, tools)
        self.last_estimated = used
        return used / self.budget if self.budget > 0 else 1.0

    def over_budget(self, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]] | None = None) -> bool:
        return self.pressure(messages, tools) > 1.0

    # ------------------------------------------------------------------ #

    def calibrate(self, actual_prompt_tokens: int) -> dict[str, float]:
        """用服务商返回的真实 token 数校准估算误差。

        这是整个预算体系里**最容易被忽略、但最能体现工程意识**的一步:
        没有它, "缓冲值取 1024"就永远是一个拍脑袋的数字;
        有了它, 你能拿出"估算平均偏高 12%, 所以缓冲取 X 足够"这种有依据的说法。

        返回当前的误差统计。
        """
        self.last_actual = actual_prompt_tokens
        if self.last_estimated > 0 and actual_prompt_tokens > 0:
            self.samples.append((self.last_estimated, actual_prompt_tokens))
            # 只保留最近 50 次, 避免长期跑下去内存无限增长
            if len(self.samples) > 50:
                self.samples.pop(0)
        return self.error_stats()

    def error_stats(self) -> dict[str, float]:
        """估算误差统计。正数表示我们估高了(安全), 负数表示估低了(危险)。"""
        if not self.samples:
            return {"samples": 0, "mean_bias": 0.0, "max_under": 0.0}

        biases = []
        for est, act in self.samples:
            if act > 0:
                biases.append((est - act) / act)
        if not biases:
            return {"samples": 0, "mean_bias": 0.0, "max_under": 0.0}

        return {
            "samples": len(biases),
            "mean_bias": sum(biases) / len(biases),
            # 最严重的一次低估 (负偏差的最小值), 用来判断缓冲够不够
            "max_under": min(biases),
        }

    def describe(self, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None) -> str:
        """给日志/CLI 用的一行摘要。"""
        used = estimate_request(messages, tools)
        p = used / self.budget if self.budget > 0 else 1.0
        bar_len = 20
        filled = min(bar_len, int(p * bar_len))
        bar = "#" * filled + "." * (bar_len - filled)
        return (
            f"[{bar}] {p*100:5.1f}%  "
            f"估算 {used:,} / 预算 {self.budget:,} tokens  "
            f"(窗口 {self.window:,})"
        )
