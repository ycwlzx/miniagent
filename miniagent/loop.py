"""手写 Agent 循环 —— 本项目的核心, 也是不用任何框架的那部分。

**为什么坚持手写**: 编排框架会把下面这段藏起来, 而这段恰恰是 Agent 工程的核心。
框架给你 `add_node` / `add_edge`, 你能搭出流程, 但说不清"一轮请求前后到底发生了什么"。

**一轮循环里, 治理动作的顺序是有讲究的** (每一步都在防一种具体的失败):

    ① 修复结构   repair_messages    → 防"上一次截断留下非法配对, 这次整批被拒"
    ② 截断结果   apply_to_messages  → 防"工具结果本身就把预算吃掉一半"
    ③ 查水位     budget.pressure    → 判断还能不能发, 不能发就先裁剪
    ④ 裁剪历史   _snip_history      → 无损降级 (只影响这次请求)
    ⑤ 发请求     llm.chat           → 拿回真实 usage
    ⑥ 校准误差   budget.calibrate   → 让"缓冲值"从拍脑袋变成有依据
    ⑦ 执行工具   registry.execute   → 每次都过 safety 判断
    ⑧ 结果回填   append tool 消息   → 回到 ①

顺序不能乱的理由:
- ① 必须在 ② 之前: 截断会改变消息内容, 但不会改变配对关系; 而修复要基于当前
  真实的结构来判断谁是孤儿。先修再截, 判断依据才是准的。
- ② 必须在 ③ 之前: 不先截断, 水位算出来会偏高, 导致做没必要的裁剪 ——
  裁剪是有代价的 (丢信息), 能在源头解决的问题不要用裁剪兜。
- ⑥ 必须在 ⑤ 之后: 校准依赖服务商返回的真实 token, 这是唯一的真值来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from miniagent.context import (
    TokenBudget,
    TruncateConfig,
    apply_to_messages,
    collect_tool_names,
    estimate_request,
    repair_messages,
)
from miniagent.llm import LLMClient
from miniagent.safety import ALLOW, DENY, SafetyPolicy
from miniagent.tools.base import ToolRegistry

SYSTEM_PROMPT = """你是一个编码助手, 可以读取和修改工作区内的文件, 也可以执行 shell 命令。

工作要求:
1. 先了解再动手 —— 改文件前先 read_file 确认原文, 不要凭印象修改。
2. 小步验证 —— 每次修改后尽量跑一次验证 (测试、语法检查、或读回文件确认)。
3. 遇到错误先分析再重试 —— 不要重复执行同一个失败的命令。
4. 完成后用一两句话说明你做了什么改动。

注意: 文件修改是不可逆的, edit_file 要求原文唯一匹配, 如果报错请重新读取文件。"""


@dataclass
class LoopOptions:
    max_turns: int = 30
    # 水位超过它就开始裁剪历史 (无损降级)
    snip_threshold: float = 0.75
    # 水位超过 1.0 且裁剪无效时, 直接结束并报错
    hard_limit: float = 1.0
    # 裁剪时至少保留最近 N 条消息 (含工具往返), 保证当前任务的连续性
    keep_recent: int = 6
    # 单轮最多执行多少个工具调用
    max_tools_per_turn: int = 8


@dataclass
class LoopResult:
    success: bool
    answer: str
    turns: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# 事件回调: CLI 用它渲染输出, 测试用它断言过程
EventHook = Callable[[str, dict[str, Any]], None]


class AgentLoop:
    """一次任务的执行器。"""

    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry,
        safety: SafetyPolicy,
        budget: TokenBudget,
        truncate_config: TruncateConfig,
        options: LoopOptions | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.safety = safety
        self.budget = budget
        self.truncate_config = truncate_config
        self.options = options or LoopOptions()
        self._emit_hook = on_event

        # 统计: verify.py 与简历里的数字都来自这里
        self.stats: dict[str, Any] = {
            "turns": 0,
            "tool_calls": 0,
            "snip_events": 0,        # 裁剪触发次数
            "snipped_messages": 0,   # 被裁掉的消息总数
            "repair_events": 0,      # 结构修复触发次数
            "truncate_events": 0,    # 工具结果截断次数
            "max_pressure": 0.0,     # 峰值水位
        }

    # ------------------------------------------------------------------ #

    def _emit(self, event: str, **payload: Any) -> None:
        if self._emit_hook:
            self._emit_hook(event, payload)

    def run(self, task: str) -> LoopResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        tools = self.registry.schemas()

        for turn in range(1, self.options.max_turns + 1):
            self.stats["turns"] = turn

            # ---------- ① 结构修复 ----------
            # 放在最前面: 上一轮如果发生截断, 这里就是兜住非法结构的地方
            repaired = repair_messages(messages)
            if repaired.changed:
                messages = repaired.messages
                self.stats["repair_events"] += 1
                self._emit("repair", turn=turn, summary=repaired.summary())

            # ---------- ② 工具结果截断 ----------
            tool_names = collect_tool_names(messages)
            messages, trunc_results = apply_to_messages(
                messages, tool_names, self.truncate_config
            )
            if trunc_results:
                self.stats["truncate_events"] += len(trunc_results)
                self._emit(
                    "truncate",
                    turn=turn,
                    count=len(trunc_results),
                    detail=[
                        f"{r.original_chars:,}→{len(r.content):,}字符"
                        for r in trunc_results
                    ],
                )

            # ---------- ③ 水位检测 ----------
            pressure = self.budget.pressure(messages, tools)
            self.stats["max_pressure"] = max(self.stats["max_pressure"], pressure)
            self._emit(
                "budget", turn=turn, pressure=pressure, bar=self.budget.describe(messages, tools)
            )

            # ---------- ④ 裁剪历史 (无损降级) ----------
            if pressure > self.options.snip_threshold:
                messages, snipped = self._snip_history(messages, tools)
                if snipped:
                    self.stats["snip_events"] += 1
                    self.stats["snipped_messages"] += snipped
                    self._emit("snip", turn=turn, dropped=snipped)

                # 裁完仍然超预算 —— 这是真的装不下了, 明确失败而不是静默丢数据
                pressure = self.budget.pressure(messages, tools)
                if pressure > self.options.hard_limit:
                    self._emit("over_budget", turn=turn, pressure=pressure)
                    return LoopResult(
                        success=False,
                        answer=(
                            f"上下文预算不足: 估算 {estimate_request(messages, tools):,} tokens "
                            f"超过预算 {self.budget.budget:,}。"
                            f"裁剪历史后仍装不下, 请把任务拆小后重试。"
                        ),
                        turns=turn,
                        messages=messages,
                        stats=self.stats,
                    )

            # ---------- ⑤ 发请求 ----------
            self._emit("thinking", turn=turn)
            response = self.llm.chat(messages, tools)

            # ---------- ⑥ 校准估算误差 ----------
            if response.prompt_tokens:
                self.budget.calibrate(response.prompt_tokens)

            # 把 assistant 消息放回历史 (带 tool_calls 时也必须保留 —— 后面要配对)
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            # ---------- 终止条件: 模型不再要求工具 ----------
            if not response.wants_tools:
                self._emit("done", turn=turn, answer=response.content or "")
                return LoopResult(
                    success=True,
                    answer=response.content or "(模型未返回内容)",
                    turns=turn,
                    messages=messages,
                    stats=self.stats,
                )

            # ---------- ⑦ 执行工具 ----------
            for tc in response.tool_calls[: self.options.max_tools_per_turn]:
                name = tc["function"]["name"]
                args_raw = tc["function"]["arguments"]
                self.stats["tool_calls"] += 1

                decision = self.safety.decide(name, self._parse_args(args_raw))
                if decision.action == DENY:
                    self.safety.deny_count += 1
                    result_text = f"[安全策略拒绝] {decision.reason}"
                    self._emit("denied", turn=turn, tool=name, reason=decision.reason)
                else:
                    if decision.needs_confirm:
                        allowed, why = self.safety.confirm(
                            name, self._parse_args(args_raw), decision.reason,
                            ask=self._ask,
                        )
                        self._emit("confirm", turn=turn, tool=name, allowed=allowed, why=why)
                        if not allowed:
                            result_text = f"[用户拒绝执行] {why}"
                            messages.append(
                                {"role": "tool", "tool_call_id": tc["id"], "content": result_text}
                            )
                            continue

                    self._emit("tool_call", turn=turn, tool=name, args=args_raw)
                    # 执行结果**一定**是字符串 (契约由 ToolRegistry 保证)
                    result_text = self.registry.execute(name, args_raw)
                    self._emit(
                        "tool_result", turn=turn, tool=name, chars=len(result_text)
                    )

                # ---------- ⑧ 结果回填 ----------
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    }
                )

        return LoopResult(
            success=False,
            answer=f"达到最大轮数 {self.options.max_turns} 仍未完成任务。",
            turns=self.options.max_turns,
            messages=messages,
            stats=self.stats,
        )

    # ------------------------------------------------------------------ #

    def _snip_history(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """从历史的**较早处**裁剪消息, 直到水位回到阈值以下。

        三个设计约束:

        1. **永远保留第一条 system 消息**。它是行为约束, 丢了模型会失忆
           (不再遵守"先读再改"这类规则), 而这种失效很隐蔽 ——
           表现为"模型变笨了", 而不是报错。

        2. **保留最近 N 条**。最近的消息是当前任务的现场, 裁掉它们等于让模型
           忘记自己刚才在干什么。这也是为什么裁剪从**较早处**开始,
           而不是简单地"从最老的开始删, 删到够为止"。

        3. **允许切断配对, 交给 repair 兜底**。这里刻意不实现"成对裁剪"的复杂逻辑:
           按消息条数裁剪必然会遇到拆散配对的情况, 与其在这里做一套复杂的配对感知,
           不如让下一轮的 repair 统一修复 —— 一个职责明确的兜底层,
           比在每个裁剪点都小心谨慎更可靠。

        返回 (新消息列表, 被丢弃的条数)。
        """
        if len(messages) <= self.options.keep_recent + 1:
            return messages, 0

        head = messages[:1]                      # system
        tail = messages[-self.options.keep_recent:]
        middle = messages[1:-self.options.keep_recent]

        dropped = 0
        # 从头开始丢, 每丢一条就检查一次水位 —— 丢够了就停, 不做无谓的信息损失
        while middle and self.budget.pressure(head + middle + tail, tools) > self.options.snip_threshold:
            middle.pop(0)
            dropped += 1

        return head + middle + tail, dropped

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_args(args_raw: Any) -> dict[str, Any]:
        """给 safety 判断用的参数解析 (容错, 失败返回空 dict)。

        注意: 这里只是为了让安全策略能读到 command 字段, 真正的参数解析
        在 ToolRegistry.execute 里 —— 那里失败会返回可读的错误文本。
        两处的分工是: 这里"尽力解析, 失败就当作没有参数", 那里"必须给出反馈"。
        """
        import json

        if isinstance(args_raw, dict):
            return args_raw
        if not isinstance(args_raw, str) or not args_raw.strip():
            return {}
        try:
            parsed = json.loads(args_raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _ask(self, prompt: str) -> str:
        """确认输入。子类或测试可以覆盖它, 从而摆脱真实终端。"""
        return input(prompt)
