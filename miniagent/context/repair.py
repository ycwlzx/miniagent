"""消息结构自修复。

**为什么必须有这个模块 (这是本项目最容易被忽略、但最致命的一环)**:

工具调用协议要求严格配对: 模型发出带 `tool_call_id` 的调用, 必须跟一条
同 id 的 `role="tool"` 结果。一旦配对关系被破坏, **服务商通常整批拒绝这次请求**,
不是忽略那一条 —— 也就是说, 一次截断可以让整个请求直接失败。

**会破坏配对的三种来源**:

1. **主动裁剪**。为了适配预算丢掉部分历史时, 很可能刚好把一对调用和结果拆开。
   这是最主流的来源, 也是"我们自己制造的"问题。
2. **异常中断**。任务被取消、进程被杀、工具超时 —— 调用发出去了, 结果没回来,
   落盘的历史里就留着一个没有结果的调用。
3. **跨服务商迁移**。不同实现对消息结构的校验宽严不一, 迁移时可能丢字段。

**修复必须按这个顺序做, 顺序错了会引入新问题**:

    ① 剥离畸形调用  →  ② 丢弃孤儿结果  →  ③ 回填缺失结果

- ① 必须在 ② 之前: 畸形调用 (缺 name) 会让它自己那条调用从"已声明"集合里消失,
  从而把它的结果变成孤儿。先清理调用, 再按清理后的集合判断谁是孤儿, 才不会误判。
- ③ 必须在最后: 它的依据是"已声明但没结果", 而这个集合只有在 ①② 做完之后才准确。

**回填的内容必须诚实**。这里用一段明确的"结果不可用"文案, **绝不伪造数据**。
理由: 结构上必须补齐 (协议要求), 但语义上不能撒谎 ——
如果回填一段看似正常的内容, 模型会基于假信息继续推理, 那比报错危险得多。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 回填文案: 明确说明"这次调用发生了但结果丢了"。
# 不要写成 "...(省略)" 之类含糊的说法 —— 模型会以为是自己该去别处找。
BACKFILL_CONTENT = "[工具结果不可用: 该调用被中断或结果已丢失。如需该信息请重新调用工具。]"

# 剥离畸形调用后, 如果某条 assistant 消息变得完全空白, 用这个占位替换
EMPTY_ASSISTANT_PLACEHOLDER = "[之前的助手消息已省略]"


@dataclass
class RepairResult:
    messages: list[dict[str, Any]]
    stripped_calls: int = 0     # 剥离的畸形调用数
    dropped_orphans: int = 0    # 丢弃的孤儿结果数
    backfilled: int = 0         # 回填的缺失结果数
    dropped_empty: int = 0      # 因清空而删除的消息数
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.stripped_calls or self.dropped_orphans or self.backfilled or self.dropped_empty
        )

    def summary(self) -> str:
        if not self.changed:
            return "结构完好, 无需修复"
        return (
            f"剥离畸形调用 {self.stripped_calls} | "
            f"丢弃孤儿结果 {self.dropped_orphans} | "
            f"回填缺失结果 {self.backfilled}"
            + (f" | 删除空消息 {self.dropped_empty}" if self.dropped_empty else "")
        )


def _is_valid_tool_call(tc: Any) -> bool:
    """判断一条 tool_call 是否可用。

    唯一的硬要求是 function.name 是非空字符串。
    没有名字的调用无法执行、也无法在结果里被引用, 留着只会污染结构 ——
    而且它会拖累它的结果被误判成孤儿。
    """
    if not isinstance(tc, dict):
        return False
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return False
    name = fn.get("name")
    return isinstance(name, str) and bool(name.strip())


def repair_messages(messages: list[dict[str, Any]]) -> RepairResult:
    """修复消息序列, 使其满足工具调用协议。"""
    result = RepairResult(messages=[])

    # ---------- ① 剥离畸形调用 ----------
    # 只有 name 合法的调用才计入 declared —— 这一步同时为 ② 建立了正确的判断依据
    declared: dict[str, str] = {}   # tool_call_id -> tool_name
    stage1: list[dict[str, Any]] = []

    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            stage1.append(msg)
            continue

        valid, invalid_count = [], 0
        for tc in msg["tool_calls"]:
            if _is_valid_tool_call(tc):
                valid.append(tc)
                declared[tc["id"]] = tc["function"]["name"]
            else:
                invalid_count += 1
        result.stripped_calls += invalid_count

        if invalid_count:
            result.notes.append(f"剥离 {invalid_count} 条畸形工具调用 (缺 name)")

        content = msg.get("content")
        if not valid:
            # 调用全被剥离, 这条消息若也没有正文就成了空壳 —— 直接删掉
            if not (isinstance(content, str) and content.strip()):
                result.dropped_empty += 1
                continue
            stage1.append({**msg, "tool_calls": None} if False else {k: v for k, v in msg.items() if k != "tool_calls"})
        else:
            stage1.append({**msg, "tool_calls": valid})

    # ---------- ② 丢弃找不到配对的孤儿结果 ----------
    fulfilled: set[str] = set()
    stage2: list[dict[str, Any]] = []

    for msg in stage1:
        if msg.get("role") != "tool":
            stage2.append(msg)
            continue
        tid = msg.get("tool_call_id")
        if not isinstance(tid, str) or not tid or tid not in declared or tid in fulfilled:
            # 三种情况都算孤儿:
            #   - 没有 id (无法归属)
            #   - id 不在已声明的调用里 (调用被裁掉了)
            #   - 同一个 id 出现了第二次 (重复结果)
            result.dropped_orphans += 1
            continue
        fulfilled.add(tid)
        stage2.append(msg)

    if result.dropped_orphans:
        result.notes.append(
            f"丢弃 {result.dropped_orphans} 条孤儿工具结果 "
            f"(对应的调用已被裁剪或缺失)"
        )

    # ---------- ③ 回填缺失的结果 ----------
    # 位置很关键: 结果必须插在**发出调用的那条 assistant 消息之后**,
    # 而且要在下一条消息之前。插在别处同样是非法结构。
    stage3: list[dict[str, Any]] = []
    for msg in stage2:
        stage3.append(msg)
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            if tc["id"] not in fulfilled:
                stage3.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": BACKFILL_CONTENT,
                    }
                )
                fulfilled.add(tc["id"])
                result.backfilled += 1

    if result.backfilled:
        result.notes.append(f"为 {result.backfilled} 条无结果的调用回填占位内容")

    result.messages = stage3
    return result


def collect_tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
    """建立 tool_call_id -> tool_name 映射。

    截断模块需要它来判断"这条结果来自哪个工具", 从而选择留头还是留尾。
    之所以要单独扫一遍: tool 消息本身**不包含工具名**, 只有 id ——
    名字只在发出调用的那条 assistant 消息里。
    """
    mapping: dict[str, str] = {}
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name")
            tid = tc.get("id")
            if isinstance(name, str) and isinstance(tid, str):
                mapping[tid] = name
    return mapping
