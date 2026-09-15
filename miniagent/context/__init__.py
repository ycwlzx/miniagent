"""上下文层: 预算、截断、修复。

**本层的硬约束: 除了"生成摘要"这一个明确的例外, 这里不调用 LLM。**

理由: 上下文处理如果嵌了模型调用, 压缩就会递归消耗上下文 ——
你为了省 token 去调一次模型, 这次调用自己又要占 token, 而且可能在
"压缩的内容本身也需要压缩"这个方向上无限递归。

所以 budget / truncate / repair 三个模块全是**纯函数或纯数据**,
可以脱离网络、脱离终端直接测试。这个约束是刻意的设计, 不是巧合。
"""

from miniagent.context.budget import (
    TokenBudget,
    estimate_message,
    estimate_request,
    estimate_text,
)
from miniagent.context.repair import (
    RepairResult,
    collect_tool_names,
    repair_messages,
)
from miniagent.context.truncate import (
    TruncateConfig,
    TruncateResult,
    apply_to_messages,
    truncate_tool_result,
)

__all__ = [
    "TokenBudget",
    "estimate_text",
    "estimate_message",
    "estimate_request",
    "TruncateConfig",
    "TruncateResult",
    "truncate_tool_result",
    "apply_to_messages",
    "RepairResult",
    "repair_messages",
    "collect_tool_names",
]
