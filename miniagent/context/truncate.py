"""工具结果截断与溢出落盘。

**要解决的问题**: 单次工具调用就能返回几十万 token, 直接把窗口撑爆。

**两道截断, 缺一不可**:

- 第一道在 `tools/shell.py` 里: 命令输出超过 30K 字符就截。
  它的目的是"不把垃圾运进内存"。
- 第二道在这里: 进历史之前再按 token 预算截一次。
  它的目的是"保证进历史的内容受控"。

有人会问"两道是不是重复了", 答案是: 第一道防的是体积, 第二道防的是预算。
工具层不知道模型的窗口有多大, 上下文层不知道命令输出了什么形态, 各自只能守自己那段。

**三个设计决策 (面试重点)**:

1. **按工具类型区分截断策略** —— 这是本项目最值得讲的一个判断:
   - 命令输出 (`run_shell`) 留**尾部**: 编译错误、测试失败、最后的统计结果都在末尾,
     把尾巴砍掉等于把最有用的信息扔了。
   - 文件读取 (`read_file`) 留**头部**: 代码的定义、导入、类声明在前面,
     从中间开始读一段陌生的代码价值很低。
   - 其他工具头尾各留, 中间明确标出省略量。
   没有"通用最优策略"这回事, 只有"按内容形态选策略"。

2. **落盘而不是丢弃**。被截掉的内容写到工作区的 `.miniagent/offload/` 下,
   截断处留下路径和原始大小, 模型需要细节时可以自己 read_file 读回来。
   这样截断就从"信息丢失"变成了"换成更便宜的承载方式"。

3. **读文件类工具豁免落盘**。这是个真实的坑:
   read_file 的结果如果落盘 → 模型去读那个落盘文件 → 又超限 → 又落盘 →
   无限循环放大, 而且每次都在磁盘上留下新文件。
   豁免名单必须显式维护, 不能靠"我觉得不会发生"。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

# 按内容形态选择保留哪一端 —— 见模块文档第 1 条
TAIL_TOOLS = frozenset({"run_shell"})
HEAD_TOOLS = frozenset({"read_file", "glob", "grep"})

# 豁免落盘的工具 (见模块文档第 3 条)
OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})

OFFLOAD_DIR = ".miniagent/offload"


@dataclass
class TruncateConfig:
    """截断配置。

    max_chars 是**字符**上限而不是 token 上限, 这是有意的:
    截断发生在进历史之前, 而 token 预算的最终把关在 loop 里 (用 budget 判断)。
    这里用一个直观的字符阈值, 让"单条结果的体积"这件事有个简单可控的上限。
    """

    max_chars: int = 8_000
    workspace: Path | None = None
    enabled: bool = True
    # 截断处保留的提示语长度上限 (防止提示语本身太长)
    notice_budget: int = 400


@dataclass
class TruncateResult:
    content: str
    truncated: bool
    original_chars: int
    offload_path: str | None = None


def truncate_tool_result(
    tool_name: str,
    content: str,
    config: TruncateConfig,
) -> TruncateResult:
    """按工具类型截断结果, 需要时落盘。

    必须满足的约束: 返回的 content 一定可以被安全放进消息历史
    (是字符串、长度受控、且带有"这里被截断了"的明确信息)。
    """
    original = len(content)
    if not config.enabled or original <= config.max_chars:
        return TruncateResult(content=content, truncated=False, original_chars=original)

    # 落盘 (豁免工具跳过) —— 先落盘再截断, 保证原始内容不丢
    offload_path: str | None = None
    if config.workspace is not None and tool_name not in OFFLOAD_EXEMPT_TOOLS:
        offload_path = _offload(content, tool_name, config.workspace)

    keep = max(1, config.max_chars - config.notice_budget)
    if tool_name in TAIL_TOOLS:
        body = content[-keep:]
        head_note = f"前面 {original - keep:,} 字符已省略"
        tail_note = ""
    elif tool_name in HEAD_TOOLS:
        body = content[:keep]
        head_note = ""
        tail_note = f"后面 {original - keep:,} 字符已省略"
    else:
        half = keep // 2
        body = f"{content[:half]}\n\n... [中间省略 {original - 2 * half:,} 字符] ...\n\n{content[-half:]}"
        head_note = tail_note = ""

    notice = _build_notice(tool_name, original, offload_path, head_note, tail_note)
    return TruncateResult(
        content=f"{notice}\n{body}",
        truncated=True,
        original_chars=original,
        offload_path=offload_path,
    )


def _build_notice(
    tool_name: str,
    original: int,
    offload_path: str | None,
    head_note: str,
    tail_note: str,
) -> str:
    """构造截断提示。

    提示必须回答模型三个问题, 否则它不知道自己看到的是不是全部:
      1. 有没有被截断?        -> "已截断"
      2. 截掉了多少?           -> 原始字符数
      3. 想看得更多该怎么办?   -> 给出可读回的路径
    """
    parts = [f"[结果已截断] {tool_name} 原始 {original:,} 字符"]
    if head_note:
        parts.append(head_note)
    if tail_note:
        parts.append(tail_note)
    if offload_path:
        parts.append(f"完整内容已保存到 {offload_path}，需要时可 read_file 分段查看")
    else:
        parts.append("(该工具结果不落盘，仅保留上述片段)")
    return " | ".join(parts)


def _offload(content: str, tool_name: str, workspace: Path) -> str:
    """把完整内容写到工作区, 返回**相对路径**。

    返回相对路径而不是绝对路径, 有两个原因:
      1. 省 token —— 绝对路径里那一长串盘符和临时目录名每轮都要重复出现
      2. 可直接喂给 read_file —— 它接受的就是相对工作区的路径
    """
    directory = workspace / OFFLOAD_DIR
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:10]
    stamp = time.strftime("%H%M%S")
    filename = f"{tool_name}_{stamp}_{digest}.txt"
    path = directory / filename
    path.write_text(content, encoding="utf-8")
    return f"{OFFLOAD_DIR}/{filename}"


def apply_to_messages(
    messages: list[dict],
    tool_names_by_id: dict[str, str],
    config: TruncateConfig,
) -> tuple[list[dict], list[TruncateResult]]:
    """遍历历史, 对所有 role="tool" 的消息执行截断。

    为什么要在历史里也扫一遍, 而不是只在工具刚执行完时截一次?
    因为同一个历史会被反复发送, 而预算在变 ——
    上一轮宽裕时没截的结果, 这一轮可能就超了。
    进入请求前统一扫一遍, 是唯一能保证"无论历史怎么来的都受控"的做法。
    """
    results: list[TruncateResult] = []
    out: list[dict] = []
    for msg in messages:
        if msg.get("role") != "tool":
            out.append(msg)
            continue
        tool_name = tool_names_by_id.get(msg.get("tool_call_id", ""), "")
        content = msg.get("content") or ""
        if not isinstance(content, str):
            out.append(msg)
            continue
        res = truncate_tool_result(tool_name, content, config)
        if res.truncated:
            results.append(res)
            out.append({**msg, "content": res.content})
        else:
            out.append(msg)
    return out, results
