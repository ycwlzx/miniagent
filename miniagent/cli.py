"""命令行入口: 装配依赖 + REPL + 事件渲染。

**这里做了什么装配** (也是唯一把各部分接起来的地方):

    .env ──> LLMClient
    workspace ──> ToolRegistry (工具需要知道工作区边界在哪)
              ──> TruncateConfig (落盘目录在工作区内)
    SafetyPolicy ──> AgentLoop
    TokenBudget  ──> AgentLoop

依赖全部显式构造、显式传递 —— 没有全局单例。
好处是测试时可以整体替换掉任意一层 (比如把 LLM 换成假的),
这也是"策略与执行分离"能落到实处的必要条件: 如果到处是全局对象,
再好的分层也只是纸面上的。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from miniagent.context import TokenBudget, TruncateConfig
from miniagent.llm import LLMClient
from miniagent.loop import AgentLoop, LoopOptions
from miniagent.safety import SafetyPolicy
from miniagent.tools import build_default_registry

# ---------------------------------------------------------------- 颜色 --

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def dim(t: str) -> str:
    return _c(t, "2")


def cyan(t: str) -> str:
    return _c(t, "36")


def green(t: str) -> str:
    return _c(t, "32")


def yellow(t: str) -> str:
    return _c(t, "33")


def red(t: str) -> str:
    return _c(t, "31")


# ------------------------------------------------------------ .env 加载 --


def load_env(project_root: Path) -> None:
    """极简 .env 加载器。

    刻意不引入 python-dotenv: 这个项目要展示"少依赖也能跑",
    而这十几行逻辑足够覆盖 配置项目 的常见用法 (KEY=VALUE、# 注释、引号可选)。
    """
    env_file = project_root / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        # 不覆盖已存在的环境变量: 命令行 export 的优先级更高
        os.environ.setdefault(key, value)


# ------------------------------------------------------------ 事件渲染 --


def make_renderer(verbose: bool = True):
    """把 loop 的内部事件渲染成可读输出。

    这里集中处理"报告什么", loop 只管"报告事件" ——
    所以同一套 loop 可以被 CLI、被 verify.py、被测试用不同的方式消费。

    顺便说明: 打印水位和治理动作**不是调试信息, 是产品的一部分**。
    用户需要知道"它为什么变慢了/它在省钱吗", 而且这些数字正是验证时要采集的。
    """

    def render(event: str, payload: dict[str, Any]) -> None:
        turn = payload.get("turn", "?")

        if event == "budget":
            bar = payload.get("bar", "")
            p = payload.get("pressure", 0.0)
            # 水位超过 75% 标黄, 超过 100% 标红 —— 让"危险"一眼可见
            paint = red if p > 1.0 else (yellow if p > 0.75 else dim)
            print(paint(f"  [{turn}] {bar}"))

        elif event == "repair":
            print(yellow(f"  [{turn}] ⚙ 结构修复: {payload.get('summary')}"))

        elif event == "truncate":
            detail = ", ".join(payload.get("detail", [])[:3])
            print(yellow(f"  [{turn}] ✂ 截断 {payload.get('count')} 条工具结果  {detail}"))

        elif event == "snip":
            print(yellow(f"  [{turn}] ↯ 裁剪历史 {payload.get('dropped')} 条"))

        elif event == "thinking":
            print(dim(f"  [{turn}] … 请求模型中"))

        elif event == "tool_call":
            if verbose:
                raw = str(payload.get("args", ""))
                shown = raw if len(raw) <= 160 else raw[:160] + "…"
                print(cyan(f"  [{turn}] → {payload.get('tool')}  {shown}"))

        elif event == "tool_result":
            print(dim(f"  [{turn}] ← {payload.get('chars'):,} 字符"))

        elif event == "confirm":
            mark = green("已允许") if payload.get("allowed") else red("已拒绝")
            print(f"  [{turn}] {mark}: {payload.get('why')}")

        elif event == "denied":
            print(red(f"  [{turn}] ⛔ 安全拒绝: {payload.get('reason')}"))

        elif event == "over_budget":
            print(red(f"  [{turn}] ✖ 超出预算 (水位 {payload.get('pressure', 0):.1%}), 停止"))

        elif event == "done":
            print(green(f"  [{turn}] ✓ 完成"))

    return render


# ------------------------------------------------------------------ 装配 --


def build_agent(
    workspace: Path,
    *,
    mode: str = "ask",
    window: int = 128_000,
    max_output: int = 4_096,
    safety_buffer: int = 1_024,
    max_chars: int = 8_000,
    on_event=None,
    llm: LLMClient | None = None,
) -> AgentLoop:
    """构造一个可以直接跑的 AgentLoop。"""
    workspace = workspace.resolve()
    client = llm or LLMClient()

    return AgentLoop(
        llm=client,
        registry=build_default_registry(workspace),
        safety=SafetyPolicy(mode=mode),
        budget=TokenBudget(
            window=window,
            max_output=max_output,
            safety_buffer=safety_buffer,
        ),
        truncate_config=TruncateConfig(max_chars=max_chars, workspace=workspace),
        options=LoopOptions(),
        on_event=on_event,
    )


# ------------------------------------------------------------------ 主流程 --


BANNER = r"""
  miniagent · 手写终端编码智能体
  工作区: {ws}
  模型:   {model}
  输入任务开始, 空行或 Ctrl+C 退出
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="miniagent",
        description="手写的终端编码智能体: 把上下文当作有限资源来管理",
    )
    parser.add_argument("-w", "--workspace", default=".", help="工作区目录 (默认当前目录)")
    parser.add_argument("-t", "--task", help="一次性任务; 不传则进入交互模式")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="跳过危险命令确认 (仅用于自动化, 请勿在重要目录使用)",
    )
    parser.add_argument("--window", type=int, default=128_000, help="模型上下文窗口")
    parser.add_argument("--max-chars", type=int, default=8_000, help="单条工具结果字符上限")
    parser.add_argument("-q", "--quiet", action="store_true", help="只输出最终答案")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    load_env(project_root)

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(red(f"工作区不存在: {workspace}"))
        return 2

    render = make_renderer(verbose=not args.quiet)
    if args.quiet:
        render = lambda event, payload: None  # noqa: E731

    try:
        agent = build_agent(
            workspace,
            mode="allow" if args.yolo else "ask",
            window=args.window,
            max_chars=args.max_chars,
            on_event=render,
        )
    except RuntimeError as exc:
        print(red(str(exc)))
        return 2

    def run_once(task: str) -> LoopResultAlias:
        print()
        result = agent.run(task)
        if args.quiet:
            print(result.answer)
        else:
            print()
            print(result.answer)
        print(dim("  " + _stats_line(result, agent)))
        return result

    # ---- 一次性模式 ----
    if args.task:
        return 0 if run_once(args.task).success else 1

    # ---- 交互模式 ----
    print(BANNER.format(ws=workspace, model=agent.llm.model))
    while True:
        try:
            task = input(cyan("miniagent> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not task:
            break
        if task in ("exit", "quit", ":q"):
            break
        try:
            run_once(task)
        except KeyboardInterrupt:
            print(yellow("\n  已中断当前任务"))
        except Exception as exc:  # noqa: BLE001 - REPL 不该因为一次异常退出
            print(red(f"  运行异常: {type(exc).__name__}: {exc}"))
    return 0


# 类型别名: 避免在函数签名里写长类型, 同时保留可读性
LoopResultAlias = Any


def _stats_line(result: Any, agent: AgentLoop) -> str:
    """一行统计。

    这些数字不是给你看着玩的 —— 它们是简历里要写的那几个数字的来源。
    每次跑完都打出来, 就不需要在"要写简历了"的时候回头补测量。
    """
    s = result.stats
    err = agent.budget.error_stats()
    err_txt = ""
    if err.get("samples"):
        # 偏差为正 = 我们估高了 (安全侧); 为负 = 估低了 (危险侧)
        err_txt = (
            f" | 估算偏差 {err['mean_bias']:+.1%}"
            f" (最差低估 {err['max_under']:+.1%}, {int(err['samples'])} 样本)"
        )
    return (
        f"轮数 {s.get('turns')} | 工具调用 {s.get('tool_calls')} | "
        f"峰值水位 {s.get('max_pressure', 0):.1%} | "
        f"裁剪 {s.get('snip_events')} 次 | 截断 {s.get('truncate_events')} 次 | "
        f"结构修复 {s.get('repair_events')} 次{err_txt}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
