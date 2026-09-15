"""危险操作的拦截策略。

**为什么要单独一层, 而不写在 shell 工具里**:

1. **职责分离**。执行逻辑 (怎么跑命令) 和安全策略 (什么命令该拦) 变化频率完全不同 ——
   安全策略是那种"出了事就要马上改"的东西, 不该跟执行代码纠缠在一起。

2. **判定需要全局视角**。有些操作单看工具没问题, 但结合上下文才危险
   (比如在刚 push 过的分支上 reset)。放在 loop 层统一判断, 后面想加规则不用改工具。

3. **可测试**。策略是一个纯函数: 输入 (工具名, 参数) -> 输出 决策。
   不依赖终端、不依赖模型, 可以直接写单测。

**必须承认的局限**:
这里的判定是**基于正则的模式匹配**, 它永远不完备。
`rm -rf` 能拦, 但 `find . -delete`、`python -c "import shutil; shutil.rmtree('/')"` 拦不住。
所以真实产品里这层是"降低误操作概率", 不是"安全保证" ——
真正的保证是工作区隔离 + 权限系统 + 备份。
把这个局限说出来, 比假装它能防住一切更专业。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

ALLOW = "allow"
ASK = "ask"
DENY = "deny"


@dataclass
class Decision:
    action: str
    reason: str = ""

    @property
    def needs_confirm(self) -> bool:
        return self.action == ASK


# (正则, 说明) —— 命中即需要确认
DANGEROUS_COMMAND_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+", "递归/强制删除"),
    (r"\brm\s+-[a-zA-Z]*r", "递归删除"),
    (r"\b(del|rd|rmdir)\s+/[sq]", "Windows 递归删除"),
    (r"\bgit\s+push\b", "推送到远端仓库"),
    (r"\bgit\s+reset\s+--hard\b", "丢弃本地改动"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*[fd]", "删除未跟踪文件"),
    (r"\bgit\s+checkout\s+--\s", "覆盖本地改动"),
    (r"\bchmod\s+777\b", "开放全部权限"),
    (r"\bchown\b", "修改文件所有者"),
    (r"(curl|wget)[^|]*\|\s*(ba)?sh\b", "下载并直接执行脚本"),
    (r"\b(shutdown|reboot|halt)\b", "关机/重启"),
    (r"\bformat\b|\bmkfs\b", "格式化磁盘"),
    (r">\s*/dev/(sd|hd|nvme)", "直接写块设备"),
    (r"\bdd\s+.*of=/dev/", "直接写块设备"),
    (r"\b(pip|npm|yarn|pnpm)\s+(install|add)\s+-g\b", "安装全局包"),
    (r"\bsudo\b", "提权执行"),
    (r"\btruncate\b|\b:>\s*\w", "清空文件内容"),
    (r"\bkill(all)?\b", "终止进程"),
]

# 命中这些即直接拒绝, 不给确认机会 (可能损坏系统或无法回滚)
DENY_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/\s*$", "删除根目录"),
    (r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+/\*", "删除根目录内容"),
    (r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+~", "删除用户主目录"),
    (r"\bmkfs\b", "格式化文件系统"),
    (r"\bdd\s+if=.*of=/dev/[sh]d", "覆写系统磁盘"),
]


class SafetyPolicy:
    """策略对象。

    mode:
        "ask"   —— 命中危险模式时询问用户 (默认, 推荐)
        "allow" —— 全部放行 (仅用于自动化脚本/验证)
        "deny"  —— 命中即拒绝 (最保守)
    """

    def __init__(self, mode: str = "ask") -> None:
        self.mode = mode
        # 用户选过"总是允许"的命令前缀, 避免同一条命令反复询问
        self._always_allowed: set[str] = set()
        self.confirm_count = 0
        self.deny_count = 0

    # ------------------------------------------------------------------ #

    def decide(self, tool_name: str, arguments: dict[str, Any]) -> Decision:
        """核心判定。纯函数, 不产生副作用 (不打印、不读输入)。

        保持无副作用是为了可测试: 同一组输入必须得到同一个决策。
        """
        if self.mode == "allow":
            return Decision(ALLOW, "策略为放行模式")

        # 只有能执行任意命令的工具需要模式匹配; 其他工具由自身边界保护
        # (文件工具已被工作区路径限制, 见 tools/fs.py 的 _resolve)
        if tool_name != "run_shell":
            return Decision(ALLOW)

        command = str(arguments.get("command", ""))

        for pattern, why in DENY_PATTERNS:
            if re.search(pattern, command):
                return Decision(DENY, f"命中禁止规则: {why}")

        for pattern, why in DANGEROUS_COMMAND_PATTERNS:
            if re.search(pattern, command):
                if self._is_remembered(command):
                    return Decision(ALLOW, f"此前已允许同类操作: {why}")
                if self.mode == "deny":
                    return Decision(DENY, f"策略为拒绝模式, 命中: {why}")
                return Decision(ASK, f"需要确认: {why}")

        return Decision(ALLOW)

    # ------------------------------------------------------------------ #

    def confirm(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        reason: str,
        ask: Callable[[str], str] | None = None,
    ) -> tuple[bool, str]:
        """向用户确认。

        返回 (是否放行, 说明)。
        `ask` 可注入, 便于在自动化测试里替换掉真实终端输入 ——
        这就是把策略与执行分开的好处之一。
        """
        self.confirm_count += 1

        if self.mode == "allow":
            return True, "自动放行"

        command = str(arguments.get("command", ""))
        prompt = (
            f"\n⚠️  {reason}\n"
            f"    工具: {tool_name}\n"
            f"    内容: {command}\n"
            f"    选择: [y] 执行一次  [n] 跳过  [a] 总是允许此类  > "
        )

        reader = ask or input
        try:
            answer = reader(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "用户中断"

        if answer in ("a", "always"):
            self._always_allowed.add(self._key(command))
            return True, "已记住: 此类操作以后不再询问"
        if answer in ("y", "yes", ""):
            return True, "用户确认执行"
        return False, "用户拒绝执行"

    # ------------------------------------------------------------------ #

    @staticmethod
    def _key(command: str) -> str:
        """把命令归一成一个"同类"标识: 取第一个词作为前缀。"""
        return command.strip().split()[0] if command.strip() else ""

    def _is_remembered(self, command: str) -> bool:
        return self._key(command) in self._always_allowed

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "confirm_count": self.confirm_count,
            "deny_count": self.deny_count,
            "always_allowed": sorted(self._always_allowed),
        }
