"""Shell 执行工具。

**为什么这个工具比其他工具危险得多, 以及怎么约束它**:

1. **必须有超时**。不带超时的 subprocess 遇到 `ping` 或等待输入的命令会永久挂起,
   整个 agent 就死在那里了。超时是硬需求, 不是优化。

2. **必须有输出上限**。`cat` 一个几十 MB 的文件、或者跑一个疯狂的循环脚本,
   输出量可以轻松超过整个上下文窗口。这里先截一道 (留着提示信息),
   下游的上下文层再截一道 —— 两道都不能省:
   这里的截断是"不把垃圾运进内存", 下游的截断是"保证进历史的内容受控"。

3. **只执行、不做安全判断**。是否要用户确认由 safety.py 决定。
   职责分离的理由: 安全策略会变 (今天允许 rm 明天不允许),
   执行逻辑不该跟着改; 而且安全判断需要拿"完整命令"来判断,
   放在 loop 里统一做更清楚。

4. **工作目录固定在 workspace**。命令默认在工作区根目录执行,
   避免模型 `cd` 到别处后后续命令全在错误的目录下跑。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from miniagent.tools.base import Tool

# 单次命令的默认超时与上限
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300
# 输出上限: 超过就只留头尾, 中间省略
MAX_OUTPUT_CHARS = 30_000


def _find_bash() -> str | None:
    """在 Windows 上优先找 Git Bash。

    理由: 编码任务里大量习惯性命令 (ls / grep / cat / head) 是 POSIX 的,
    用 cmd.exe 会直接报"不是内部或外部命令", 让模型困惑于"工具为什么不好用"。
    找到 bash 就让体验一致; 找不到再退回系统默认 shell。
    """
    if os.name != "nt":
        return None
    candidates = [
        shutil_which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin\bash.exe"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None


def shutil_which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


class RunShell(Tool):
    name = "run_shell"
    description = (
        "在工作区目录下执行一条 shell 命令并返回输出。"
        "用于运行测试、查看目录、执行脚本、查询 git 状态等。"
        "危险命令(如删除、推送)会先请求用户确认。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令"},
            "timeout": {
                "type": "integer",
                "description": f"超时秒数, 默认 {DEFAULT_TIMEOUT}, 最大 {MAX_TIMEOUT}",
            },
        },
        "required": ["command"],
    }

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()
        self._bash = _find_bash()

    def run(self, command: str, timeout: int = DEFAULT_TIMEOUT, **_: Any) -> str:
        limit = max(1, min(int(timeout), MAX_TIMEOUT))

        # 记录用的是什么 shell: 排查"为什么这条命令能跑那条不能"时非常有用
        if self._bash:
            argv = [self._bash, "-lc", command]
            shell_note = "bash"
        else:
            argv = command
            shell_note = "default"

        try:
            proc = subprocess.run(
                argv,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
                shell=(shell_note == "default"),
            )
        except subprocess.TimeoutExpired:
            return (
                f"[超时] 命令执行超过 {limit} 秒被终止: {command}\n"
                f"如果是长任务, 请拆成更小的步骤, 或显式传更大的 timeout。"
            )
        except OSError as exc:
            return f"[执行失败] {type(exc).__name__}: {exc}"

        out = (proc.stdout or "").rstrip()
        err = (proc.stderr or "").rstrip()
        combined = out
        if err:
            combined = f"{out}\n[stderr]\n{err}" if out else f"[stderr]\n{err}"

        if not combined.strip():
            combined = "(无输出)"

        combined = self._cap(combined)

        status = "成功" if proc.returncode == 0 else f"退出码 {proc.returncode}"
        return f"[{status}] $ {command}\n{combined}"

    @staticmethod
    def _cap(text: str) -> str:
        """输出上限: 头尾各留一半, 中间明确标出省略了多少。

        为什么头尾都留而不是只留尾:
        编译错误和测试结果往往在输出尾部, 但命令的开头常包含
        "在哪个目录、用的什么参数"这类上下文, 只留尾会丢掉它。
        """
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        half = MAX_OUTPUT_CHARS // 2
        omitted = len(text) - 2 * half
        return (
            f"{text[:half]}\n\n"
            f"... [本工具层截断: 省略中间 {omitted} 字符; "
            f"完整输出请重定向到文件后用 read_file 分段查看] ...\n\n"
            f"{text[-half:]}"
        )
