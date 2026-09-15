"""文件系统工具。

**四个设计取舍** (都是"能改"和"改对"之间的区别, 面试可讲):

1. **读取时带行号**。
   模型后续要用 edit 改文件, 而 edit 要求精确匹配原文。
   带行号能让它准确定位, 同时让"读了哪些行"变得可见 —— 否则模型不知道
   自己看到的是全文件还是片段, 容易基于残缺内容做判断。

2. **默认限制单次读取行数** (MAX_READ_LINES)。
   不限制的话, 模型很容易一次要求读整个大文件, 直接撞上下游的截断逻辑。
   在源头控制比在下游截断更省事 —— 但**两者都要有**:
   源头防不住模型显式传一个很大的 limit, 那是下游截断要兜的。

3. **edit 要求 old_string 在文件中唯一**。
   如果不唯一就报错并说明出现了几次, 而不是随手改第一处。
   理由: 文件改动是"放大器" —— 改错一处, 后续所有基于它的操作全错,
   而且错误会以"模型看懂了但结果不对"的形式表现出来, 极难排查。

4. **所有路径必须落在工作区内** (越界直接报错)。
   这是安全边界: 模型可以读项目代码, 但不该随手读到系统文件。
   报错要明确 (告诉它越界了), 不要静默放行也不要假装文件不存在 ——
   静默会让模型以为是路径写错, 反复重试同一个越界路径。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from miniagent.tools.base import Tool

# 单次读取的默认与上限行数 —— 上限存在的意义见模块文档第 2 条
MAX_READ_LINES = 400
HARD_READ_LIMIT = 2000

# grep 默认返回的最大匹配数, 防止一次搜索撑爆结果
MAX_GREP_MATCHES = 80


def _resolve(path: str, workspace: Path) -> Path:
    """把用户/模型给的路径解析为工作区内的绝对路径。

    越界时抛 ValueError —— 由 ToolRegistry 兜住并转成可读错误文本。
    """
    raw = Path(path).expanduser()
    target = raw if raw.is_absolute() else (workspace / raw)
    target = target.resolve()
    root = workspace.resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ValueError(
            f"路径越界: {target} 不在工作区 {root} 内。"
            f"请只操作工作区内的文件 (用相对路径)。"
        ) from None
    return target


class _WorkspaceTool(Tool):
    """带工作区的工具基类。"""

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).resolve()


class ReadFile(_WorkspaceTool):
    name = "read_file"
    description = (
        "读取工作区内某个文本文件的内容, 返回带行号的文本。"
        "适合查看代码、配置、日志。大文件请用 offset/limit 分次读取, "
        "不要一次读取整个大文件。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作区的文件路径"},
            "offset": {
                "type": "integer",
                "description": "起始行号 (从 1 开始), 默认 1",
            },
            "limit": {
                "type": "integer",
                "description": f"最多读取行数, 默认 {MAX_READ_LINES}",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES, **_: Any) -> str:
        target = _resolve(path, self.workspace)
        if not target.exists():
            # 自己捕获可预期错误, 给出比异常更有用的提示 (列出同目录有什么)
            parent = target.parent
            hint = ""
            if parent.exists() and parent.is_dir():
                siblings = [p.name for p in sorted(parent.iterdir())[:15]]
                hint = f" 该目录下现有: {', '.join(siblings)}"
            return f"[文件不存在] {path}{hint}"
        if target.is_dir():
            return f"[是目录不是文件] {path}。请用 glob 列出目录内容。"

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"[读取失败] {path}: {exc}"

        total = len(lines)
        start = max(1, int(offset))
        # 硬上限优先于调用方参数: 模型可以要 100000 行, 但不能真的给它
        count = min(int(limit), HARD_READ_LIMIT)
        chunk = lines[start - 1 : start - 1 + count]

        body = "\n".join(f"{start + i:>6}\t{line}" for i, line in enumerate(chunk))
        end = start - 1 + len(chunk)
        header = f"[{path}] 共 {total} 行, 显示 {start}-{end} 行"
        return f"{header}\n{body}"


class WriteFile(_WorkspaceTool):
    name = "write_file"
    description = (
        "把内容整体写入一个新文件。已存在的文件会被覆盖 —— "
        "修改已有文件请优先用 edit_file, 避免覆盖掉你不知道的内容。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作区的文件路径"},
            "content": {"type": "string", "description": "要写入的完整内容"},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str, **_: Any) -> str:
        target = _resolve(path, self.workspace)
        existed = target.exists()
        if existed:
            # 覆盖是有风险的操作: 在结果里明确写出来, 让模型在下一轮能意识到
            old_lines = len(
                target.read_text(encoding="utf-8", errors="replace").splitlines()
            )
            note = f"[警告] 该文件已存在 (原有 {old_lines} 行), 已被整体覆盖。"
        else:
            note = ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        new_lines = len(content.splitlines())
        verb = "覆盖" if existed else "创建"
        return f"已{verb} {path} ({new_lines} 行)。{note}".strip()


class EditFile(_WorkspaceTool):
    name = "edit_file"
    description = (
        "在文件中精确替换一段文本。old_string 必须在文件中**唯一出现**, "
        "否则会报错 —— 这是为了防止改错位置。替换后请重新读取文件确认。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对于工作区的文件路径"},
            "old_string": {"type": "string", "description": "要被替换的原文, 必须唯一"},
            "new_string": {"type": "string", "description": "替换成的新内容"},
        },
        "required": ["path", "old_string", "new_string"],
    }

    def run(self, path: str, old_string: str, new_string: str, **_: Any) -> str:
        target = _resolve(path, self.workspace)
        if not target.exists():
            return f"[文件不存在] {path}"
        text = target.read_text(encoding="utf-8", errors="replace")
        count = text.count(old_string)
        if count == 0:
            return (
                f"[未找到匹配] 在 {path} 中找不到 old_string。"
                f"请先 read_file 确认原文 (注意空格和缩进必须完全一致)。"
            )
        if count > 1:
            return (
                f"[匹配不唯一] old_string 在 {path} 中出现了 {count} 次, "
                f"拒绝执行。请提供更长的上下文让它唯一。"
            )
        target.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return f"已修改 {path} (替换 1 处)。"


class GlobFiles(_WorkspaceTool):
    name = "glob"
    description = (
        "按通配符查找文件路径, 如 '**/*.py' 或 'src/*.md'。"
        "用于了解项目结构, 返回按路径排序的结果。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "glob 模式, 如 **/*.py"},
        },
        "required": ["pattern"],
    }

    # 这些目录永远不该被遍历: 体积大且无信息量
    _SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}

    def run(self, pattern: str, **_: Any) -> str:
        matches: list[str] = []
        for p in self.workspace.glob(pattern):
            if not p.is_file():
                continue
            if any(part in self._SKIP_DIRS for part in p.parts):
                continue
            matches.append(str(p.relative_to(self.workspace)).replace("\\", "/"))
        matches.sort()
        if not matches:
            return f"[无匹配] 模式 {pattern!r} 没有找到文件。"
        head = matches[:100]
        body = "\n".join(head)
        more = f"\n... 共 {len(matches)} 个, 只显示前 100 个" if len(matches) > 100 else ""
        return f"[glob {pattern}] 命中 {len(matches)} 个文件:\n{body}{more}"


class GrepFiles(_WorkspaceTool):
    name = "grep"
    description = (
        "在工作区的文件中按关键词或正则搜索, 返回匹配的文件:行号:内容。"
        "找函数定义、变量引用、报错信息时比逐个读文件高效得多。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "要搜索的文本或正则"},
            "glob": {
                "type": "string",
                "description": "限定文件范围, 如 '*.py', 默认搜索常见文本文件",
            },
        },
        "required": ["pattern"],
    }

    _DEFAULT_GLOB = "*.{py,md,txt,toml,json,yaml,yml,cfg,ini,js,ts,tsx,jsx,html,css}"

    def run(self, pattern: str, glob: str | None = None, **_: Any) -> str:
        # 用 ripgrep 优先 (快), 没有则回退到 Python 实现
        rg = self._try_ripgrep(pattern, glob)
        if rg is not None:
            return rg
        return self._python_grep(pattern, glob)

    def _try_ripgrep(self, pattern: str, glob: str | None) -> str | None:
        import shutil

        if not shutil.which("rg"):
            return None
        cmd = ["rg", "-n", "--no-heading", "--color", "never",
               "--glob", glob or self._DEFAULT_GLOB, pattern, str(self.workspace)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode not in (0, 1):
            return None  # 让 Python 实现兜底
        return self._format(proc.stdout, pattern)

    def _python_grep(self, pattern: str, glob: str | None) -> str:
        import re

        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        g = glob or self._DEFAULT_GLOB
        hits: list[str] = []
        for p in self.workspace.rglob("*"):
            if not p.is_file():
                continue
            if any(part in GlobFiles._SKIP_DIRS for part in p.parts):
                continue
            if not p.match(g):
                continue
            try:
                for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = str(p.relative_to(self.workspace)).replace("\\", "/")
                        hits.append(f"{rel}:{i}:{line.strip()[:160]}")
                        if len(hits) >= MAX_GREP_MATCHES * 3:
                            break
            except OSError:
                continue
        return self._format("\n".join(hits), pattern)

    def _format(self, raw: str, pattern: str) -> str:
        lines = [l for l in raw.splitlines() if l.strip()]
        # rg 输出的是绝对路径, 转成相对路径更省 token
        root = str(self.workspace).replace("\\", "/") + "/"
        lines = [l.replace(root, "").replace("\\", "/") for l in lines]
        if not lines:
            return f"[无匹配] 没有找到 {pattern!r}。"
        kept = lines[:MAX_GREP_MATCHES]
        body = "\n".join(kept)
        more = f"\n... 共 {len(lines)} 条匹配, 只显示前 {MAX_GREP_MATCHES} 条" if len(lines) > len(kept) else ""
        return f"[grep {pattern}] 命中 {len(lines)} 条:\n{body}{more}"
