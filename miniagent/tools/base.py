"""工具系统: 统一契约 + 注册表。

**三个必须约束住的地方** (每一条都是踩过坑才知道的, 面试可以直接讲):

1. **工具执行失败不抛异常, 返回错误文本**。
   工具失败是常态 —— 文件不存在、命令没权限、执行超时。
   如果失败就中断循环, 模型就失去了自我纠正的机会 (它看到错误后本可以换个路径重试)。
   所以: 错误是"结果"的一种形态, 而不是流程的中断信号。

2. **工具返回值必须是字符串**。
   它最终要作为 role="tool" 消息的内容进入对话历史, 而历史里的内容就是文本。
   如果这里返回结构化对象, 那么从工具到历史的整条路径上都要反复做序列化转换,
   迟早会在某个环节出现"一半是对象一半是文本"的不一致。

3. **arguments 解析必须容错**。
   模型偶尔会吐出非法 JSON (少引号、多逗号、带 markdown 代码块标记)。
   直接 json.loads 抛异常, 一次异常就白费一整轮对话。
   容错失败时也要返回**可读的错误说明**, 让模型知道该怎么改。

对应地, 工具自己不需要处理"失败了怎么办" —— 那是注册表统一负责的。
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tool(ABC):
    """所有工具的基类。

    子类只需声明三件事: 名字、描述 (给模型看的)、参数 schema。
    描述的质量直接决定模型会不会用对工具 —— 它不是给人看的注释, 是**提示词的一部分**。
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    def schema(self) -> dict[str, Any]:
        """转成 OpenAI tool calling 需要的格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """执行工具, 返回**字符串**。

        允许抛异常: 注册表会捕获并转成错误文本。
        但更好的做法是自己捕获可预期的错误 (如文件不存在), 给出更有用的提示。
        """
        raise NotImplementedError


class ToolRegistry:
    """工具注册表: 负责 schema 汇总、参数解析、执行与错误兜底。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # 执行统计, verify.py 用
        self.call_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("工具必须有 name")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        """给 LLM 的工具清单。每次请求都要带上 —— 所以工具越多, 固定开销越大。"""
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, arguments: str | dict[str, Any] | None) -> str:
        """执行工具, **永远返回字符串**。

        这是 loop 唯一需要调用的入口 —— 它保证了:
        - 参数解析失败有可读提示
        - 工具内部异常不会炸穿循环
        - 结果一定可以被安全地放进消息历史
        """
        self.call_counts[name] = self.call_counts.get(name, 0) + 1

        tool = self._tools.get(name)
        if tool is None:
            self.error_counts[name] = self.error_counts.get(name, 0) + 1
            return (
                f"[工具不存在] 没有名为 {name!r} 的工具。"
                f"可用工具: {', '.join(self._tools.keys())}"
            )

        # --- 参数解析 (容错) ---
        if isinstance(arguments, str):
            text = arguments.strip()
            # 模型有时会包一层 markdown 代码块
            if text.startswith("```"):
                text = text.strip("`")
                if text.lstrip().lower().startswith("json"):
                    text = text.lstrip()[4:]
                text = text.strip()
            if not text:
                kwargs: dict[str, Any] = {}
            else:
                try:
                    kwargs = json.loads(text)
                except json.JSONDecodeError as exc:
                    self.error_counts[name] = self.error_counts.get(name, 0) + 1
                    return (
                        f"[参数解析失败] 工具 {name} 收到的参数不是合法 JSON: {exc.msg}。"
                        f"原始内容前 200 字符: {text[:200]}"
                    )
                if not isinstance(kwargs, dict):
                    return f"[参数解析失败] 工具 {name} 的参数必须是 JSON 对象。"
        elif isinstance(arguments, dict):
            kwargs = arguments
        elif arguments is None:
            kwargs = {}
        else:
            return f"[参数解析失败] 无法处理的参数类型: {type(arguments).__name__}"

        # --- 执行 ---
        try:
            result = tool.run(**kwargs)
        except TypeError as exc:
            # 参数名不匹配是最常见的模型错误, 单独给提示
            self.error_counts[name] = self.error_counts.get(name, 0) + 1
            return (
                f"[参数错误] 工具 {name} 的参数不匹配: {exc}。"
                f"请检查参数名是否与 schema 一致。"
            )
        except Exception as exc:  # noqa: BLE001 - 这里就是要兜住一切
            self.error_counts[name] = self.error_counts.get(name, 0) + 1
            return (
                f"[工具执行失败] {name}: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=3)}"
            )

        # --- 强制字符串 ---
        if result is None:
            return "[无输出]"
        if not isinstance(result, str):
            return str(result)
        return result


def build_default_registry(workspace: Path) -> ToolRegistry:
    """装配默认工具集。

    **刻意只有 6 个工具**。理由:
    工具越多, 每次请求要带上去的 schema 越大 (固定开销, 且不可裁剪)。
    真正决定 Agent 好不好用的是"上下文管得好不好", 不是工具数量。

    workspace 是通过构造函数注入的, 不是全局变量 —— 这样测试时可以把工作区
    指到一个临时目录, 而不用改动进程状态。文件类工具靠它来判定"路径是否越界"。
    """
    from miniagent.tools.fs import EditFile, GlobFiles, GrepFiles, ReadFile, WriteFile
    from miniagent.tools.shell import RunShell

    registry = ToolRegistry()
    for tool in (
        ReadFile(workspace),
        WriteFile(workspace),
        EditFile(workspace),
        GlobFiles(workspace),
        GrepFiles(workspace),
        RunShell(workspace),
    ):
        registry.register(tool)
    return registry
