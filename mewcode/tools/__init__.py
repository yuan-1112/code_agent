"""工具注册表及默认工具装配入口。

这个文件的职责不是实现具体工具，而是负责：
1. 管理“当前有哪些工具已经注册”。
2. 控制工具的启用、禁用、延迟发现与协议适配。
3. 提供 create_default_registry()，把默认工具集合一次性装配好。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mewcode.tools.base import Tool

if TYPE_CHECKING:
    from mewcode.cache import FileCache


class ToolRegistry:
    """维护当前会话可用工具的注册表。

    _tools:
        保存“工具名 -> 工具实例”的映射。
    _disabled:
        记录被临时禁用的工具名。
    _discovered:
        记录已经通过 ToolSearch 等机制暴露给模型的延迟工具。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled: set[str] = set()
        self._discovered: set[str] = set()

    def register(self, tool: Tool) -> None:
        """注册一个工具实例。"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """按工具名获取工具实例。"""
        return self._tools.get(name)

    def is_enabled(self, name: str) -> bool:
        """判断某个工具当前是否可用。"""
        return name in self._tools and name not in self._disabled

    def enable(self, name: str) -> None:
        """重新启用某个已注册工具。"""
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        """禁用某个工具，但不从注册表中删除。"""
        if name in self._tools:
            self._disabled.add(name)

    def enable_all(self) -> None:
        """一次性恢复所有工具的可用状态。"""
        self._disabled.clear()

    def mark_discovered(self, name: str) -> None:
        """标记一个延迟工具已经被暴露给模型。"""
        self._discovered.add(name)

    def is_discovered(self, name: str) -> bool:
        """判断某个延迟工具是否已经被发现。"""
        return name in self._discovered

    def get_deferred_tool_names(self) -> list[str]:
        """列出所有尚未发现、但允许延迟加载的工具名。"""
        return [
            name
            for name, tool in self._tools.items()
            if getattr(tool, "should_defer", False)
            and name not in self._discovered
            and name not in self._disabled
        ]

    def search_deferred(
        self, query: str, max_results: int, protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        """按关键词搜索延迟工具，并按相关度返回 schema 列表。

        输入:
            query: 用户或模型提供的搜索关键词。
            max_results: 最多返回多少个工具。
            protocol: 不同模型协议需要不同 schema 结构，这里顺带做协议适配。
        输出:
            供模型直接加载的工具 schema 列表。
        """
        query_lower = query.lower()
        scored: list[tuple[int, str, Tool]] = []
        for name, tool in self._tools.items():
            if not getattr(tool, "should_defer", False):
                continue
            if name in self._disabled:
                continue
            score = 0
            name_lower = name.lower()
            desc_lower = (tool.description or "").lower()
            if query_lower in name_lower:
                score += 10
            if query_lower in desc_lower:
                score += 5
            for word in query_lower.split():
                if word in name_lower:
                    score += 3
                if word in desc_lower:
                    score += 1
            if score > 0:
                scored.append((score, name, tool))
        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for _, _name, tool in scored[:max_results]:
            base = tool.get_schema()
            if protocol in ("openai", "openai-compat"):
                results.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                results.append(base)
        return results

    def find_deferred_by_names(
        self, names: list[str], protocol: str = "anthropic"
    ) -> list[dict[str, Any]]:
        """按工具名直接取回延迟工具 schema。"""
        results: list[dict[str, Any]] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                continue
            if not getattr(tool, "should_defer", False):
                continue
            base = tool.get_schema()
            if protocol in ("openai", "openai-compat"):
                results.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                results.append(base)
        return results

    def list_tools(self) -> list[Tool]:
        """返回当前注册表中的全部工具实例。"""
        return list(self._tools.values())

    def get_all_schemas(self, protocol: str = "anthropic") -> list[dict[str, Any]]:
        """返回当前允许暴露给模型的全部工具 schema。

        这里会自动跳过：
        - 被禁用的工具
        - 尚未发现的延迟工具
        """
        schemas: list[dict[str, Any]] = []
        for name, tool in self._tools.items():
            if name in self._disabled:
                continue
            if getattr(tool, "should_defer", False) and name not in self._discovered:
                continue
            base = tool.get_schema()
            if protocol in ("openai", "openai-compat"):
                schemas.append({
                    "type": "function",
                    "name": base["name"],
                    "description": base["description"],
                    "parameters": base["input_schema"],
                })
            else:
                schemas.append(base)
        return schemas


def create_default_registry(
    file_cache: FileCache | None = None,
    file_history: Any = None,
) -> ToolRegistry:
    """创建默认工具注册表。

    输入:
        file_cache: 可选的文件内容缓存，用于降低重复读文件成本。
        file_history: 可选的文件编辑历史记录器，用于追踪写操作。
    输出:
        一个已经注册好常用文件工具与命令工具的 ToolRegistry。
    """
    from mewcode.tools.bash import Bash
    from mewcode.tools.edit_file import EditFile
    from mewcode.tools.file_state_cache import FileStateCache
    from mewcode.tools.glob import Glob
    from mewcode.tools.grep import Grep
    from mewcode.tools.read_file import ReadFile
    from mewcode.tools.write_file import WriteFile

    # 文件状态缓存负责强制执行“先读后改”和“改前未发生外部变更”两道安全门。
    file_state_cache = FileStateCache()

    registry = ToolRegistry()
    # 这里注册的是最基础的本地开发工具集，供普通代码编辑场景直接使用。
    registry.register(
        ReadFile(file_cache=file_cache, file_state_cache=file_state_cache)
    )
    registry.register(
        WriteFile(
            file_cache=file_cache,
            file_history=file_history,
            file_state_cache=file_state_cache,
        )
    )
    registry.register(
        EditFile(
            file_cache=file_cache,
            file_history=file_history,
            file_state_cache=file_state_cache,
        )
    )
    registry.register(Bash())
    registry.register(Glob())
    registry.register(Grep())
    return registry
