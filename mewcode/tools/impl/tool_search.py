"""延迟工具搜索与加载工具。

这个文件实现的不是业务工具，而是“工具的工具”。
它允许模型在当前工具列表不完整时，按关键词搜索尚未暴露的延迟工具，
再把这些工具的完整 schema 加载进当前会话。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if __import__("typing").TYPE_CHECKING:
    from mewcode.tools import ToolRegistry


class ToolSearchParams(BaseModel):
    """ToolSearch 的输入参数。"""

    query: str
    max_results: int = 5


class ToolSearchTool(Tool):
    """搜索并激活延迟工具。

    这个工具通常配合 ToolRegistry 里的 should_defer 机制使用：
    - 普通工具一开始就可以直接暴露给模型。
    - 延迟工具先不暴露，等模型明确需要时再通过 ToolSearch 拉进来。
    """

    name = "ToolSearch"
    description = (
        "Search for and load additional tools that are not immediately available. "
        "Use query 'select:<name>[,<name>...]' to load specific tools by name, "
        "or provide keywords to search by relevance."
    )
    params_model = ToolSearchParams
    category = "read"
    # ToolSearch 自己必须始终可见，否则模型将无法主动发现其他延迟工具。
    should_defer = False

    def __init__(
        self,
        registry: ToolRegistry,
        protocol: str = "anthropic",
    ) -> None:
        """保存注册表引用和当前模型协议。

        输入:
            registry: 工具注册表，用于查询和标记延迟工具。
            protocol: 当前会话所用模型协议，用于输出对应格式的 schema。
        """
        self._registry = registry
        self._protocol = protocol

    def get_schema(self) -> dict[str, Any]:
        """返回 ToolSearch 自身的 schema。"""
        schema = self.params_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    async def execute(self, params: BaseModel) -> ToolResult:
        """搜索或直接选中延迟工具，并把其 schema 返回给上层。

        输入:
            params: ToolSearchParams，包含搜索词和最大结果数。
        输出:
            ToolResult，内容是可直接加载的新工具 schema 或错误提示。
        """
        assert isinstance(params, ToolSearchParams)
        query = params.query
        max_results = params.max_results

        if query.startswith("select:"):
            # select:toolA,toolB 这种格式表示“跳过搜索，按名字直接加载”。
            names = [n.strip() for n in query[7:].split(",")]
            schemas = self._registry.find_deferred_by_names(names, self._protocol)
        else:
            # 普通关键词搜索会进入注册表的简单打分逻辑。
            schemas = self._registry.search_deferred(
                query, max_results, self._protocol
            )

        if not schemas:
            deferred_names = self._registry.get_deferred_tool_names()
            return ToolResult(
                output=(
                    f'No matching deferred tools for "{query}". '
                    f'Available: {", ".join(deferred_names)}'
                )
            )

        for schema in schemas:
            # 一旦 schema 被返回给上层，就将对应工具标记为“已发现”，
            # 后续 get_all_schemas() 就可以把它们继续带给模型。
            if "name" in schema:
                self._registry.mark_discovered(schema["name"])

        return ToolResult(
            output=(
                f"Found {len(schemas)} tool(s). Their full schemas are now loaded:\n\n"
                f"{json.dumps(schemas, indent=2, ensure_ascii=False)}"
            )
        )
