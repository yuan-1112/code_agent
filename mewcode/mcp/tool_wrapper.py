
from __future__ import annotations

from typing import Any

from mcp import types as mcp_types
from pydantic import BaseModel, create_model

from mewcode.mcp.client import MCPClient
from mewcode.tools.base import Tool, ToolResult


def _build_params_model(
    tool_name: str, input_schema: dict[str, Any]
) -> type[BaseModel]:
    """根据 MCP 工具的 JSON Schema 动态生成 Pydantic 参数模型。

    输入:
        tool_name: 工具名称，用于生成模型类名。
        input_schema: MCP 工具声明的输入 schema。

    输出:
        返回一个继承自 `BaseModel` 的动态模型类。

    说明:
        MCP 工具的参数描述是 JSON Schema，而 MewCode 内部工具执行前希望
        拿到一个可校验、可序列化的 Pydantic 模型。这个函数的作用就是在
        两者之间搭一层转换桥。

        这样处理后，MCP 工具也可以像内置工具一样享受参数必填校验、
        类型转换、默认值处理等能力。
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for name, prop in properties.items():
        # 先把 JSON Schema 的类型字符串转成 Python 类型对象。
        py_type = _json_type_to_python(prop.get("type", "string"))
        if name in required:
            # `...` 在 Pydantic 中表示必填字段。
            field_definitions[name] = (py_type, ...)
        else:
            # 非必填字段统一允许为 None，并把默认值设为 None。
            field_definitions[name] = (py_type | None, None)

    return create_model(f"{tool_name}Params", **field_definitions)


def _json_type_to_python(json_type: str) -> type:
    """把 JSON Schema 的基础类型映射为 Python 类型。

    输入:
        json_type: JSON Schema 中的 `type` 字符串。

    输出:
        返回对应的 Python 类型对象。

    说明:
        这里只处理最常见的基础类型，足够覆盖绝大多数 MCP 工具参数。
        如果遇到未识别类型，则保守退回 `str`，避免因为 schema 写法差异导致
        整个工具无法接入。
    """
    mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    return mapping.get(json_type, str)


def _extract_text(content: list[Any]) -> str:
    """把 MCP 多模态返回内容尽量整理成可展示的文本。

    输入:
        content: `CallToolResult.content` 中的 block 列表。

    输出:
        返回一个适合直接展示给 Agent 或用户的字符串。

    说明:
        MCP 工具返回的不一定只有纯文本，可能还会包含图片、嵌入资源等 block。
        MewCode 当前这层更偏向文本工作流，因此这里采用“文本保留原文，
        非文本保留占位信息”的策略，让上层至少知道返回了什么类型的内容。
    """
    parts: list[str] = []
    for block in content:
        if isinstance(block, mcp_types.TextContent):
            parts.append(block.text)
        elif isinstance(block, mcp_types.ImageContent):
            parts.append(f"[image: {block.mimeType}]")
        elif isinstance(block, mcp_types.EmbeddedResource):
            resource = block.resource
            if hasattr(resource, "text"):
                parts.append(resource.text)
            else:
                parts.append(f"[binary resource: {resource.uri}]")
    return "\n".join(parts) if parts else "(no output)"


class MCPToolWrapper(Tool):
    """把 MCP 原生工具定义适配成 MewCode 内部 `Tool` 接口。

    这个类的存在是因为两套系统的关注点不同：
    - MCP 世界里，工具由服务端通过 schema 声明。
    - MewCode 世界里，工具需要实现统一的 `Tool` 接口，具备名称、描述、
      参数模型、执行入口等属性。

    包装器的职责就是把这两套接口接起来，使 Agent 在调用工具时感知不到
    “这是内置工具还是来自外部 MCP 服务端的工具”。
    """

    def __init__(
        self,
        server_name: str,
        tool_def: mcp_types.Tool,
        client: MCPClient,
    ) -> None:
        """根据 MCP 工具定义创建一个可注册的内部工具对象。

        输入:
            server_name: 当前工具所属的 MCP 服务端名称。
            tool_def: MCP SDK 返回的工具定义对象。
            client: 对应服务端的客户端连接。

        输出:
            无返回值；会完成内部 `Tool` 所需字段初始化。

        关键设计:
            1. `name` 使用 `mcp_<server>_<tool>` 形式，避免和内置工具重名。
            2. `should_defer = True` 表示默认延迟暴露，减少 system prompt 负担。
            3. `is_concurrency_safe = False` 表示默认不承诺并发安全，避免多个请求
               同时操作同一底层连接导致状态错乱。
            4. `params_model` 动态由 MCP schema 生成，保证外部工具也能接入统一的
               参数校验流程。
        """
        self._server_name = server_name
        self._tool_def = tool_def
        self._client = client
        self.name = f"mcp_{server_name}_{tool_def.name}"
        self.description = tool_def.description or tool_def.name
        self.category = "command"
        self.is_concurrency_safe = False
        self.should_defer = True
        self.params_model = _build_params_model(
            tool_def.name, tool_def.inputSchema
        )

    @property
    def mcp_tool_name(self) -> str:
        """返回 MCP 服务端原始工具名。

        输入:
            无。

        输出:
            返回服务端声明的原始工具名，不带 `mcp_` 包装前缀。

        说明:
            内部注册名和 MCP 原始名是两套名字：
            - 内部注册名用于 MewCode 工具系统中避免重名。
            - 原始名用于真正调用 `client.call_tool()` 时与服务端对接。
        """
        return self._tool_def.name


    def get_schema(self) -> dict[str, Any]:
        """返回提供给 MewCode 工具系统的 schema 描述。

        输入:
            无。

        输出:
            返回包含名称、描述、输入 schema 的字典。

        说明:
            这里对 `inputSchema` 基本采取透传策略，让 LLM 看到的参数结构与
            MCP 服务端声明保持一致，减少中间层再加工导致的信息偏差。
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self._tool_def.inputSchema,
        }


    async def execute(self, params: BaseModel) -> ToolResult:
        """执行包装后的 MCP 工具，并把结果转换成内部 `ToolResult`。

        输入:
            params: 已通过 Pydantic 校验的参数模型实例。

        输出:
            返回 MewCode 内部统一的 `ToolResult`。

        主流程说明:
            1. 如果底层客户端已失活，先尝试自动重连。
            2. 将 Pydantic 模型转成普通字典，并去掉值为 `None` 的可选字段。
            3. 调用 MCP 原始工具。
            4. 把多模态 content 提取成文本。
            5. 按 `isError` 标记是否为业务错误。

        错误处理说明:
            - 重连失败：返回错误结果，不向上抛异常。
            - 工具调用异常：把客户端标记为失活，等待下次自动重连。

        这种处理方式更适合 Agent 工作流，因为 Agent 更容易消费“工具执行失败”
        这样的结构化结果，而不是被未捕获异常直接打断整轮推理。
        """
        if not self._client.is_alive:
            try:
                await self._client.connect()
            except Exception as e:
                return ToolResult(
                    output=f"MCP server '{self._server_name}' reconnect failed: {e}",
                    is_error=True,
                )

        try:
            result = await self._client.call_tool(
                # 调用服务端时必须使用原始 MCP 工具名，而不是内部包装后的名字。
                self._tool_def.name, params.model_dump(exclude_none=True)
            )
        except Exception as e:
            # 一旦调用阶段出现异常，通常说明连接状态可能已不可靠，先标记失活。
            self._client._alive = False
            return ToolResult(
                output=f"MCP tool call failed: {e}",
                is_error=True,
            )

        # MCP 返回的是多模态 block 列表，这里统一整理成文本输出给上层。
        text = _extract_text(result.content)
        return ToolResult(output=text, is_error=bool(result.isError))
