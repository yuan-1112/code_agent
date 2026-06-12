
from __future__ import annotations

import logging

from mewcode.config import MCPServerConfig
from mewcode.mcp.client import MCPClient
from mewcode.mcp.tool_wrapper import MCPToolWrapper
from mewcode.tools import ToolRegistry

logger = logging.getLogger(__name__)


class MCPManager:
    """统一管理多个 MCP 服务端连接与工具注册流程。

    这个类位于 `mewcode.mcp` 这一层的最外侧，职责更偏向“编排”：
    1. 保存 MCP 服务端配置。
    2. 按配置创建 `MCPClient`。
    3. 从每个 MCP 服务端拉取工具定义。
    4. 把工具定义包装成 MewCode 内部统一的 `Tool` 对象并注册。
    5. 在程序结束时统一关闭所有连接。

    这里不会关心单个连接内部到底是 stdio 还是 http，也不会关心
    MCP 返回的工具 schema 细节如何转成内部工具模型；这些细节分别下沉到
    `client.py` 和 `tool_wrapper.py` 中处理。
    """

    def __init__(self) -> None:
        """初始化 MCP 管理器的内部状态。

        输入:
            无。

        输出:
            无返回值；仅创建两个内部字典。

        说明:
            `_configs` 负责保存“服务名 -> 配置”映射。
            `_clients` 负责保存“服务名 -> 已连接客户端”映射。

            这种设计把“静态配置”和“动态连接状态”分开存储，后续读取、
            重连、关闭时都会更清晰。
        """
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}


    def load_configs(self, configs: list[MCPServerConfig]) -> None:
        """加载 MCP 服务端配置列表，并转成按名称索引的字典。

        输入:
            configs: MCP 服务端配置对象列表。

        输出:
            无返回值；配置会写入 `_configs`。

        说明:
            这里没有额外做重名校验，而是采用“后写覆盖前写”的策略。
            这种行为通常适合支持多层配置来源的场景，例如全局配置、
            项目配置、本地覆盖配置逐层叠加时，后面的配置可以覆盖前面同名项。
        """
        for cfg in configs:
            self._configs[cfg.name] = cfg


    async def register_all_tools(self, registry: ToolRegistry) -> list[str]:
        """连接所有已配置的 MCP 服务端，并把其工具统一注册到工具表。

        输入:
            registry: MewCode 的全局工具注册表。

        输出:
            返回一个错误信息列表；每一项代表某个 MCP 服务端初始化失败。

        主流程说明:
            1. 遍历所有 MCP 配置。
            2. 为每个配置创建 `MCPClient` 并建立连接。
            3. 调用 `list_tools()` 拉取该服务端公开的工具定义。
            4. 使用 `MCPToolWrapper` 包装每个工具，使其符合内部 `Tool` 接口。
            5. 将包装后的工具注册到 `ToolRegistry`。

        容错策略:
            单个服务端失败不会中断整个注册流程，而是记录错误并继续处理后续
            服务端。这样可以避免“一个 MCP 挂了，整个 Agent 都起不来”的问题。
        """
        errors: list[str] = []
        for name, config in self._configs.items():
            try:
                # 为当前服务端创建独立客户端，并立即建立连接。
                client = MCPClient(config)
                await client.connect()
                self._clients[name] = client

                # 拉取该服务端暴露的所有 MCP 工具定义。
                tools = await client.list_tools()
                for tool_def in tools:
                    # 包装器负责把 MCP 原生工具转成 MewCode 内部统一工具接口。
                    wrapper = MCPToolWrapper(name, tool_def, client)
                    registry.register(wrapper)
                    logger.info("Registered MCP tool: %s", wrapper.name)

            except Exception as e:
                # 收集错误而不是直接抛出，便于启动阶段统一汇报问题。
                msg = f"MCP server '{name}': {e}"
                logger.warning(msg)
                errors.append(msg)

        return errors


    async def get_client(self, name: str) -> MCPClient | None:
        """按服务名获取 MCP 客户端，必要时执行延迟创建或自动重连。

        输入:
            name: MCP 服务端名称。

        输出:
            返回对应的 `MCPClient`；如果配置中不存在该名称，返回 `None`。

        说明:
            这个方法既是“读取缓存”的入口，也是“懒加载连接”的入口。
            如果客户端尚未创建，则根据 `_configs` 现场创建；
            如果客户端对象还在，但底层连接已经失活，则重新建立连接。

            这里采用“重新 new 一个客户端实例”而不是在旧实例上硬重连，
            是为了避免旧的 `AsyncExitStack` 或会话状态残留，降低资源状态混乱
            的概率。
        """
        client = self._clients.get(name)
        if client is None:
            # 缓存里没有客户端时，尝试从配置中找到对应服务并现场建立连接。
            config = self._configs.get(name)
            if config is None:
                return None
            client = MCPClient(config)
            await client.connect()
            self._clients[name] = client
            return client

        if not client.is_alive:
            logger.info("Reconnecting MCP server '%s'", name)
            # 先关闭旧对象，再创建新对象，避免旧连接残留。
            await client.close()
            client = MCPClient(self._configs[name])
            await client.connect()
            self._clients[name] = client

        return client


    async def shutdown(self) -> None:
        """关闭所有已创建的 MCP 客户端并清空缓存。

        输入:
            无。

        输出:
            无返回值。

        说明:
            这个方法通常在程序退出、Agent 停止或需要整体释放资源时调用。
            即使某个客户端关闭失败，也只记调试日志，不阻断其余客户端清理。
            最后会清空 `_clients`，确保管理器内部不再持有失效引用。
        """
        for name, client in self._clients.items():
            try:
                await client.close()
                logger.info("MCP server '%s' closed", name)
            except Exception:
                logger.debug("Error closing MCP server '%s'", name, exc_info=True)
        self._clients.clear()
