
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any

import httpx
from mcp import ClientSession, types
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from mewcode.config import MCPServerConfig, build_child_env, resolve_env_vars

logger = logging.getLogger(__name__)


class MCPClient:
    """封装单个 MCP 服务端的连接、会话与资源生命周期。

    这个类是 `manager.py` 与底层 MCP SDK 之间的桥梁，主要负责三件事：
    1. 根据配置建立传输层连接，支持 stdio 与 streamable HTTP 两种模式。
    2. 持有 MCP SDK 的 `ClientSession`，对外提供列出工具、调用工具等能力。
    3. 统一管理连接期间创建的异步资源，确保异常和关闭阶段都能正确清理。

    从分层角度看：
    - `MCPManager` 负责“管理多个客户端”。
    - `MCPClient` 负责“管理一个客户端连接”。
    - `MCPToolWrapper` 负责“把 MCP 工具接进内部 Tool 体系”。
    """

    def __init__(self, config: MCPServerConfig) -> None:
        """根据配置创建一个尚未连接的 MCP 客户端对象。

        输入:
            config: 单个 MCP 服务端的配置对象。

        输出:
            无返回值；仅初始化内部状态字段。

        字段说明:
            `config`: 原始配置，后续连接方式、地址、命令都从这里读取。
            `name`: 服务端名称，主要用于日志与错误信息。
            `_session`: MCP SDK 的会话对象；真正的协议调用都通过它完成。
            `_stack`: `AsyncExitStack`，用于统一托管连接期内创建的异步资源。
            `_alive`: 当前连接是否可用的快速标记位。
        """
        self.config = config
        self.name = config.name
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._alive = False


    @property
    def is_alive(self) -> bool:
        """返回当前客户端连接是否处于可用状态。

        输入:
            无。

        输出:
            `True` 表示当前连接被标记为存活，`False` 表示未连接或已失效。

        说明:
            这里使用 `@property` 而不是普通方法，表示这是一个“状态读取”
            行为，不应带有副作用。调用方可以用 `client.is_alive` 直接判断，
            写法上更像读取对象属性。
        """
        return self._alive


    async def connect(self) -> None:
        """建立与 MCP 服务端的连接，并初始化协议会话。

        输入:
            无；连接所需信息全部来自 `self.config`。

        输出:
            无返回值；连接成功后会更新 `_session`、`_stack`、`_alive`。

        主流程说明:
            1. 如果当前已经存活，直接返回，避免重复连接。
            2. 创建 `AsyncExitStack`，作为本次连接生命周期的资源托管中心。
            3. 根据配置决定使用 stdio 还是 http 传输层。
            4. 基于读写流创建 `ClientSession`。
            5. 调用 `session.initialize()` 完成 MCP 协议初始化握手。
            6. 标记连接成功。

        异常处理说明:
            只要中途任意一步失败，就调用 `_cleanup_stack()` 释放已创建资源，
            避免出现半连接、子进程泄漏、http 客户端未关闭等问题。
        """
        if self._alive:
            return

        # ExitStack 相当于“资源回收登记簿”，后面创建的资源都会挂到这里。
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        try:
            # 配置层已经决定了连接类型，这里只负责按类型分派到具体实现。
            if self.config.is_stdio:
                read, write = await self._connect_stdio()
            else:
                read, write = await self._connect_http()

            # MCP SDK 通过 read/write 抽象统一底层传输，stdio/http 都能复用。
            session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            # initialize 是 MCP 会话的握手阶段，成功后才能 list_tools / call_tool。
            await session.initialize()
            self._session = session
            self._alive = True
            logger.info("MCP server '%s' connected", self.name)
        except Exception:
            await self._cleanup_stack()
            raise


    async def _connect_stdio(self) -> tuple[Any, Any]:
        """通过子进程 stdio 方式连接 MCP 服务端。

        输入:
            无；所需命令、参数、环境变量来自 `self.config`。

        输出:
            返回 MCP SDK 需要的 `(read, write)` 读写端。

        说明:
            这种模式通常用于“本地启动一个 MCP server 进程，然后通过标准输入/
            标准输出通信”的场景。命令本身并不在这里解释业务含义，这里只负责
            把配置转成 SDK 认识的 `StdioServerParameters`。
        """
        assert self._stack is not None
        assert self.config.command is not None

        params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=build_child_env(self.config.env),
        )

        # 某些 MCP 服务端会向 stderr 打日志，这里把它导向 devnull，
        # 避免污染主程序标准输出；同时把文件句柄也纳入统一清理流程。
        devnull = open(os.devnull, "w")
        self._stack.callback(devnull.close)
        read, write = await self._stack.enter_async_context(
            stdio_client(params, errlog=devnull)
        )
        return read, write

    async def _connect_http(self) -> tuple[Any, Any]:
        """通过 streamable HTTP 方式连接 MCP 服务端。

        输入:
            无；URL 与请求头来自 `self.config`。

        输出:
            返回 MCP SDK 需要的 `(read, write)` 读写端。

        说明:
            HTTP 模式的重点不在“自己发 REST 请求”，而是借助 MCP SDK 提供的
            `streamable_http_client` 建立一条符合 MCP 协议的数据通道。
            这里先创建 `httpx.AsyncClient`，再交给 MCP SDK 使用。
        """
        assert self._stack is not None
        assert self.config.url is not None

        # header 中可能包含 ${TOKEN} 这类环境变量占位符，连接前先解析。
        resolved_headers = {
            k: resolve_env_vars(v) for k, v in self.config.headers.items()
        }
        http_client = httpx.AsyncClient(
            headers=resolved_headers,
            follow_redirects=True,
        )
        # 把 http 客户端加入 ExitStack，确保关闭连接时一并释放。
        await self._stack.enter_async_context(http_client)

        result = await self._stack.enter_async_context(
            streamable_http_client(self.config.url, http_client=http_client)
        )
        read, write = result[0], result[1]
        return read, write


    async def list_tools(self) -> list[types.Tool]:
        """从 MCP 服务端拉取当前可用工具定义列表。

        输入:
            无。

        输出:
            返回 `types.Tool` 列表，每个元素描述一个 MCP 工具的 schema。

        说明:
            这个方法只负责“把服务端声明的工具定义拿回来”，并不直接注册到
            MewCode。真正的注册由 `MCPManager.register_all_tools()` 完成。
        """
        assert self._session is not None
        result = await self._session.list_tools()
        # 转成普通 list，避免调用方依赖 SDK 返回对象的内部结构。
        return list(result.tools)


    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        """调用 MCP 服务端上的某个工具。

        输入:
            name: MCP 原始工具名，不带内部包装前缀。
            arguments: 传给工具的参数字典。

        输出:
            返回 MCP SDK 的 `CallToolResult`，其中包含 content 与 isError 等字段。

        说明:
            这个方法保持尽量薄，只做协议调用转发，不做结果格式转换。
            结果如何提取成文本、如何转成内部 `ToolResult`，交给上层包装器处理。
        """
        assert self._session is not None
        return await self._session.call_tool(name, arguments)

    async def close(self) -> None:
        """关闭当前 MCP 客户端并释放其关联资源。

        输入:
            无。

        输出:
            无返回值。

        说明:
            这里先把 `_alive` 置为 `False`、把 `_session` 清空，再执行资源释放。
            这样即使关闭过程和其他逻辑存在时序交叠，外部也不会再把该连接误判为
            仍然可用。
        """
        self._alive = False
        self._session = None
        await self._cleanup_stack()

    async def _cleanup_stack(self) -> None:
        """关闭 `AsyncExitStack` 中登记的全部资源。

        输入:
            无。

        输出:
            无返回值；结束后 `_stack` 会被置空。

        说明:
            `AsyncExitStack` 会按后进先出的顺序释放资源，这很适合处理
            “先创建 http client，再创建 stream，再创建 session”这类多层嵌套资源。
            关闭时只需要统一退出 stack，不需要手动记住每一层如何回收。
        """
        if self._stack is not None:
            try:
                # 传入三个 None 表示正常退出上下文，而不是带异常退出。
                await self._stack.__aexit__(None, None, None)
            except RuntimeError as e:
                # 某些异步关闭场景会出现 cancel scope 提示，这里按预期关闭处理。
                if "cancel scope" in str(e):
                    logger.debug("Cancel scope cleanup (expected during shutdown): %s", e)
                else:
                    raise
            except Exception:
                logger.debug("Error closing stack for '%s'", self.name, exc_info=True)
            self._stack = None
