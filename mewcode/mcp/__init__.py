


from __future__ import annotations

from mewcode.mcp.manager import MCPManager

# 当前子模块对外主入口只暴露 `MCPManager`。
# 外部代码通常不需要直接感知 client / wrapper 等内部实现细节，
# 只要通过管理器完成配置加载、工具注册和连接关闭即可。
__all__ = ["MCPManager"]
