"""权限模式与模式矩阵。

这个文件不做真实检查，它只回答一个问题：
“如果前面的所有硬性检查都没有命中，那么当前权限模式对某类工具的默认态度是什么？”

可以把这里理解成权限系统的“最后一张查表表格”。
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from mewcode.tools.base import ToolCategory

# PermissionChecker.check() 最终会返回的三种效果。
DecisionEffect = Literal["allow", "deny", "ask"]


class PermissionMode(str, Enum):
    """权限模式枚举。

    这里继承 str + Enum，目的是让枚举成员既有枚举语义，
    又能直接当作普通字符串参与比较或序列化。
    """

    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    PLAN = "plan"
    BYPASS = "bypassPermissions"
    CUSTOM = "custom"
    DONT_ASK = "dontAsk"


# 模式矩阵：外层是权限模式，内层是工具分类，值是默认决策效果。
_MODE_MATRIX: dict[PermissionMode, dict[ToolCategory, DecisionEffect]] = {
    PermissionMode.DEFAULT: {"read": "allow", "write": "ask", "command": "ask"},
    PermissionMode.ACCEPT_EDITS: {"read": "allow", "write": "allow", "command": "ask"},
    PermissionMode.PLAN: {"read": "allow", "write": "ask", "command": "ask"},
    PermissionMode.BYPASS: {"read": "allow", "write": "allow", "command": "allow"},
    PermissionMode.CUSTOM: {"read": "ask", "write": "ask", "command": "ask"},
    PermissionMode.DONT_ASK: {"read": "allow", "write": "allow", "command": "allow"},
}


def mode_decide(mode: PermissionMode, category: ToolCategory) -> DecisionEffect:
    """根据模式矩阵返回某类工具的默认决策。

    输入:
        mode: 当前权限模式。
        category: 工具分类（read / write / command）。
    输出:
        allow / deny / ask 三者之一。
    """
    return _MODE_MATRIX[mode][category]
