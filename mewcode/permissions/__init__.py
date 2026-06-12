"""权限系统公共导出。

这个文件不实现权限逻辑，只负责把 permissions 子包里最常用的类型、
检查器和辅助函数统一导出，方便外部模块直接从 mewcode.permissions 导入。
"""

from mewcode.permissions.checker import Decision, PermissionChecker
from mewcode.permissions.dangerous import DangerousCommandDetector
from mewcode.permissions.modes import DecisionEffect, PermissionMode, mode_decide
from mewcode.permissions.rules import Rule, RuleEngine, extract_content, parse_rule
from mewcode.permissions.sandbox import PathSandbox

__all__ = [
    "Decision",
    "DecisionEffect",
    "DangerousCommandDetector",
    "PathSandbox",
    "PermissionChecker",
    "PermissionMode",
    "Rule",
    "RuleEngine",
    "extract_content",
    "mode_decide",
    "parse_rule",
]
