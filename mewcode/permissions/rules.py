"""权限规则引擎。

这个文件负责三件事：
1. 定义 Rule 结构与匹配逻辑。
2. 从 YAML 文件加载权限规则。
3. 按优先级评估 allow / deny 规则。

这里的规则语法故意设计得偏简单，例如：
    Bash(git *)
    ReadFile(src/*.py)

这样普通用户不必写正则，也能快速定制权限行为。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Literal

import yaml

Effect = Literal["allow", "deny"]

# 规则语法形如 ToolName(pattern)。
_RULE_RE = re.compile(r"^(\w+)\((.+)\)$")

# 对不同工具，从参数里抽取不同字段作为“规则匹配内容”。
_CONTENT_FIELDS: dict[str, str] = {
    "Bash": "command",
    "ReadFile": "file_path",
    "WriteFile": "file_path",
    "EditFile": "file_path",
    "Glob": "pattern",
    "Grep": "pattern",
}


@dataclass(frozen=True)
class Rule:
    """一条权限规则。

    tool_name:
        要匹配的工具名。
    pattern:
        用 fnmatch 语法匹配的内容模式。
    effect:
        规则命中后是 allow 还是 deny。
    """

    tool_name: str
    pattern: str
    effect: Effect

    def matches(self, tool_name: str, content: str) -> bool:
        """判断当前规则是否命中一条工具调用。"""
        if self.tool_name != tool_name:
            return False
        return fnmatch(content, self.pattern)


def parse_rule(raw: str, effect: Effect) -> Rule:
    """把一条字符串规则解析成 Rule 对象。

    输入:
        raw: 形如 Bash(git *) 的规则文本。
        effect: allow 或 deny。
    输出:
        Rule 对象。
    """
    match = _RULE_RE.match(raw.strip())
    if not match:
        raise ValueError(f"无效的规则语法: {raw}")
    return Rule(tool_name=match.group(1), pattern=match.group(2), effect=effect)


def extract_content(tool_name: str, arguments: dict[str, Any]) -> str:
    """从工具参数中抽取供规则与权限系统匹配的核心内容。

    输入:
        tool_name: 工具名称。
        arguments: 工具参数字典。
    输出:
        对应工具的核心匹配内容；若没有映射，则返回空字符串。
    """
    field = _CONTENT_FIELDS.get(tool_name)
    if field is None:
        return ""
    return str(arguments.get(field, ""))


def _load_rules_file(path: Path) -> list[Rule]:
    """从单个 YAML 文件加载规则列表。

    输入:
        path: 规则文件路径。
    输出:
        解析成功的 Rule 列表；任何无效项都会被跳过。
    """
    if not path.is_file():
        return []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    rules: list[Rule] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        rule_str = entry.get("rule", "")
        effect = entry.get("effect", "")
        if effect not in ("allow", "deny"):
            continue
        try:
            rules.append(parse_rule(rule_str, effect))
        except ValueError:
            # 单条规则解析失败时只跳过该条，不影响整个权限系统启动。
            continue
    return rules


class RuleEngine:
    """三层权限规则引擎。"""

    def __init__(
        self,
        user_rules_path: Path | None = None,
        project_rules_path: Path | None = None,
        local_rules_path: Path | None = None,
    ) -> None:
        """保存三层规则文件路径。"""
        self._user_path = user_rules_path
        self._project_path = project_rules_path
        self._local_path = local_rules_path

    def _load_tiers(self) -> list[list[Rule]]:
        """加载三层规则文件。

        返回顺序固定为：
        1. 用户级
        2. 项目级
        3. 本地级
        """
        tiers: list[list[Rule]] = []
        for path in (self._user_path, self._project_path, self._local_path):
            tiers.append(_load_rules_file(path) if path else [])
        return tiers

    def evaluate(self, tool_name: str, content: str) -> Effect | None:
        """评估某次工具调用是否命中权限规则。

        输入:
            tool_name: 当前工具名。
            content: 从参数中抽取出的核心匹配内容。
        输出:
            allow / deny / None。
        """
        for rules in self._load_tiers():
            # 同一层内采用“后写覆盖前写”的策略，因此倒序匹配。
            for rule in reversed(rules):
                if rule.matches(tool_name, content):
                    return rule.effect
        return None

    def append_local_rule(self, rule: Rule) -> None:
        """向本地级规则文件追加一条新规则。

        这个方法常用于“HITL 里用户点了始终允许/始终拒绝”之后，
        把用户刚做出的选择持久化到本地规则文件。
        """
        if self._local_path is None:
            return
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_rules_file(self._local_path)
        existing.append(rule)
        entries = [
            {"rule": f"{item.tool_name}({item.pattern})", "effect": item.effect}
            for item in existing
        ]
        self._local_path.write_text(
            yaml.dump(entries, allow_unicode=True),
            encoding="utf-8",
        )
