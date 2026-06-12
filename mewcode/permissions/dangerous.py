"""危险命令检测与安全只读命令白名单。

这个文件负责 command 类工具的两道早期过滤：
1. 把明显安全的只读命令快速放行。
2. 把明显危险的命令模式快速拒绝。

它不关心权限模式、规则引擎或路径沙箱，
只专注于“这条 shell 命令本身看起来安不安全”。
"""

from __future__ import annotations

import re

# 危险命令模式列表。
# 每条规则由“预编译正则 + 中文原因”组成，命中后直接拒绝。
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/\s*$"), "递归强制删除根目录"),
    (re.compile(r"mkfs\."), "格式化磁盘"),
    (re.compile(r"dd\s+if=.*of=/dev/"), "直接写磁盘设备"),
    (re.compile(r"chmod\s+-R\s+777\s+/"), "递归修改根目录权限"),
    (re.compile(r":\(\)\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"curl\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r"wget\s+.*\|\s*(ba)?sh"), "管道执行远程脚本"),
    (re.compile(r">\s*/dev/sd"), "覆盖磁盘设备"),
]


# 安全只读命令白名单。
# 这些命令大多只读、不修改文件系统，适合在 default 模式下直接放行。
_SAFE_COMMANDS = frozenset({
    "ls", "dir", "pwd", "echo", "cat", "head", "tail", "wc",
    "find", "which", "whereis", "whoami", "hostname", "uname",
    "date", "cal", "uptime", "df", "du", "free", "env", "printenv",
    "file", "stat", "readlink", "realpath", "basename", "dirname",
    "sort", "uniq", "tr", "cut", "awk", "sed", "grep", "egrep", "fgrep",
    "diff", "comm", "tee", "xargs", "true", "false", "test",
    "git status", "git log", "git diff", "git show", "git branch",
    "git tag", "git remote", "git rev-parse", "git ls-files",
    "git blame", "git stash list", "go version", "go env",
    "node -v", "npm -v", "npx", "python --version", "pip list",
    "cargo --version", "rustc --version", "java -version", "java --version",
})


def is_safe_command(command: str) -> bool:
    """判断一条命令是否属于安全只读命令。

    输入:
        command: 待检查的 shell 命令。
    输出:
        bool，表示是否可被视为安全只读。
    """
    trimmed = command.strip()
    if not trimmed:
        return False

    # 先排除带管道、重定向、命令替换等组合能力的命令，
    # 避免“表面看似安全，实际后半段很危险”的情况。
    for ch in ("|", ";", "&&", ">", "$(", "`"):
        if ch in trimmed:
            return False

    # 允许“完整匹配白名单命令”或“白名单命令 + 参数”的形式。
    for safe in _SAFE_COMMANDS:
        if trimmed == safe or trimmed.startswith(safe + " "):
            return True
    return False


class DangerousCommandDetector:
    """危险命令检测器。"""

    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None) -> None:
        """初始化危险命令模式列表。

        输入:
            extra_patterns: 额外追加的正则模式与原因说明。
        """
        self._patterns = list(_DANGEROUS_PATTERNS)
        if extra_patterns:
            for regex_str, reason in extra_patterns:
                self._patterns.append((re.compile(regex_str), reason))

    def detect(self, command: str) -> tuple[bool, str]:
        """检测命令是否命中危险模式。

        输入:
            command: 待检查的 shell 命令。
        输出:
            (hit, reason)
            - hit 为 True 表示命中危险模式。
            - reason 为对应的人类可读原因。
        """
        for pattern, reason in self._patterns:
            if pattern.search(command):
                return True, reason
        return False, ""
