"""路径沙箱。

这个文件的职责很单一：
判断一次文件读写目标路径是否落在允许的目录范围内。

默认允许：
1. 项目根目录
2. 系统临时目录

可选再追加其他允许目录。
"""

from __future__ import annotations

import tempfile
from pathlib import Path


class PathSandbox:
    """路径沙箱检查器。"""

    def __init__(
        self,
        project_root: str,
        extra_allowed: list[str] | None = None,
    ) -> None:
        """初始化允许访问的根目录集合。

        输入:
            project_root: 项目根目录。
            extra_allowed: 额外允许访问的路径列表。
        """
        root = Path(project_root).resolve()
        self._allowed_roots: list[Path] = [
            root,
            Path(tempfile.gettempdir()).resolve(),
        ]
        if extra_allowed:
            for path in extra_allowed:
                self._allowed_roots.append(Path(path).resolve())

    @property
    def project_root(self) -> Path:
        """返回当前项目根目录。"""
        return self._allowed_roots[0]

    def check(self, path: str) -> tuple[bool, str]:
        """检查目标路径是否处于允许范围内。

        输入:
            path: 待检查的路径，可以是相对路径、绝对路径或包含 ~ 的路径。
        输出:
            (ok, reason)
            - ok 为 True 表示允许访问。
            - ok 为 False 时，reason 说明拒绝原因。
        """
        p = Path(path).expanduser()
        if not p.is_absolute():
            # 相对路径统一视为相对于 project_root。
            p = self.project_root / p
        abs_path = p.absolute()

        try:
            # 路径存在时，严格 resolve，顺带解析符号链接和 .. 逃逸。
            real_path = abs_path.resolve(strict=True)
        except OSError:
            # 路径还不存在时，退化为“从最近存在的祖先目录继续推导”，
            # 这样创建新文件时也能做沙箱校验。
            ancestor = abs_path
            while not ancestor.exists():
                parent = ancestor.parent
                if parent == ancestor:
                    return False, f"无法解析路径: {path}"
                ancestor = parent
            try:
                resolved_ancestor = ancestor.resolve(strict=True)
            except OSError:
                return False, f"无法解析路径: {path}"
            real_path = resolved_ancestor / abs_path.relative_to(ancestor)

        for root in self._allowed_roots:
            try:
                real_path.relative_to(root)
                return True, ""
            except ValueError:
                continue

        return False, f"路径 {path} 超出沙箱范围"
