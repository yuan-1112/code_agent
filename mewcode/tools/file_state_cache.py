"""文件状态缓存。

这个文件专门服务于 ReadFile / WriteFile / EditFile 这一条链路。
它的核心目标是避免模型在“没有读取最新文件内容”或“文件已被外部修改”
的情况下盲目改文件。
"""

from __future__ import annotations

from pathlib import Path


class FileStateCache:
    """跟踪“某个文件最近一次被读取时的状态”。

    存储结构:
        {绝对路径: (读取时的文本内容, 读取时的 mtime_ns)}

    典型流程:
    1. ReadFile 成功读取文件后，调用 record() 记录状态。
    2. WriteFile / EditFile 在写之前，调用 check() 校验安全性。
    3. 写成功后，调用 update() 把缓存刷新成最新状态。
    """

    def __init__(self) -> None:
        """初始化内存中的文件状态缓存。"""
        self._cache: dict[str, tuple[str, int]] = {}

    def record(self, path: str, content: str, mtime_ns: int) -> None:
        """记录一次成功读取后的文件状态。

        输入:
            path: 文件绝对路径。
            content: 读取到的完整文本内容。
            mtime_ns: 读取时的纳秒级修改时间。
        输出:
            无。内部缓存会被更新。
        """
        self._cache[path] = (content, mtime_ns)

    def check(self, path: str) -> tuple[bool, str]:
        """检查某个文件当前是否允许继续写入或编辑。

        输入:
            path: 目标文件的绝对路径。
        输出:
            (ok, error_message)
            - ok 为 True 表示可以继续写。
            - ok 为 False 表示应先重新读取文件。
        """
        entry = self._cache.get(path)
        if entry is None:
            return False, "Error: file has not been read yet. Read it first before editing."

        _, cached_mtime_ns = entry
        try:
            current_mtime_ns = Path(path).stat().st_mtime_ns
        except OSError:
            # 文件可能已经不存在。
            # 对于 WriteFile，这种情况允许继续，让后续写操作负责创建文件。
            # 对于 EditFile，后面本身还会再做存在性检查。
            return True, ""

        if current_mtime_ns != cached_mtime_ns:
            return (
                False,
                "Error: file has been modified since last read. Read it again before editing.",
            )

        return True, ""

    def update(self, path: str) -> None:
        """在成功写回文件后刷新缓存。

        输入:
            path: 已被成功写入的文件绝对路径。
        输出:
            无。若刷新失败，则删除旧缓存，避免保留过期状态。
        """
        try:
            file_path = Path(path)
            content = file_path.read_text(encoding="utf-8")
            mtime_ns = file_path.stat().st_mtime_ns
            self._cache[path] = (content, mtime_ns)
        except OSError:
            # 如果写完后无法再读取，就清掉旧状态，避免缓存与真实文件脱节。
            self._cache.pop(path, None)
