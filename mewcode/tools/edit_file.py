"""精确替换式编辑工具。

这个工具不是整文件覆盖，而是在文件中找到 old_string，并替换成 new_string。
为了降低误改风险，这里要求 old_string 必须唯一命中。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.cache import FileCache
    from mewcode.tools.file_state_cache import FileStateCache


class Params(BaseModel):
    """EditFile 的输入参数。"""

    file_path: str = Field(description="Path to the file to edit")
    old_string: str = Field(description="The exact string to find and replace (must be unique in file)")
    new_string: str = Field(description="The replacement string")


class EditFile(Tool):
    """在文件中做一次唯一匹配的精确替换。"""

    name = "EditFile"
    description = (
        "Replace an exact string in a file. The old_string must appear exactly once in the file.\n"
        "You MUST read the file with ReadFile before editing. This tool will fail otherwise."
    )
    params_model = Params
    category = "write"

    def __init__(
        self,
        file_cache: FileCache | None = None,
        file_history: Any = None,
        file_state_cache: FileStateCache | None = None,
    ) -> None:
        """保存编辑时依赖的缓存和历史组件。"""
        self._cache = file_cache
        self.file_history = file_history
        self._state_cache = file_state_cache

    async def execute(self, params: Params) -> ToolResult:
        """执行精确替换式编辑。

        输入:
            params.file_path: 要修改的文件。
            params.old_string: 旧文本，必须唯一命中。
            params.new_string: 新文本。
        输出:
            ToolResult，说明编辑是否成功。
        """
        if self.file_history is not None:
            self.file_history.track_edit(params.file_path)

        path = Path(params.file_path)
        if not path.exists():
            return ToolResult(output=f"Error: file not found: {params.file_path}", is_error=True)

        if self._state_cache:
            resolved = str(path.resolve())
            ok, err_msg = self._state_cache.check(resolved)
            if not ok:
                return ToolResult(output=err_msg, is_error=True)

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            return ToolResult(output=f"Error reading file: {exc}", is_error=True)

        count = content.count(params.old_string)
        if count == 0:
            return ToolResult(output="Error: old_string not found in file", is_error=True)
        if count > 1:
            # 要求唯一命中，是为了避免把同样的字符串误替换到多个位置。
            return ToolResult(
                output=f"Error: old_string found {count} times, must be unique",
                is_error=True,
            )

        new_content = content.replace(params.old_string, params.new_string, 1)
        try:
            path.write_text(new_content, encoding="utf-8")
            if self._cache:
                self._cache.invalidate(str(path.resolve()))
            if self._state_cache:
                self._state_cache.update(str(path.resolve()))
        except Exception as exc:
            return ToolResult(output=f"Error writing file: {exc}", is_error=True)

        return ToolResult(output=f"Successfully edited {params.file_path}")
