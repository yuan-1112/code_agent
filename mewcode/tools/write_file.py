"""整文件写入工具。

这个工具用于“用一份新内容覆盖整个文件”。
与普通写文件不同，这里会在写前检查文件是否已经被读取且未发生外部变更，
目的是降低模型误覆盖用户最新修改的风险。
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
    """WriteFile 的输入参数。"""

    file_path: str = Field(description="Path to the file to write")
    content: str = Field(description="Content to write to the file")


class WriteFile(Tool):
    """将整段内容写入目标文件，必要时自动创建父目录。"""

    name = "WriteFile"
    description = (
        "Write content to a file, creating parent directories if needed. Overwrites existing files.\n"
        "You MUST read existing files with ReadFile before overwriting them. This tool will fail otherwise."
    )
    params_model = Params
    category = "write"

    def __init__(
        self,
        file_cache: FileCache | None = None,
        file_history: Any = None,
        file_state_cache: FileStateCache | None = None,
    ) -> None:
        """保存写文件时需要协作的缓存与历史组件。"""
        self._cache = file_cache
        self.file_history = file_history
        self._state_cache = file_state_cache

    async def execute(self, params: Params) -> ToolResult:
        """执行整文件写入。

        输入:
            params.file_path: 目标文件路径。
            params.content: 要写入的完整内容。
        输出:
            ToolResult，说明写入是否成功。
        """
        if self.file_history is not None:
            # 先把本次编辑动作记到历史里，方便上层做追踪或撤销类功能。
            self.file_history.track_edit(params.file_path)

        path = Path(params.file_path)

        if self._state_cache and path.exists():
            resolved = str(path.resolve())
            ok, err_msg = self._state_cache.check(resolved)
            if not ok:
                return ToolResult(output=err_msg, is_error=True)

        try:
            # 对于不存在的父目录，自动补建，减少额外目录准备步骤。
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(params.content, encoding="utf-8")
            if self._cache:
                # 文件已被覆盖，旧缓存内容已经失效，需要立即清掉。
                self._cache.invalidate(str(path.resolve()))
            if self._state_cache:
                # 写成功后把缓存状态刷新成最新版本。
                self._state_cache.update(str(path.resolve()))
        except Exception as exc:
            return ToolResult(output=f"Error writing file: {exc}", is_error=True)
        return ToolResult(output=f"Successfully wrote to {params.file_path}")
