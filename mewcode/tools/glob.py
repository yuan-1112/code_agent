"""按文件名模式搜索文件的工具。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from mewcode.tools.base import SKIP_DIRS, Tool, ToolResult


class Params(BaseModel):
    """Glob 工具的输入参数。"""

    pattern: str = Field(description="Glob pattern to match (e.g. '**/*.py')")
    path: str = Field(default=".", description="Base directory to search from")


class Glob(Tool):
    """在指定目录下按 glob 模式查找文件。"""

    name = "Glob"
    description = "Find files matching a glob pattern, returning relative paths."
    params_model = Params
    category = "read"
    is_concurrency_safe = True

    async def execute(self, params: Params) -> ToolResult:
        """遍历目录并返回匹配到的文件相对路径列表。"""
        base = Path(params.path)
        if not base.exists():
            return ToolResult(output=f"Error: path not found: {params.path}", is_error=True)

        try:
            matches = sorted(
                str(path.relative_to(base))
                for path in base.glob(params.pattern)
                if path.is_file() and not any(part in SKIP_DIRS for part in path.parts)
            )
        except Exception as exc:
            return ToolResult(output=f"Error: {exc}", is_error=True)

        if not matches:
            return ToolResult(output="No files matched the pattern.")
        return ToolResult(output="\n".join(matches))
