"""结构化输出工具。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult


class SyntheticOutputParams(BaseModel):
    """SyntheticOutput 的输入参数。"""

    output: dict[str, Any] | list[Any] | str


class SyntheticOutputTool(Tool):
    """把最终结果强制整理成 JSON 或字符串输出。"""

    name = "SyntheticOutput"
    description = (
        "Return structured output in JSON format. "
        "Use this tool to return your final response as structured data "
        "in non-interactive or coordinator mode sessions."
    )
    params_model = SyntheticOutputParams
    category = "read"
    is_concurrency_safe = True
    is_system_tool = True

    def __init__(self, json_schema: dict[str, Any] | None = None) -> None:
        """可选注入输出 schema，用于在返回前做基础校验。"""
        self._json_schema = json_schema

    async def execute(self, params: BaseModel) -> ToolResult:
        """校验并返回结构化输出。"""
        output_params: SyntheticOutputParams = params  # type: ignore[assignment]

        if self._json_schema is not None:
            error = self._validate_schema(output_params.output)
            if error:
                return ToolResult(output=f"Output does not match required schema: {error}", is_error=True)

        if isinstance(output_params.output, str):
            return ToolResult(output=output_params.output)

        return ToolResult(output=json.dumps(output_params.output, ensure_ascii=False, indent=2))

    def _validate_schema(self, data: Any) -> str | None:
        """对输出做一层轻量 schema 校验。

        这里不是完整 JSON Schema 校验器，只处理最常见的类型与 required 字段检查。
        """
        schema = self._json_schema
        if schema is None:
            return None

        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                return f"Expected object, got {type(data).__name__}"
            if expected_type == "array" and not isinstance(data, list):
                return f"Expected array, got {type(data).__name__}"
            if expected_type == "string" and not isinstance(data, str):
                return f"Expected string, got {type(data).__name__}"

        if "required" in schema and isinstance(data, dict):
            missing = [key for key in schema["required"] if key not in data]
            if missing:
                return f"Missing required fields: {', '.join(missing)}"

        return None
