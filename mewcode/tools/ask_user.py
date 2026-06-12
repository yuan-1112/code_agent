"""向真实用户发起提问的工具。"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult


class QuestionItem(BaseModel):
    """描述一个待展示给用户的问题。"""

    type: str = Field(description="Question type: text, radio, select, checkbox")
    name: str = Field(description="Question identifier")
    message: str = Field(description="Question text to display")
    options: list[str] = Field(
        default_factory=list,
        description="Options for radio/select/checkbox types",
    )


class AskUserParams(BaseModel):
    """AskUser 的输入参数。"""

    questions: list[QuestionItem] = Field(
        description="List of questions to ask the user"
    )


class AskUserEvent:
    """在工具层与外部 UI 层之间传递提问请求的事件对象。"""

    def __init__(
        self,
        questions: list[dict[str, Any]],
        future: asyncio.Future[dict[str, str]],
    ) -> None:
        self.questions = questions
        self.future = future


class AskUserTool(Tool):
    """把问题抛给外部界面，再异步等待用户回答。"""

    name = "AskUserQuestion"
    description = (
        "Ask the user one or more questions when you need information "
        "that cannot be determined from code or context alone. Supports "
        "text input, radio (single select), select, and checkbox (multi select) "
        "question types."
    )
    params_model = AskUserParams
    category: str = "read"
    is_system_tool = True
    should_defer = True

    def __init__(self) -> None:
        """初始化当前待处理的提问事件。"""
        self._pending_event: AskUserEvent | None = None

    async def execute(self, params: AskUserParams) -> ToolResult:
        """发起提问并等待用户回答。

        输入:
            params.questions: 问题列表。
        输出:
            ToolResult，内容为“问题名: 回答”的多行文本。
        """
        questions_data = [question.model_dump() for question in params.questions]

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, str]] = loop.create_future()

        # 把提问事件暴露给外部界面层，由界面负责展示并在 future 中回填答案。
        self._pending_event = AskUserEvent(questions=questions_data, future=future)

        try:
            answers = await asyncio.wait_for(future, timeout=300)
        except asyncio.TimeoutError:
            return ToolResult(
                output="User did not respond within 5 minutes", is_error=True
            )
        finally:
            self._pending_event = None

        lines = []
        for question in params.questions:
            answer = answers.get(question.name, "(no answer)")
            lines.append(f"{question.name}: {answer}")

        return ToolResult(output="\n".join(lines))
