"""团队共享任务创建工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.teams.manager import TeamManager


class TaskCreateParams(BaseModel):
    """TaskCreate 的输入参数。"""

    title: str
    description: str = ""
    assignee: str = ""
    blocks: list[str] | None = None
    blocked_by: list[str] | None = None


class TaskCreateTool(Tool):
    """向团队共享任务板中添加一个新任务。"""

    name = "TaskCreate"
    description = (
        "Create a shared task in the team's task board. "
        "Supports dependency tracking with blocks/blocked_by fields."
    )
    params_model = TaskCreateParams
    category = "command"
    is_concurrency_safe = True

    def __init__(
        self,
        team_manager: TeamManager,
        team_name: str,
        agent_name: str = "",
    ) -> None:
        """保存团队上下文和创建者名称。"""
        self._team_manager = team_manager
        self._team_name = team_name
        self._agent_name = agent_name

    async def execute(self, params: BaseModel) -> ToolResult:
        """创建任务，并返回新任务的核心信息。"""
        task_params: TaskCreateParams = params  # type: ignore[assignment]

        store = self._team_manager.get_task_store(self._team_name)
        if store is None:
            return ToolResult(output=f"Task store not found for team '{self._team_name}'", is_error=True)

        task = store.create(
            title=task_params.title,
            description=task_params.description,
            assignee=task_params.assignee,
            blocks=task_params.blocks,
            blocked_by=task_params.blocked_by,
            created_by=self._agent_name,
        )

        return ToolResult(
            output=(
                f"Task created:\n"
                f"  ID: {task.id}\n"
                f"  Title: {task.title}\n"
                f"  Status: {task.status}\n"
                f"  Assignee: {task.assignee or '(unassigned)'}"
            )
        )
