"""团队共享任务更新工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.teams.manager import TeamManager


class TaskUpdateParams(BaseModel):
    """TaskUpdate 的输入参数。"""

    task_id: str
    status: str | None = None
    assignee: str | None = None
    description: str | None = None
    add_blocks: list[str] | None = None
    add_blocked_by: list[str] | None = None


VALID_STATUSES = {"pending", "in_progress", "completed", "blocked"}


class TaskUpdateTool(Tool):
    """更新共享任务的状态、负责人、描述或依赖关系。"""

    name = "TaskUpdate"
    description = (
        "Update a shared task's status, assignee, description, or dependencies. "
        "Use add_blocks/add_blocked_by to add dependency relations."
    )
    params_model = TaskUpdateParams
    category = "command"
    is_concurrency_safe = True

    def __init__(self, team_manager: TeamManager, team_name: str) -> None:
        """保存团队上下文。"""
        self._team_manager = team_manager
        self._team_name = team_name

    async def execute(self, params: BaseModel) -> ToolResult:
        """对指定任务应用一组局部更新。"""
        task_params: TaskUpdateParams = params  # type: ignore[assignment]

        if task_params.status and task_params.status not in VALID_STATUSES:
            return ToolResult(
                output=f"Invalid status '{task_params.status}'. Must be one of: {', '.join(sorted(VALID_STATUSES))}",
                is_error=True,
            )

        store = self._team_manager.get_task_store(self._team_name)
        if store is None:
            return ToolResult(output=f"Task store not found for team '{self._team_name}'", is_error=True)

        task = store.update(
            task_id=task_params.task_id,
            status=task_params.status,
            assignee=task_params.assignee,
            description=task_params.description,
            add_blocks=task_params.add_blocks,
            add_blocked_by=task_params.add_blocked_by,
        )

        if task is None:
            return ToolResult(output=f"Task '{task_params.task_id}' not found", is_error=True)

        changes: list[str] = []
        if task_params.status:
            changes.append(f"status -> {task_params.status}")
        if task_params.assignee is not None:
            changes.append(f"assignee -> {task_params.assignee or '(unassigned)'}")
        if task_params.description is not None:
            changes.append("description updated")
        if task_params.add_blocks:
            changes.append(f"blocks += {', '.join(task_params.add_blocks)}")
        if task_params.add_blocked_by:
            changes.append(f"blocked_by += {', '.join(task_params.add_blocked_by)}")

        return ToolResult(
            output=f"Task {task.id} updated: {'; '.join(changes) if changes else 'no changes'}"
        )
