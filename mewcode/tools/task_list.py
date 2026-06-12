"""团队共享任务列表工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.teams.manager import TeamManager


class TaskListParams(BaseModel):
    """TaskList 的输入参数。"""

    status: str | None = None
    assignee: str | None = None


class TaskListTool(Tool):
    """列出团队当前任务，可按状态和负责人过滤。"""

    name = "TaskList"
    description = (
        "List all shared tasks in the team's task board. "
        "Optionally filter by status (pending/in_progress/completed/blocked) or assignee."
    )
    params_model = TaskListParams
    category = "read"
    is_concurrency_safe = True

    def __init__(self, team_manager: TeamManager, team_name: str) -> None:
        """保存团队上下文。"""
        self._team_manager = team_manager
        self._team_name = team_name

    async def execute(self, params: BaseModel) -> ToolResult:
        """查询任务列表，并拼装成适合模型阅读的文本。"""
        task_params: TaskListParams = params  # type: ignore[assignment]

        store = self._team_manager.get_task_store(self._team_name)
        if store is None:
            return ToolResult(output=f"Task store not found for team '{self._team_name}'", is_error=True)

        tasks = store.list_tasks(status=task_params.status, assignee=task_params.assignee)

        if not tasks:
            filters = []
            if task_params.status:
                filters.append(f"status={task_params.status}")
            if task_params.assignee:
                filters.append(f"assignee={task_params.assignee}")
            filter_str = f" (filters: {', '.join(filters)})" if filters else ""
            return ToolResult(output=f"No tasks found{filter_str}")

        status_icons = {
            "pending": "[PENDING]",
            "in_progress": "[IN_PROGRESS]",
            "completed": "[COMPLETED]",
            "blocked": "[BLOCKED]",
        }

        lines = [f"Tasks ({len(tasks)}):"]
        for task in tasks:
            icon = status_icons.get(task.status, "[UNKNOWN]")
            assignee = f" [{task.assignee}]" if task.assignee else ""
            deps = ""
            if task.blocked_by:
                deps = f" (blocked by: {', '.join(task.blocked_by)})"
            lines.append(f"  {icon} [{task.id}] {task.title}{assignee}{deps}")

        return ToolResult(output="\n".join(lines))
