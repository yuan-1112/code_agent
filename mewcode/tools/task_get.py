"""团队共享任务详情查询工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.teams.manager import TeamManager


class TaskGetParams(BaseModel):
    """TaskGet 的输入参数。"""

    task_id: str


class TaskGetTool(Tool):
    """按任务 ID 查询共享任务详情。"""

    name = "TaskGet"
    description = "Get details of a shared task by ID, including dependency information."
    params_model = TaskGetParams
    category = "read"
    is_concurrency_safe = True

    def __init__(self, team_manager: TeamManager, team_name: str) -> None:
        """保存团队上下文。"""
        self._team_manager = team_manager
        self._team_name = team_name

    async def execute(self, params: BaseModel) -> ToolResult:
        """读取单个任务详情，并整理成可读文本。"""
        task_params: TaskGetParams = params  # type: ignore[assignment]

        store = self._team_manager.get_task_store(self._team_name)
        if store is None:
            return ToolResult(output=f"Task store not found for team '{self._team_name}'", is_error=True)

        task = store.get(task_params.task_id)
        if task is None:
            return ToolResult(output=f"Task '{task_params.task_id}' not found", is_error=True)

        lines = [
            f"Task {task.id}:",
            f"  Title:      {task.title}",
            f"  Status:     {task.status}",
            f"  Assignee:   {task.assignee or '(unassigned)'}",
            f"  Created by: {task.created_by or '(unknown)'}",
        ]
        if task.description:
            lines.append(f"  Description: {task.description}")
        if task.blocks:
            lines.append(f"  Blocks:     {', '.join(task.blocks)}")
        if task.blocked_by:
            lines.append(f"  Blocked by: {', '.join(task.blocked_by)}")

        return ToolResult(output="\n".join(lines))
