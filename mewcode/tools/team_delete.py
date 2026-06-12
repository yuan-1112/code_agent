"""团队删除工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.agent import Agent
    from mewcode.teams.manager import TeamManager


class TeamDeleteParams(BaseModel):
    """TeamDelete 的输入参数。"""

    team_name: str


class TeamDeleteTool(Tool):
    """删除团队，并在必要时恢复主 agent 的完整工具集。"""

    name = "TeamDelete"
    description = (
        "Delete an Agent Team. Terminates all pane processes, removes worktrees, "
        "cleans up mailbox and team directory. Requires all members to be idle."
    )
    params_model = TeamDeleteParams
    category = "command"
    is_concurrency_safe = False

    def __init__(self, team_manager: TeamManager, parent_agent: Agent | None = None) -> None:
        """保存团队管理器和可选的父 agent 引用。"""
        self._team_manager = team_manager
        self._parent_agent = parent_agent

    async def execute(self, params: BaseModel) -> ToolResult:
        """删除指定团队，并清理协调者模式状态。"""
        team_params: TeamDeleteParams = params  # type: ignore[assignment]

        from mewcode.teams.manager import TeamError

        try:
            self._team_manager.delete_team(team_params.team_name)
        except TeamError as exc:
            return ToolResult(output=str(exc), is_error=True)
        except Exception as exc:
            return ToolResult(output=f"Failed to delete team: {exc}", is_error=True)

        coordinator_note = ""
        if self._parent_agent and self._parent_agent.coordinator_mode:
            full_registry = getattr(self._parent_agent, "_full_registry", None)
            if full_registry is not None:
                self._parent_agent.registry = full_registry
                self._parent_agent._full_registry = None
            self._parent_agent.coordinator_mode = False
            coordinator_note = "\nCoordinator Mode deactivated: full tools restored."

        return ToolResult(output=f"Team '{team_params.team_name}' deleted successfully.{coordinator_note}")
