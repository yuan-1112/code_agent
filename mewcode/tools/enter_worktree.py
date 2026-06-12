"""进入隔离 worktree 的工具。"""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult
from mewcode.worktree.slug import validate_slug

if TYPE_CHECKING:
    from mewcode.worktree.manager import WorktreeManager


class EnterWorktreeParams(BaseModel):
    """EnterWorktree 的输入参数。"""

    name: Optional[str] = Field(
        default=None,
        description=(
            'Optional name for the worktree. Each "/"-separated segment may '
            "contain only letters, digits, dots, underscores, and dashes; "
            "max 64 chars total. A random name is generated if not provided."
        ),
    )


class EnterWorktreeTool(Tool):
    """创建新的 git worktree，并把当前会话切进去。"""

    name = "EnterWorktree"
    description = (
        "Creates an isolated worktree (via git) and switches the session into it"
    )
    params_model = EnterWorktreeParams
    category = "command"
    should_defer = True

    def __init__(self, worktree_manager: WorktreeManager) -> None:
        """保存 worktree 管理器引用。"""
        self._manager = worktree_manager

    async def execute(self, params: EnterWorktreeParams) -> ToolResult:
        """创建并进入一个新的隔离 worktree 会话。"""
        if self._manager.get_current_session() is not None:
            return ToolResult(
                output="Already in a worktree session", is_error=True
            )

        slug = params.name or f"wt-{secrets.token_hex(4)}"

        err = validate_slug(slug)
        if err:
            return ToolResult(output=f"Invalid worktree name: {err}", is_error=True)

        try:
            wt = await self._manager.create(slug)
            session = await self._manager.enter(slug)
        except Exception as exc:
            return ToolResult(
                output=f"Error creating worktree: {exc}", is_error=True
            )

        branch_info = f" on branch {wt.branch}" if wt.branch else ""
        return ToolResult(
            output=(
                f"Created worktree at {session.worktree_path}{branch_info}. "
                "The session is now working in the worktree. "
                "Use ExitWorktree to leave mid-session, or exit the session to be prompted."
            )
        )
