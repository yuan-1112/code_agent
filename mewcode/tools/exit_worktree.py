"""退出隔离 worktree 的工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult
from mewcode.worktree.changes import count_worktree_changes

if TYPE_CHECKING:
    from mewcode.worktree.manager import WorktreeManager


class ExitWorktreeParams(BaseModel):
    """ExitWorktree 的输入参数。"""

    action: str = Field(
        description='"keep" leaves the worktree and branch on disk; "remove" deletes both.',
    )
    discard_changes: Optional[bool] = Field(
        default=None,
        description=(
            'Required true when action is "remove" and the worktree has '
            "uncommitted files or unmerged commits. "
            "The tool will refuse and list them otherwise."
        ),
    )


class ExitWorktreeTool(Tool):
    """从当前 worktree 会话退出，并可选择保留或删除该 worktree。"""

    name = "ExitWorktree"
    description = (
        "Exits a worktree session created by EnterWorktree and restores "
        "the original working directory"
    )
    params_model = ExitWorktreeParams
    category = "command"
    should_defer = True

    def __init__(self, worktree_manager: WorktreeManager) -> None:
        """保存 worktree 管理器引用。"""
        self._manager = worktree_manager

    async def execute(self, params: ExitWorktreeParams) -> ToolResult:
        """退出当前 worktree，会在删除前做丢失风险检查。"""
        session = self._manager.get_current_session()
        if session is None:
            return ToolResult(
                output=(
                    "No-op: there is no active EnterWorktree session to exit. "
                    "This tool only operates on worktrees created by EnterWorktree "
                    "in the current session - it will not touch worktrees created "
                    "manually or in a previous session. No filesystem changes were made."
                ),
                is_error=True,
            )

        action = params.action
        if action not in ("keep", "remove"):
            return ToolResult(
                output=f'Invalid action "{action}". Must be "keep" or "remove".',
                is_error=True,
            )

        discard = params.discard_changes or False

        if action == "remove" and not discard:
            # 删除 worktree 前先确认其中是否还有未保存的工作，避免误删。
            changes = count_worktree_changes(
                session.worktree_path, session.original_head_commit
            )
            if changes.uncommitted > 0 or changes.new_commits > 0:
                parts = []
                if changes.uncommitted > 0:
                    word = "file" if changes.uncommitted == 1 else "files"
                    parts.append(f"{changes.uncommitted} uncommitted {word}")
                if changes.new_commits > 0:
                    word = "commit" if changes.new_commits == 1 else "commits"
                    parts.append(f"{changes.new_commits} {word}")
                return ToolResult(
                    output=(
                        f"Worktree has {' and '.join(parts)}. "
                        "Removing will discard this work permanently. "
                        "Confirm with the user, then re-invoke with "
                        'discard_changes: true - or use action: "keep" '
                        "to preserve the worktree."
                    ),
                    is_error=True,
                )

        worktree_path = session.worktree_path
        original_cwd = session.original_cwd
        wt_name = session.worktree_name

        try:
            await self._manager.exit(wt_name, action=action, discard_changes=discard)
        except Exception as exc:
            return ToolResult(
                output=f"Error exiting worktree: {exc}", is_error=True
            )

        if action == "keep":
            return ToolResult(
                output=(
                    f"Exited worktree. Your work is preserved at {worktree_path}. "
                    f"Session is now back in {original_cwd}."
                )
            )

        return ToolResult(
            output=(
                f"Exited and removed worktree at {worktree_path}. "
                f"Session is now back in {original_cwd}."
            )
        )
