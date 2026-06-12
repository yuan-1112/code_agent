"""队友消息发送工具。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.teams.manager import TeamManager

log = logging.getLogger(__name__)


class SendMessageParams(BaseModel):
    """SendMessage 的输入参数。"""

    to: str
    message: str
    summary: str = ""
    message_type: str = "text"
    metadata: dict[str, Any] | None = None


VALID_MESSAGE_TYPES = {"text", "shutdown_request", "shutdown_response"}


class SendMessageTool(Tool):
    """在团队协作模式下向其他 agent 发送消息。"""

    name = "SendMessage"
    description = (
        "Send a message to a teammate by name or agent ID. "
        "Use to='*' to broadcast to all teammates. "
        "For text messages, include a short summary (5-10 words). "
        "Supports structured types: shutdown_request, shutdown_response."
    )
    params_model = SendMessageParams
    category = "command"
    is_concurrency_safe = True

    def __init__(
        self,
        team_manager: TeamManager,
        team_name: str,
        from_agent_id: str,
        from_agent_name: str = "",
    ) -> None:
        """保存团队上下文和发送方信息。"""
        self._team_manager = team_manager
        self._team_name = team_name
        self._from_agent_id = from_agent_id
        self._from_agent_name = from_agent_name

    async def execute(self, params: BaseModel) -> ToolResult:
        """向指定队友或全体队友发送消息。"""
        message_params: SendMessageParams = params  # type: ignore[assignment]

        if message_params.message_type not in VALID_MESSAGE_TYPES:
            return ToolResult(
                output=f"Invalid message_type '{message_params.message_type}'. Must be one of: {', '.join(sorted(VALID_MESSAGE_TYPES))}",
                is_error=True,
            )

        if message_params.message_type == "text" and not message_params.summary:
            return ToolResult(
                output="Text messages require a 'summary' field (5-10 words).",
                is_error=True,
            )

        from mewcode.teams.mailbox import create_message
        from mewcode.teams.registry import AgentNameRegistry

        team = self._team_manager.get_team(self._team_name)
        if team is None:
            return ToolResult(output=f"Team '{self._team_name}' not found", is_error=True)

        mailbox = self._team_manager.get_mailbox(self._team_name)
        if mailbox is None:
            return ToolResult(output=f"Mailbox not found for team '{self._team_name}'", is_error=True)

        msg = create_message(
            from_agent=self._from_agent_name or self._from_agent_id,
            to_agent=message_params.to,
            content=message_params.message,
            summary=message_params.summary,
            message_type=message_params.message_type,
            metadata=message_params.metadata,
        )

        registry = AgentNameRegistry.instance()

        if message_params.to == "*":
            # 广播时跳过自己，并且把 team lead 也补进目标列表。
            member_ids = [
                member.agent_id for member in team.members
                if member.agent_id != self._from_agent_id
            ]
            if team.lead_agent_id != self._from_agent_id:
                member_ids.append(team.lead_agent_id)
            mailbox.broadcast(member_ids, msg, exclude=self._from_agent_id)
            self._wake_pane_members(member_ids)
            return ToolResult(output=f"Message broadcast to {len(member_ids)} teammates.")

        target_id = registry.resolve(message_params.to)
        if target_id is None:
            return ToolResult(
                output=f"Cannot resolve recipient '{message_params.to}'. Check the name or agent ID.",
                is_error=True,
            )

        mailbox.write(target_id, msg)
        self._wake_pane(target_id)

        return ToolResult(output=f"Message sent to '{message_params.to}'.")

    def _wake_pane(self, agent_id: str) -> None:
        """尝试唤醒目标 agent 所在终端面板，让其尽快处理新消息。"""
        pane_id = self._team_manager.get_pane_id(agent_id)
        if pane_id is None:
            return
        try:
            from mewcode.teams.spawn_tmux import send_keys_to_pane

            send_keys_to_pane(pane_id, "")
        except Exception:
            # 唤醒失败不影响消息本身已经写入邮箱。
            pass

    def _wake_pane_members(self, agent_ids: list[str]) -> None:
        """批量唤醒多个队友面板。"""
        for agent_id in agent_ids:
            self._wake_pane(agent_id)
