"""子 agent 启动工具。

这个文件是 tools 目录里最偏“编排层”的一个工具实现。
它负责把一次 Agent 工具调用分发到几种不同路径：
1. 普通子 agent，同步或后台运行。
2. worktree 隔离子 agent。
3. 团队队友型 agent。

因此阅读这份文件时，建议始终带着一个问题：
“当前这段代码是在决定走哪条分支，还是在真正启动某种子 agent？”
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.agent import Agent
    from mewcode.agents.loader import AgentLoader
    from mewcode.agents.parser import AgentDef
    from mewcode.agents.task_manager import TaskManager
    from mewcode.agents.trace import TraceManager
    from mewcode.client import LLMClient

log = logging.getLogger(__name__)


class AgentToolParams(BaseModel):
    """Agent 工具的输入参数。"""

    prompt: str
    description: str
    subagent_type: str | None = None
    model: str | None = None
    run_in_background: bool = False
    name: str | None = None
    isolation: str | None = None
    team_name: str | None = Field(
        default=None,
        description=(
            "REQUIRED when creating team members. Spawns the agent as a long-running "
            "teammate under this team (created via TeamCreate). Unlike regular sub-agents, "
            "team members run in their own terminal, persist after the lead returns, and "
            "communicate with each other via SendMessage. Without team_name the agent "
            "runs as a one-shot sub-agent that blocks and returns inline."
        ),
    )


# 配置文件里的 permission_mode 是字符串，这里把它映射到内部枚举名。
PERMISSION_MODE_MAP = {
    "default": "DEFAULT",
    "acceptEdits": "ACCEPT_EDITS",
    "dontAsk": "DONT_ASK",
}


# 队友 agent 与普通子 agent 最大区别在于：它不会把结果直接返回给主用户，
# 而是要通过 SendMessage 与团队 lead 或其他队友协作。
TEAMMATE_ADDENDUM = (
    "\n\nIMPORTANT: You are running as an agent in a team.\n"
    "Just writing a response in text is not visible to others\n"
    "on your team - you MUST use the SendMessage tool.\n"
    "The user interacts primarily with the team lead.\n"
    "Your work is coordinated through the task system\n"
    "and teammate messaging.\n\n"
    "You are working in an isolated Git worktree. "
    "All file paths you use MUST be relative to your current working directory. "
    "Do NOT use absolute paths from the original project - they are outside your sandbox and will be rejected."
)


class AgentTool(Tool):
    """根据参数启动子 agent、队友 agent 或 worktree agent。"""

    name = "Agent"
    description = (
        "Launch a sub-agent to handle a task in an isolated context. "
        "Use subagent_type to select a predefined agent type (e.g. Explore, Plan, general-purpose), "
        "or leave it empty to fork the current conversation. "
        "Use team_name to spawn a teammate in an existing team."
    )
    params_model = AgentToolParams
    category = "command"
    is_concurrency_safe = False

    def __init__(
        self,
        agent_loader: AgentLoader,
        task_manager: TaskManager,
        trace_manager: TraceManager,
        parent_agent: Agent,
        enable_fork: bool = False,
        provider_config: Any = None,
        worktree_manager: Any = None,
        team_manager: Any = None,
    ) -> None:
        """保存启动子 agent 所需的所有外部依赖。"""
        self._agent_loader = agent_loader
        self._task_manager = task_manager
        self._trace_manager = trace_manager
        self._parent_agent = parent_agent
        self._enable_fork = enable_fork
        self._provider_config = provider_config
        self._worktree_manager = worktree_manager
        self._team_manager = team_manager

    async def execute(self, params: BaseModel) -> ToolResult:
        """根据参数路由到不同的子 agent 启动路径。

        主要分支：
        - team_name 有值：走队友路径。
        - isolation=worktree：走 worktree 隔离路径。
        - 其余情况：走普通子 agent 路径。
        """
        p: AgentToolParams = params  # type: ignore[assignment]

        if p.team_name:
            return await self._execute_as_teammate(p)

        isolation = ""
        if p.subagent_type:
            defn = self._agent_loader.get(p.subagent_type)
            if defn and defn.isolation:
                isolation = defn.isolation

        if isolation == "worktree":
            return await self._execute_with_worktree(p)

        from mewcode.agent import Agent as AgentClass
        from mewcode.agents.fork import ForkError, build_forked_messages
        from mewcode.agents.parser import AgentDef
        from mewcode.agents.tool_filter import resolve_agent_tools
        from mewcode.conversation import ConversationManager
        from mewcode.permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionChecker,
            PermissionMode,
            RuleEngine,
        )

        definition: AgentDef | None = None
        conversation: ConversationManager

        if p.subagent_type:
            # 指定了 subagent_type 时，从 agent 模板定义中读取系统提示词、
            # 最大轮数、权限模式、工具白名单等信息。
            definition = self._agent_loader.get(p.subagent_type)
            if definition is None:
                return ToolResult(
                    output=f"Unknown agent type: '{p.subagent_type}'. "
                    f"Available types: {', '.join(t for t, _ in self._agent_loader.list_agents())}",
                    is_error=True,
                )
            conversation = ConversationManager()
        else:
            # 未指定类型时，只有开启 fork 模式才允许基于父对话直接派生子 agent。
            if not self._enable_fork:
                return ToolResult(
                    output="Fork mode is not enabled. "
                    "Set 'enable_fork: true' in config.yaml to use fork, "
                    "or specify a subagent_type parameter.",
                    is_error=True,
                )
            try:
                parent_conv = getattr(self._parent_agent, "_current_conversation", None)
                if parent_conv is None:
                    return ToolResult(
                        output="Cannot fork: no active conversation in parent agent.",
                        is_error=True,
                    )
                conversation = build_forked_messages(parent_conv, p.prompt)
            except ForkError as exc:
                return ToolResult(output=str(exc), is_error=True)

            definition = AgentDef(
                agent_type="fork",
                when_to_use="Forked from parent agent",
                system_prompt="",
                disallowed_tools=[],
                model="inherit",
                max_turns=self._parent_agent.max_iterations,
                permission_mode="dontAsk",
                source="builtin",
            )

        # 选择子 agent 使用哪个模型客户端：
        # 可能继承父 agent，也可能使用请求里显式指定的模型。
        client = self._select_llm(p, definition)

        # background 与 fork 的组合关系比较特殊：
        # 只要启用了 fork 模式，这里就强制后台运行，避免父子对话互相阻塞。
        is_background = p.run_in_background or definition.background
        if self._enable_fork:
            is_background = True

        # coordinator 模式下，父 agent 可能已把 registry 缩成调度视角版本；
        # 这里优先拿完整 registry，再按子 agent 定义做一次细粒度过滤。
        base_registry = getattr(self._parent_agent, "_full_registry", None) or self._parent_agent.registry
        filtered_registry = resolve_agent_tools(
            base_registry, definition, is_background
        )

        # 每个子 agent 都有自己独立的权限检查器，保证不同隔离策略互不影响。
        pm_str = definition.permission_mode
        pm_enum = getattr(
            PermissionMode,
            PERMISSION_MODE_MAP.get(pm_str, "DEFAULT"),
            PermissionMode.DEFAULT,
        )
        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(self._parent_agent.work_dir),
            rule_engine=RuleEngine(),
            mode=pm_enum,
        )

        # 真正实例化子 agent。到这一步为止，前面所有逻辑都只是“准备启动条件”。
        sub_agent = AgentClass(
            client=client,
            registry=filtered_registry,
            protocol=self._parent_agent.protocol,
            work_dir=self._parent_agent.work_dir,
            max_iterations=definition.max_turns,
            permission_checker=checker,
            context_window=self._parent_agent.context_window,
            instructions_content=definition.system_prompt,
            hook_engine=self._parent_agent.hook_engine,
        )
        sub_agent.parent_id = self._parent_agent.agent_id
        sub_agent.trace_id = self._parent_agent.trace_id or self._parent_agent.agent_id

        if p.subagent_type is None:
            # fork 模式会复制父 agent 的 replacement_state，
            # 以便父子共用 prompt cache 时仍能保持 tool_result 替换逻辑一致。
            from mewcode.context import clone_replacement_state

            sub_agent.replacement_state = clone_replacement_state(
                self._parent_agent.replacement_state
            )

        # trace 节点是子 agent 生命周期记录的入口，后续完成/失败都会回写到这里。
        trace_node = self._trace_manager.create(
            agent_type=definition.agent_type,
            parent_id=self._parent_agent.agent_id,
            trace_id=sub_agent.trace_id,
        )
        sub_agent.agent_id = trace_node.agent_id

        agent_name = p.name or p.subagent_type or f"agent-{trace_node.agent_id}"
        is_fork = p.subagent_type is None

        if is_background:
            # 后台模式下，不等待子 agent 结果，而是交给 TaskManager 持续运行。
            if is_fork:
                sub_agent._fork_conversation = conversation
            task_id = self._task_manager.launch(
                agent=sub_agent,
                task="" if is_fork else p.prompt,
                name=agent_name,
                fork_conversation=conversation if is_fork else None,
            )
            return ToolResult(
                output=f"Sub-agent launched in background.\n"
                f"Task ID: {task_id}\n"
                f"Agent: {agent_name}\n"
                f"Type: {definition.agent_type}\n"
                f"The system will notify automatically when it completes.\n"
                f"Do NOT wait, sleep, or poll. Report the task ID to the user and move on.",
            )

        try:
            # 前台模式下，同步等待子 agent 跑到结束。
            if is_fork:
                result_text = await sub_agent.run_to_completion("", conversation)
            else:
                result_text = await sub_agent.run_to_completion(p.prompt)
        except Exception as exc:
            self._trace_manager.complete(trace_node.agent_id, "failed")
            return ToolResult(
                output=f"Sub-agent failed: {exc}", is_error=True
            )

        self._trace_manager.update(
            trace_node.agent_id,
            input_tokens=sub_agent.total_input_tokens,
            output_tokens=sub_agent.total_output_tokens,
        )
        self._trace_manager.complete(trace_node.agent_id, "completed")

        return ToolResult(output=result_text or "(sub-agent returned no output)")

    async def _execute_as_teammate(self, p: AgentToolParams) -> ToolResult:
        """以“团队队友”的方式启动子 agent。

        这条链路和普通子 agent 的不同点在于：
        - 会创建独立 worktree。
        - 会构建专门的 teammate 工具集。
        - 结果不会直接返回给主用户，而是通过团队机制协作。
        """
        if self._team_manager is None:
            return ToolResult(output="TeamManager not configured.", is_error=True)
        if self._worktree_manager is None:
            return ToolResult(output="WorktreeManager not configured for team spawn.", is_error=True)

        from mewcode.agent import Agent as AgentClass
        from mewcode.agents.fork import ForkError, build_forked_messages
        from mewcode.agents.parser import AgentDef
        from mewcode.agents.tool_filter import build_teammate_tools
        from mewcode.conversation import ConversationManager
        from mewcode.permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionChecker,
            PermissionMode,
            RuleEngine,
        )
        from mewcode.teams.models import BackendType, TeammateInfo
        from mewcode.teams.registry import AgentNameRegistry

        team = self._team_manager.get_team(p.team_name)
        if team is None:
            return ToolResult(output=f"Team '{p.team_name}' not found. Create it first with TeamCreate.", is_error=True)

        # 队友名需要在团队内唯一；若重名则自动追加计数后缀。
        base_name = p.name or p.subagent_type or "worker"
        existing_names = {member.name for member in team.members}
        teammate_name = base_name
        if teammate_name in existing_names:
            counter = 2
            while f"{base_name}-{counter}" in existing_names:
                counter += 1
            teammate_name = f"{base_name}-{counter}"

        definition: AgentDef
        conversation: ConversationManager | None = None
        is_fork = False

        if p.subagent_type:
            defn = self._agent_loader.get(p.subagent_type)
            if defn is None:
                return ToolResult(
                    output=f"Unknown agent type: '{p.subagent_type}'. "
                    f"Available: {', '.join(t for t, _ in self._agent_loader.list_agents())}",
                    is_error=True,
                )
            definition = defn
        else:
            # 队友也允许 fork 父对话，但只有 enable_fork 打开时才生效。
            if self._enable_fork:
                try:
                    parent_conv = getattr(self._parent_agent, "_current_conversation", None)
                    if parent_conv is None:
                        return ToolResult(output="Cannot fork: no active conversation.", is_error=True)
                    conversation = build_forked_messages(parent_conv, p.prompt)
                    is_fork = True
                except ForkError as exc:
                    return ToolResult(output=str(exc), is_error=True)

            definition = AgentDef(
                agent_type="teammate",
                when_to_use="Team member",
                system_prompt="",
                disallowed_tools=[],
                model="inherit",
                max_turns=self._parent_agent.max_iterations,
                permission_mode="dontAsk",
                source="builtin",
            )

        # 每个队友都分配独立 worktree，避免相互写同一工作目录。
        wt_name = f"team-{p.team_name}/{teammate_name}"
        try:
            wt = await self._worktree_manager.create(wt_name, "HEAD")
        except Exception as exc:
            return ToolResult(output=f"Failed to create worktree for teammate: {exc}", is_error=True)

        client = self._select_llm(p, definition)

        # 团队后端可能是 tmux、iTerm2 或进程内执行，不同后端会决定后续启动方式。
        backend = self._team_manager.detect_backend()

        trace_node = self._trace_manager.create(
            agent_type=definition.agent_type,
            parent_id=self._parent_agent.agent_id,
            trace_id=self._parent_agent.trace_id or self._parent_agent.agent_id,
        )
        agent_id = trace_node.agent_id

        full_registry = getattr(self._parent_agent, "_full_registry", None) or self._parent_agent.registry
        full_tools = [tool.name for tool in full_registry.list_tools()]
        log.info(
            "[teammate] full_tools=%d names=%s backend=%s def_tools=%s def_disallowed=%s",
            len(full_tools),
            full_tools,
            backend.value,
            getattr(definition, "tools", []),
            getattr(definition, "disallowed_tools", []),
        )
        teammate_registry = build_teammate_tools(
            parent_registry=full_registry,
            team_manager=self._team_manager,
            team_name=p.team_name,
            agent_id=agent_id,
            agent_name=teammate_name,
            backend_type=backend.value,
            definition=definition,
        )
        teammate_tools = [tool.name for tool in teammate_registry.list_tools()]
        log.info("[teammate] result_tools=%d names=%s", len(teammate_tools), teammate_tools)

        # 队友说明会在原始 system_prompt 后追加，强制提醒其通过消息系统协作。
        instructions = (definition.system_prompt or "") + TEAMMATE_ADDENDUM

        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(wt.path),
            rule_engine=RuleEngine(),
            mode=PermissionMode.DONT_ASK,
        )

        sub_agent = AgentClass(
            client=client,
            registry=teammate_registry,
            protocol=self._parent_agent.protocol,
            work_dir=wt.path,
            max_iterations=definition.max_turns,
            permission_checker=checker,
            context_window=self._parent_agent.context_window,
            instructions_content=instructions,
            hook_engine=self._parent_agent.hook_engine,
        )
        sub_agent.parent_id = self._parent_agent.agent_id
        sub_agent.trace_id = self._parent_agent.trace_id or self._parent_agent.agent_id
        sub_agent.agent_id = agent_id
        sub_agent.team_name = p.team_name
        sub_agent._team_manager = self._team_manager

        # 注册名称和成员信息后，团队其他组件才能通过名字解析到该队友。
        AgentNameRegistry.instance().register(teammate_name, agent_id)

        member = TeammateInfo(
            name=teammate_name,
            agent_id=agent_id,
            agent_type=definition.agent_type,
            model=p.model or definition.model,
            worktree_path=wt.path,
            backend_type=backend.value,
            is_active=True,
        )
        self._team_manager.register_member(p.team_name, member)

        if backend in (BackendType.TMUX, BackendType.ITERM2):
            return self._spawn_pane_teammate(
                p, member, backend, wt, agent_id, teammate_name
            )

        # 进程内后端直接交给 TaskManager 运行，由系统后续通知完成结果。
        task_id = self._task_manager.launch(
            agent=sub_agent,
            task="" if is_fork else p.prompt,
            name=teammate_name,
            fork_conversation=conversation if is_fork else None,
        )

        return ToolResult(
            output=(
                f"Teammate '{teammate_name}' spawned in team '{p.team_name}'.\n"
                f"Agent ID: {agent_id}\n"
                f"Backend: {backend.value}\n"
                f"Worktree: {wt.path}\n"
                f"Task ID: {task_id}\n"
                f"The system will notify when it completes."
            )
        )

    def _spawn_pane_teammate(
        self,
        p: Any,
        member: Any,
        backend: Any,
        wt: Any,
        agent_id: str,
        teammate_name: str,
    ) -> ToolResult:
        """使用 tmux 或 iTerm2 面板启动独立队友进程。"""
        from mewcode.teams.models import BackendType

        mailbox = self._team_manager.get_mailbox(p.team_name)
        mailbox_dir = str(mailbox._base_dir) if mailbox else ""

        try:
            if backend == BackendType.TMUX:
                from mewcode.teams.spawn_tmux import spawn_tmux_teammate

                pane_info = spawn_tmux_teammate(
                    team_name=p.team_name,
                    teammate_name=teammate_name,
                    worktree_path=wt.path,
                    prompt=p.prompt,
                    agent_type=p.subagent_type or "",
                    model=p.model or "",
                    mailbox_dir=mailbox_dir,
                )
                self._team_manager.register_pane_id(agent_id, pane_info.pane_id)
            elif backend == BackendType.ITERM2:
                from mewcode.teams.spawn_iterm2 import spawn_iterm2_teammate

                pane_info = spawn_iterm2_teammate(
                    team_name=p.team_name,
                    teammate_name=teammate_name,
                    worktree_path=wt.path,
                    prompt=p.prompt,
                    agent_type=p.subagent_type or "",
                    model=p.model or "",
                    mailbox_dir=mailbox_dir,
                )
        except Exception as exc:
            log.warning("Pane spawn failed, falling back to in-process: %s", exc)
            return ToolResult(
                output=f"Pane spawn failed ({exc}), teammate not started. Retry or set teammate_mode to in-process.",
                is_error=True,
            )

        return ToolResult(
            output=(
                f"Teammate '{teammate_name}' spawned in team '{p.team_name}'.\n"
                f"Agent ID: {agent_id}\n"
                f"Backend: {backend.value} (pane)\n"
                f"Worktree: {wt.path}\n"
                f"The teammate is running in an independent process."
            )
        )

    def _select_llm(
        self,
        params: AgentToolParams,
        definition: AgentDef,
    ) -> LLMClient:
        """决定子 agent 使用哪个 LLMClient。

        优先级：
        1. 请求参数里显式指定 model。
        2. agent 定义里声明的 model（且不为 inherit）。
        3. 继承父 agent 当前的 client。
        """
        model_override = params.model or (
            definition.model if definition.model != "inherit" else None
        )

        if model_override and model_override != "inherit":
            client = self._create_client_for_model(model_override)
            if client is not None:
                return client

        return self._parent_agent.client

    async def _execute_with_worktree(self, p: AgentToolParams) -> ToolResult:
        """以独立 worktree 的方式运行子 agent。"""
        if self._worktree_manager is None:
            return ToolResult(
                output="Worktree isolation is not available: WorktreeManager not configured.",
                is_error=True,
            )

        from mewcode.agent import Agent as AgentClass
        from mewcode.agents.parser import AgentDef
        from mewcode.agents.tool_filter import resolve_agent_tools
        from mewcode.permissions import (
            DangerousCommandDetector,
            PathSandbox,
            PermissionChecker,
            PermissionMode,
            RuleEngine,
        )
        from mewcode.worktree.integration import (
            build_worktree_notice,
            generate_worktree_name,
        )

        definition: AgentDef | None = None
        if p.subagent_type:
            definition = self._agent_loader.get(p.subagent_type)
            if definition is None:
                return ToolResult(
                    output=f"Unknown agent type: '{p.subagent_type}'. "
                    f"Available types: {', '.join(t for t, _ in self._agent_loader.list_agents())}",
                    is_error=True,
                )
        else:
            definition = AgentDef(
                agent_type="worktree-agent",
                when_to_use="Isolated worktree agent",
                system_prompt="",
                disallowed_tools=[],
                model="inherit",
                max_turns=self._parent_agent.max_iterations,
                permission_mode="dontAsk",
                source="builtin",
            )

        wt_name = generate_worktree_name()
        try:
            wt = await self._worktree_manager.create(wt_name, "HEAD")
        except Exception as exc:
            return ToolResult(
                output=f"Failed to create worktree: {exc}",
                is_error=True,
            )

        # 在真正用户任务前拼接一段说明，告诉子 agent 原目录与 worktree 的关系。
        notice = build_worktree_notice(self._parent_agent.work_dir, wt.path)
        task = notice + "\n\n" + p.prompt

        client = self._select_llm(p, definition)

        base_registry = getattr(self._parent_agent, "_full_registry", None) or self._parent_agent.registry
        filtered_registry = resolve_agent_tools(
            base_registry, definition, False
        )

        pm_str = definition.permission_mode
        pm_enum = getattr(
            PermissionMode,
            PERMISSION_MODE_MAP.get(pm_str, "DEFAULT"),
            PermissionMode.DEFAULT,
        )
        checker = PermissionChecker(
            detector=DangerousCommandDetector(),
            sandbox=PathSandbox(wt.path),
            rule_engine=RuleEngine(),
            mode=pm_enum,
        )

        sub_agent = AgentClass(
            client=client,
            registry=filtered_registry,
            protocol=self._parent_agent.protocol,
            work_dir=wt.path,
            max_iterations=definition.max_turns,
            permission_checker=checker,
            context_window=self._parent_agent.context_window,
            instructions_content=definition.system_prompt,
            hook_engine=self._parent_agent.hook_engine,
        )
        sub_agent.parent_id = self._parent_agent.agent_id
        sub_agent.trace_id = self._parent_agent.trace_id or self._parent_agent.agent_id

        trace_node = self._trace_manager.create(
            agent_type=definition.agent_type,
            parent_id=self._parent_agent.agent_id,
            trace_id=sub_agent.trace_id,
        )
        sub_agent.agent_id = trace_node.agent_id

        try:
            result_text = await sub_agent.run_to_completion(task)
        except Exception as exc:
            self._trace_manager.complete(trace_node.agent_id, "failed")
            return ToolResult(
                output=f"Sub-agent in worktree failed: {exc}",
                is_error=True,
            )

        self._trace_manager.update(
            trace_node.agent_id,
            input_tokens=sub_agent.total_input_tokens,
            output_tokens=sub_agent.total_output_tokens,
        )
        self._trace_manager.complete(trace_node.agent_id, "completed")

        cleanup = await self._worktree_manager.auto_cleanup(wt_name, wt.head_commit)
        if cleanup.kept:
            result_text = (result_text or "") + (
                f"\n[Worktree preserved at {cleanup.path}, branch {cleanup.branch}]"
            )

        return ToolResult(output=result_text or "(sub-agent returned no output)")

    def _create_client_for_model(self, model_alias: str) -> LLMClient | None:
        """按模型别名创建新的客户端实例。

        这里的目的不是修改父 agent 的 client，而是给子 agent 单独换一个模型。
        """
        if self._provider_config is None:
            return None

        from mewcode.client import create_client
        from mewcode.config import ProviderConfig

        model_map = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-6-20250514",
            "opus": "claude-opus-4-6-20250514",
        }
        model_id = model_map.get(model_alias, model_alias)

        config = ProviderConfig(
            name=f"sub-{model_alias}",
            protocol=self._provider_config.protocol,
            base_url=self._provider_config.base_url,
            model=model_id,
            api_key=self._provider_config.api_key,
            context_window=self._provider_config.context_window,
        )
        try:
            return create_client(config)
        except Exception:
            # 如果新模型客户端创建失败，就回退到父 agent 当前客户端。
            return None
