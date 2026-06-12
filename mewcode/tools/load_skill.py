"""动态加载 skill 的工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from mewcode.agent import Agent
    from mewcode.skills.loader import SkillLoader


class LoadSkillParams(BaseModel):
    """LoadSkill 的输入参数。"""

    name: str = Field(description="The name of the skill to load")


class LoadSkill(Tool):
    """激活一个 skill，并在必要时注册 skill 自带工具。"""

    name = "LoadSkill"
    description = (
        "Load and activate a skill by name. "
        "The skill's SOP will be pinned to the environment context "
        "and any specialized tools will be registered."
    )
    params_model = LoadSkillParams
    category = "read"
    is_concurrency_safe = False
    is_system_tool = True

    def __init__(self) -> None:
        """初始化 skill loader 和 agent 引用。"""
        self._loader: SkillLoader | None = None
        self._agent: Agent | None = None

    def set_loader(self, loader: SkillLoader) -> None:
        """注入 skill loader，供 execute() 查找技能定义。"""
        self._loader = loader

    def set_agent(self, agent: Agent) -> None:
        """注入当前 agent，供激活 skill 时更新上下文和注册工具。"""
        self._agent = agent

    async def execute(self, params: BaseModel) -> ToolResult:
        """加载并激活指定 skill。

        输入:
            params.name: 要加载的 skill 名称。
        输出:
            ToolResult，说明 skill 是否成功激活，以及是否注册了附加工具。
        """
        assert isinstance(params, LoadSkillParams)

        if self._loader is None or self._agent is None:
            return ToolResult(
                output="Error: LoadSkill not properly initialized",
                is_error=True,
            )

        skill = self._loader.get(params.name)
        if skill is None:
            available = ", ".join(name for name, _ in self._loader.get_catalog())
            return ToolResult(
                output=f"Error: unknown skill '{params.name}'. Available skills: {available}",
                is_error=True,
            )

        # 激活 skill 的核心步骤是把 SOP 固定进环境上下文，后续每轮推理都能看到。
        self._agent.activate_skill(skill.name, skill.prompt_body)

        tool_count = 0
        if skill.is_directory and skill.source_path is not None:
            from mewcode.skills.directory import register_skill_tools

            skill_dir = skill.source_path.parent
            tool_count = register_skill_tools(skill_dir, self._agent.registry)

        parts = [f"Skill '{skill.name}' activated. SOP pinned to environment context."]
        if tool_count > 0:
            parts.append(f"{tool_count} specialized tool(s) registered.")
        return ToolResult(output=" ".join(parts))
