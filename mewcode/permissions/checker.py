"""权限检查主编排器。

这个文件是整个权限系统的入口，负责把多个独立防线串成一条固定顺序的检查链：
1. Plan 模式特例放行。
2. 安全只读命令白名单。
3. 危险命令黑名单。
4. 路径沙箱。
5. 规则引擎。
6. 权限模式兜底。
7. 最终落到人工确认（HITL）。

真正的危险命令模式、路径检查、规则加载都不在这里实现；
这里的职责是“决定先检查谁、命中后何时提前返回”。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from mewcode.permissions.dangerous import DangerousCommandDetector, is_safe_command
from mewcode.permissions.modes import DecisionEffect, PermissionMode, mode_decide
from mewcode.permissions.rules import RuleEngine, extract_content
from mewcode.permissions.sandbox import PathSandbox
from mewcode.tools.base import Tool

# Plan 模式并不是绝对只读；为了让“先规划再确认”这条链路能工作，
# 这里显式放行少数编排型工具。
_PLAN_MODE_ALLOWED_TOOLS = frozenset(
    {"Agent", "ToolSearch", "AskUserQuestion", "ExitPlanMode"}
)


@dataclass
class Decision:
    """封装一次权限检查的最终结果。

    effect:
        三种可能之一：allow / deny / ask。
    reason:
        人类可读的说明文本，既可写日志，也可直接展示给用户。
    """

    effect: DecisionEffect
    reason: str


class PermissionChecker:
    """权限检查器。

    这个类本身不实现危险命令检测、路径解析或规则匹配细节，
    而是通过依赖注入接收这些组件，再在 check() 中按固定层次串起来。
    """

    def __init__(
        self,
        detector: DangerousCommandDetector,
        sandbox: PathSandbox,
        rule_engine: RuleEngine,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> None:
        """初始化权限检查器。

        输入:
            detector: 危险命令检测器。
            sandbox: 路径沙箱，用于限制文件读写范围。
            rule_engine: 规则引擎，用于匹配用户级/项目级/本地级规则。
            mode: 当前权限模式。
        """
        self.detector = detector
        self.sandbox = sandbox
        self.rule_engine = rule_engine
        self.mode = mode
        # 计划模式下允许写入的目标计划文件路径，由上层在进入 plan mode 时注入。
        self.plan_file_path: str = ""

    def check(self, tool: Tool, arguments: dict[str, Any]) -> Decision:
        """对一次工具调用做完整权限判定。

        输入:
            tool: 即将执行的工具对象。
            arguments: 模型给出的工具参数。
        输出:
            Decision，表示本次调用是允许、拒绝还是需要人工确认。
        """
        # 先抽取“真正用于权限判断的核心内容”。
        # 例如 Bash 看 command，ReadFile/WriteFile 看 file_path。
        content = extract_content(tool.name, arguments)

        # Layer 0: Plan 模式特例放行。
        # Plan 模式整体倾向于阻断写操作和命令，但有少量例外必须放行，
        # 否则计划链路本身无法工作。
        if self.mode == PermissionMode.PLAN:
            if tool.name in _PLAN_MODE_ALLOWED_TOOLS:
                return Decision(effect="allow", reason="Plan mode: allowed tool")
            if tool.name in ("WriteFile", "EditFile") and content:
                if self._is_plan_file(content):
                    return Decision(
                        effect="allow",
                        reason="Plan mode: plan file write",
                    )

        # Layer 1: 安全只读命令自动放行。
        # 这一步只对 command 工具生效，目的是减少用户被频繁读操作打断确认。
        if tool.category == "command" and is_safe_command(content or ""):
            return Decision(effect="allow", reason="Safe read-only command")

        # Layer 1b: 危险命令黑名单。
        # 只对 command 工具生效；如果命中高风险命令模式，直接拒绝。
        if tool.category == "command":
            hit, reason = self.detector.detect(content)
            if hit:
                return Decision(effect="deny", reason=f"危险命令拦截: {reason}")

        # Layer 2: 路径沙箱。
        # 只对 read / write 生效；如果路径逃逸到沙箱外，直接拒绝。
        if tool.category in ("read", "write") and content:
            ok, reason = self.sandbox.check(content)
            if not ok:
                return Decision(effect="deny", reason=f"路径沙箱拦截: {reason}")

        # Layer 3: 规则引擎。
        # 这里允许用户通过 permissions.yaml 主动声明 allow / deny。
        rule_result = self.rule_engine.evaluate(tool.name, content)
        if rule_result == "allow":
            return Decision(effect="allow", reason="权限规则放行")
        if rule_result == "deny":
            return Decision(effect="deny", reason="权限规则拒绝")

        # Layer 4: 权限模式兜底。
        # 如果前面都没有做出明确决策，就交给当前模式矩阵决定。
        effect = mode_decide(self.mode, tool.category)
        if effect == "allow":
            return Decision(
                effect="allow",
                reason=f"权限模式 {self.mode.value} 放行",
            )
        if effect == "deny":
            return Decision(
                effect="deny",
                reason=f"权限模式 {self.mode.value} 拒绝",
            )

        # Layer 5: 最终落到 HITL。
        # 也就是先停下来，交给 UI 或外层系统向用户发起确认。
        return Decision(effect="ask", reason="需要用户确认")

    def _is_plan_file(self, target_path: str) -> bool:
        """判断目标路径是否属于计划模式允许写入的计划文件。

        输入:
            target_path: 工具调用里要写入的目标路径。
        输出:
            bool，表示是否可视为计划文件。
        """
        if not self.plan_file_path or not target_path:
            return ".mewcode/plans/" in target_path
        try:
            abs_target = os.path.abspath(target_path)
            abs_plan = os.path.abspath(self.plan_file_path)
            if abs_target == abs_plan:
                return True
        except Exception:
            # 绝对路径比较失败时，退化到文件名和目录前缀判断。
            pass
        if os.path.basename(target_path) == os.path.basename(self.plan_file_path):
            return True
        return ".mewcode/plans/" in target_path
