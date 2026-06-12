"""命令执行工具。

这个工具允许 agent 通过子进程执行 shell 命令，并把标准输出与标准错误
统一收集后返回。它通常用于代码构建、测试、查看目录、运行脚本等场景。
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from mewcode.tools.base import Tool, ToolResult

MAX_TIMEOUT = 600


class Params(BaseModel):
    """Bash 工具的输入参数。"""

    command: str = Field(description="Shell command to execute")
    timeout: int = Field(default=120, description="Timeout in seconds (max 600)")


class Bash(Tool):
    """执行一条 shell 命令，并返回 stdout/stderr。"""

    name = "Bash"
    description = "Execute a shell command and return stdout and stderr."
    params_model = Params
    category = "command"

    async def execute(self, params: Params) -> ToolResult:
        """运行命令，等待执行结束，再整理输出。

        输入:
            params.command: 目标命令。
            params.timeout: 超时时间，超过 MAX_TIMEOUT 会被截断。
        输出:
            ToolResult，内容为标准输出和标准错误的拼接文本。
        """
        timeout = min(params.timeout, MAX_TIMEOUT)

        try:
            proc = await asyncio.create_subprocess_shell(
                params.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            # 命令超时后主动结束子进程，避免后台残留。
            proc.kill()
            await proc.wait()
            return ToolResult(output=f"Error: command timed out after {timeout}s", is_error=True)
        except Exception as exc:
            return ToolResult(output=f"Error executing command: {exc}", is_error=True)

        parts: list[str] = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout.decode(errors='replace')}")
        if stderr:
            parts.append(f"STDERR:\n{stderr.decode(errors='replace')}")
        if not parts:
            parts.append("(no output)")

        output = "\n".join(parts)
        return ToolResult(output=output, is_error=proc.returncode != 0)
