"""工具系统的基础抽象定义。

这个文件位于 mewcode 工具体系的最底层，主要负责两件事：
1. 定义“一个工具至少要具备哪些字段和方法”。
2. 定义模型流式输出过程中会在内部流转的事件对象。

阅读这个文件时，可以先把 Tool 看成“所有具体工具的统一父类”，
再把 ToolResult 和 StreamEvent 看成“工具执行结果”和“模型流式事件”的
统一数据格式。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

# 这些目录通常体量大、价值低、或属于依赖缓存目录，默认搜索工具会跳过。
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".tox", ".mypy_cache"}

# 限制单次工具输出体积，避免工具把过长文本直接塞回模型上下文。
MAX_OUTPUT_CHARS = 10000

# category 用于区分工具的读写风险等级，方便上层做权限或展示分类。
ToolCategory = Literal["read", "write", "command"]


@dataclass
class ToolResult:
    """封装单次工具执行结果。

    output:
        返回给上层调度逻辑和模型的文本内容。
    is_error:
        标记本次执行是否失败。上层可据此决定是否将结果当作错误处理。
    """

    output: str
    is_error: bool = False


class Tool(ABC):
    """所有工具的抽象基类。

    具体工具只要继承这个类，并实现 execute()，就能被统一注册到
    ToolRegistry 里，再由模型通过工具调用机制触发执行。
    """

    name: str
    description: str
    params_model: type[BaseModel]
    category: ToolCategory = "read"
    is_concurrency_safe: bool = False
    is_system_tool: bool = False
    should_defer: bool = False

    @property
    def is_read_only(self) -> bool:
        """判断当前工具是否属于只读工具。"""
        return self.category == "read"

    def get_schema(self) -> dict[str, Any]:
        """生成暴露给模型的工具 schema。

        这里的 schema 会被发送给不同的模型协议，帮助模型知道：
        - 工具名称是什么
        - 工具的作用是什么
        - 工具接受哪些参数
        """
        schema = self.params_model.model_json_schema()
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    @abstractmethod
    async def execute(self, params: BaseModel) -> ToolResult:
        """执行具体工具逻辑。

        输入:
            params: 已通过 Pydantic 校验后的参数对象。
        输出:
            ToolResult: 统一的工具执行结果。
        """
        ...


# --- 流式事件 ---
# 下面这些 dataclass 不属于“工具实现”，而属于“模型返回事件的统一格式”。
# client.py 在读取不同模型 SDK 的流式响应后，会把原始事件翻译成这些对象。
# 上层只需要消费这些统一事件，而不必关心底层到底是 Anthropic 还是 OpenAI。


@dataclass
class TextDelta:
    """模型新吐出的一小段正文文本。"""

    text: str


@dataclass
class ToolCallStart:
    """模型刚开始调用某个工具时的事件。"""

    tool_name: str
    tool_id: str


@dataclass
class ToolCallDelta:
    """模型流式输出工具参数时的增量片段。"""

    text: str


@dataclass
class ToolCallComplete:
    """工具调用参数已经完整拼好时的事件。"""

    tool_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ThinkingDelta:
    """模型思考内容的增量片段。"""

    text: str


@dataclass
class ThinkingComplete:
    """模型一段思考内容结束后的完整结果。"""

    thinking: str
    signature: str


@dataclass
class StreamEnd:
    """一次模型流式响应的结束事件。

    除了 stop_reason 之外，这里还会携带 token 使用情况，
    方便上层统计本轮请求的输入、输出和缓存命中成本。
    """

    stop_reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    # cache_read 表示命中已有缓存前缀的 token 数。
    # cache_creation 表示本轮新写入缓存的 token 数。
    # 有些协议只返回 cache_read，不返回 cache_creation，此时 creation 会保持为 0。
    cache_read: int = 0
    cache_creation: int = 0


StreamEvent = (
    TextDelta
    | ThinkingDelta
    | ThinkingComplete
    | ToolCallStart
    | ToolCallDelta
    | ToolCallComplete
    | StreamEnd
)
