
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mewcode.conversation import (
    ConversationManager,
    Message,
    ToolResultBlock,
    estimate_tokens,
)
from mewcode.serialization import build_messages

"""上下文管理模块。

这个文件负责控制 Agent 对话历史的体积，避免上下文无限增长后把模型窗口撑满。

整体分成三大块：
1. Layer 1：工具结果预算控制。
   先在本地处理大工具结果，包括落盘、替换预览、裁剪过旧结果。
2. Layer 2：整段会话摘要压缩。
   当前缀历史累计过大时，调用 LLM 把早期对话压成一段结构化摘要。
3. Post-compact recovery：压缩后恢复块。
   摘要会让模型失去很多近期细节，所以这里会把最近读过的文件、最近触发的技能、
   当前可用工具，再重新附加到摘要消息后面，降低“压完就失忆”的概率。

阅读这份代码时，建议顺着这条主线看：
`apply_tool_result_budget()` -> `auto_compact()` -> `build_recovery_attachment()`。
前面的大量辅助函数，基本都是围绕这三步服务的。
"""

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SINGLE_RESULT_CHAR_LIMIT = 50_000
AGGREGATE_CHAR_LIMIT = 200_000
PREVIEW_CHARS = 2_000

KEEP_RECENT_TURNS = 10
OLD_RESULT_SNIP_CHARS = 2_000
SNIPPED_TAG = "<snipped>"

SUMMARY_OUTPUT_RESERVE = 20_000
AUTO_COMPACT_SAFETY_MARGIN = 13_000
MANUAL_COMPACT_SAFETY_MARGIN = 3_000

# Layer 2 "保留近期原文"窗口（对应 Claude Code compact.ts 的
# buildPostCompactMessages messagesToKeep）。压缩时，尾部消息按 token 累计不超过
# KEEP_RECENT_TOKENS、或消息数不少于 MIN_KEEP_MESSAGES（取先满足的条件保底）保留原文，
# 不纳入摘要。累计超过 KEEP_MAX_TOKENS 时停止，防止单条超大消息吞掉整个窗口。
KEEP_RECENT_TOKENS = 10_000
MIN_KEEP_MESSAGES = 5
KEEP_MAX_TOKENS = 40_000

# 前缀 token 数低于此阈值时不值得做摘要——摘要往返的开销比回收的空间还大，
# 退化为不压缩、保留原始历史（避免「压了个寂寞」）。
MIN_SUMMARIZE_PREFIX_TOKENS = 2_000

PERSISTED_TAG = "<persisted-output>"

SESSION_SUBDIR = ".mewcode/session/tool-results"


# ---------------------------------------------------------------------------
# 事件
# ---------------------------------------------------------------------------


@dataclass
class CompactBoundary:
    """Layer 2 压缩的结构化结果，上交给 session 层处理。

    `summary` 是大模型对被摘要前缀生成的摘要；`keep` 是 auto_compact 原样保留、
    未做改动的近期尾部消息。session 层（持有 sessionId / 文件句柄）会把二者一起
    内联进一条 compact_boundary 记录，这样 resume 时就能重建压缩后的状态。
    用这种方式把写操作解耦出去，能让 auto_compact 保持纯粹、不依赖任何 session。
    """

    summary: str
    keep: list[Message]


@dataclass
class CompactEvent:
    before_tokens: int
    # 摘要成功时填充，调用方可据此持久化 compact_boundary 记录。
    # 未产出摘要时为 None。
    boundary: CompactBoundary | None = None


# ---------------------------------------------------------------------------
# 内容替换状态 — Design B（决策冻结，不做原地修改）
# ---------------------------------------------------------------------------

@dataclass
class ContentReplacementState:
    seen_ids: set[str] = field(default_factory=set)
    replacements: dict[str, str] = field(default_factory=dict)


@dataclass
class ContentReplacementRecord:
    tool_use_id: str
    replacement: str
    kind: str = "tool-result"


def create_replacement_state() -> ContentReplacementState:
    """创建一份空的内容替换状态。

    输入:
        无。

    输出:
        返回新的 `ContentReplacementState`。

    说明:
        Layer 1 不直接原地篡改会话历史，而是通过这份状态记录：
        哪些 `tool_use_id` 已经见过，哪些结果已经被替换成了 preview。
        这样预算控制逻辑会更稳定，也更方便恢复。
    """
    return ContentReplacementState()


def clone_replacement_state(src: ContentReplacementState) -> ContentReplacementState:
    """复制内容替换状态。

    输入:
        src: 原始替换状态。

    输出:
        返回一份可独立修改的新状态对象。

    说明:
        这里显式复制集合和字典，避免后续修改时和旧状态共享底层容器。
    """
    return ContentReplacementState(
        seen_ids=set(src.seen_ids),
        replacements=dict(src.replacements),
    )


REPLACEMENT_RECORDS_FILENAME = "replacement_records.jsonl"


def append_replacement_records(
    session_dir: Path, records: list[ContentReplacementRecord]
) -> None:
    """把本轮新增的替换记录追加写入磁盘。

    输入:
        session_dir: 当前会话目录。
        records: 本轮新增的替换记录列表。

    输出:
        无返回值。

    说明:
        使用 JSONL 追加写，而不是每次重写整个文件，适合按轮次持续积累替换决策。
    """
    if not records:
        return
    path = session_dir / REPLACEMENT_RECORDS_FILENAME
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "kind": r.kind,
                "tool_use_id": r.tool_use_id,
                "replacement": r.replacement,
            }, ensure_ascii=False) + "\n")


def load_replacement_records(session_dir: Path) -> list[ContentReplacementRecord]:
    """从 session 目录读取历史替换记录。

    输入:
        session_dir: 当前会话目录。

    输出:
        返回历史 `ContentReplacementRecord` 列表。
    """
    path = session_dir / REPLACEMENT_RECORDS_FILENAME
    if not path.exists():
        return []
    out: list[ContentReplacementRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(ContentReplacementRecord(
                kind=obj.get("kind", "tool-result"),
                tool_use_id=obj["tool_use_id"],
                replacement=obj["replacement"],
            ))
    return out


def reconstruct_replacement_state(
    messages: list[Message],
    records: list[ContentReplacementRecord],
    inherited_replacements: Mapping[str, str] | None = None,
) -> ContentReplacementState:
    """根据历史消息和持久化记录重建替换状态。

    输入:
        messages: 当前恢复后的消息历史。
        records: 从磁盘读取到的历史替换记录。
        inherited_replacements: 可选的上层补充替换映射。

    输出:
        返回重建后的 `ContentReplacementState`。

    说明:
        只有当前历史中仍然存在的 `tool_use_id` 才应该进入状态，
        已经脱离当前历史的旧记录不应继续污染新的上下文。
    """
    state = create_replacement_state()
    candidate_ids: set[str] = set()
    for msg in messages:
        for tr in msg.tool_results:
            candidate_ids.add(tr.tool_use_id)
    state.seen_ids.update(candidate_ids)
    for r in records:
        if r.kind == "tool-result" and r.tool_use_id in candidate_ids:
            state.replacements[r.tool_use_id] = r.replacement
    if inherited_replacements:
        for tool_use_id, replacement in inherited_replacements.items():
            if tool_use_id in candidate_ids and tool_use_id not in state.replacements:
                state.replacements[tool_use_id] = replacement
    return state


# ---------------------------------------------------------------------------
# Session 目录管理
# ---------------------------------------------------------------------------

def ensure_session_dir(work_dir: str) -> Path:
    """确保工具结果落盘目录存在。

    输入:
        work_dir: 当前工作目录。

    输出:
        返回 session 目录路径。

    说明:
        Layer 1 的大结果落盘和 Layer 2 的清理步骤都会用到这里返回的目录。
    """
    session_dir = Path(work_dir) / SESSION_SUBDIR
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def cleanup_tool_results(session_dir: Path) -> None:
    """清空工具结果落盘目录，并重建为空目录。

    输入:
        session_dir: 当前会话目录。

    输出:
        无返回值。

    说明:
        Layer 2 压缩完成后，旧的大结果文件引用通常已经失效，直接整目录清理最简单。
    """
    if session_dir.exists():
        shutil.rmtree(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Layer 1：大型工具结果落盘
# ---------------------------------------------------------------------------

def persist_tool_result(tool_use_id: str, content: str, session_dir: Path) -> Path:
    """把超大工具结果写入磁盘。

    输入:
        tool_use_id: 工具调用唯一标识。
        content: 工具输出全文。
        session_dir: 当前会话目录。

    输出:
        返回写入目标路径。

    说明:
        使用 `os.O_EXCL` 确保文件已存在时创建失败，从而让这一步具备幂等性。
        对同一个 `tool_use_id`，第一次写成功后，后续重复调用只会复用原文件。
    """
    file_path = session_dir / f"{tool_use_id}.txt"
    try:
        fd = os.open(str(file_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except FileExistsError:
        pass
    return file_path


def make_persisted_preview(content: str, file_path: Path) -> str:
    """为已落盘的大结果构造一段可放回上下文的预览文本。

    输入:
        content: 工具输出全文。
        file_path: 完整内容落盘后的文件路径。

    输出:
        返回带 `PERSISTED_TAG` 的预览字符串。

    说明:
        这段预览既告诉模型“全文已移出上下文”，也保留少量开头片段，
        让模型能快速判断是否值得再用 `ReadFile` 回读原文。
    """
    size_kb = len(content.encode("utf-8")) // 1024
    preview = content[:PREVIEW_CHARS]
    return (
        f"{PERSISTED_TAG}\n"
        f"输出太大（{size_kb}KB），完整内容已保存到：\n"
        f"{file_path}\n"
        f"\n"
        f"预览（前 2KB）：\n"
        f"{preview}\n"
        f"</persisted-output>"
    )


def _count_turns(messages: list[Message]) -> int:
    """统计会话轮次数。

    输入:
        messages: 消息列表。

    输出:
        返回轮次数。

    说明:
        这里把“没有工具调用的 assistant 消息”视为一轮结束标记，
        因为它通常代表这一轮任务处理已经形成自然语言回复。
    """
    count = 0
    for m in messages:
        if m.role == "assistant" and not m.tool_uses:
            count += 1
    return count


# 这个辅助函数只做一件事：在保留消息其余字段不变的前提下，
# 用新的 tool_results 重新构造一条 Message。
# Layer 1 会多次替换工具结果内容，如果直接原地改旧对象，后续恢复状态和调试都会更绕，
# 所以这里统一走“复制旧消息，换新结果”的方式。
def _copy_message_with_results(
    msg: Message, new_tool_results: list[ToolResultBlock]
) -> Message:
    """复制消息对象，并仅替换其工具结果列表。"""
    return Message(
        role=msg.role,
        content=msg.content,
        tool_uses=list(msg.tool_uses),
        tool_results=new_tool_results,
        thinking_blocks=list(msg.thinking_blocks),
    )


# 这是 Layer 1 的第三趟扫描。
# 前两趟主要解决“当前消息里工具结果太大”，这一趟解决“老历史里的长结果还在持续占空间”。
# 处理方式更激进：最近若干轮尽量不动，更早轮次里的长结果只保留极短预览，
# 不再像 persisted preview 那样保留可回读全文。
def _snip_stale_messages(
    history: list[Message],
) -> list[Message]:
    """裁剪过旧轮次中的长工具结果。

    输入:
        history: 当前消息历史。

    输出:
        返回裁剪后的历史列表。

    说明:
        这是 Layer 1 的 Pass 3。
        前两步优先解决“当前消息太大”，这一步则解决“很久以前的长结果还在占空间”。
    """
    """构造压缩后的摘要入口消息。

    输入:
        summary: 早期对话摘要。
        attachment: 压缩后恢复块。
        has_keep_tail: 是否仍保留了最近原文尾部。
        transcript_path: 压缩前完整会话转录路径。

    输出:
        返回新的消息列表，目前包含一条承载摘要的 `user` 消息。

    说明:
        这条消息是压缩后的新起点，用来告诉模型：
        早期历史已经浓缩成摘要，近期原文可能还保留，完整细节需要时应重新读取。
    """
    """裁剪过旧轮次中的长工具结果。

    输入:
        history: 当前消息历史。

    输出:
        返回裁剪后的历史列表。

    说明:
        这是 Layer 1 的 Pass 3。
        前两步优先解决“当前消息太大”，这一步则解决“很久以前的长结果还在占空间”。
    """
    total_turns = _count_turns(history)
    if total_turns <= KEEP_RECENT_TURNS:
        return history

    out: list[Message] = []
    turns_seen = 0
    old_boundary = total_turns - KEEP_RECENT_TURNS

    for msg in history:
        if msg.role == "assistant" and not msg.tool_uses:
            turns_seen += 1
        if turns_seen > old_boundary or not msg.tool_results:
            out.append(msg)
            continue

        new_results: list[ToolResultBlock] = []
        changed = False
        for tr in msg.tool_results:
            if (
                tr.content.startswith(SNIPPED_TAG)
                or tr.content.startswith(PERSISTED_TAG)
                or len(tr.content) <= OLD_RESULT_SNIP_CHARS
            ):
                new_results.append(tr)
                continue
            preview = tr.content[:200]
            orig_len = len(tr.content)
            new_content = (
                f"{SNIPPED_TAG}\n"
                f"(旧结果已裁剪，原始长度 {orig_len} 字符)\n"
                f"{preview}\n"
                f"… (snipped)"
            )
            new_results.append(ToolResultBlock(
                tool_use_id=tr.tool_use_id,
                content=new_content,
                is_error=tr.is_error,
            ))
            changed = True

        out.append(_copy_message_with_results(msg, new_results) if changed else msg)

    return out


def apply_tool_result_budget(
    conversation: ConversationManager,
    session_dir: Path,
    state: ContentReplacementState,
) -> tuple[ConversationManager, list[ContentReplacementRecord]]:
    """
    Design B: 不 mutate 原 conversation。

    返回一个新的 ConversationManager，其中 tool_result.content 已根据 state.replacements
    应用了决策，并对本轮 fresh 候选执行了 Pass 1（单条超限）+ Pass 2（聚合超限）。
    Pass 3（陈旧裁剪）在新 history 上跑，仍然 stateless（边界 drift 是已知 trade-off）。

    state 会被 mutate：本轮新决定的 id 进入 seen_ids，新决定替换的 id 进入 replacements。
    """
    # 这是 Layer 1 的主入口。
    # 整体策略是“读旧历史 -> 做替换决策 -> 生成新历史”，
    # 而不是直接在旧 conversation 上原地改内容。
    new_records: list[ContentReplacementRecord] = []
    new_history: list[Message] = []

    for msg in conversation.history:
        if not msg.tool_results:
            new_history.append(msg)
            continue

        decisions: dict[str, str] = {}
        fresh: list[ToolResultBlock] = []

        for tr in msg.tool_results:
            if tr.tool_use_id in state.replacements:
                # 之前已经替换过，直接复用旧决策，保证同一结果在后续轮次里表现一致。
                decisions[tr.tool_use_id] = state.replacements[tr.tool_use_id]
            elif tr.tool_use_id in state.seen_ids:
                # 已经见过但没做过替换，说明之前的决策是“原文保留”。
                decisions[tr.tool_use_id] = tr.content
            elif tr.content.startswith(PERSISTED_TAG):
                # 已被外部（如某些工具本身）打上 persisted-output 标签 —— 视为已知决策
                state.seen_ids.add(tr.tool_use_id)
                state.replacements[tr.tool_use_id] = tr.content
                decisions[tr.tool_use_id] = tr.content
                new_records.append(ContentReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=tr.content,
                ))
            else:
                fresh.append(tr)

        # Pass 1：单条超限
        persisted_p1: set[str] = set()
        for tr in fresh:
            if len(tr.content) > SINGLE_RESULT_CHAR_LIMIT:
                # Pass 1 先解决极端大的单条结果，避免“一条就炸上下文”。
                fp = persist_tool_result(tr.tool_use_id, tr.content, session_dir)
                preview = make_persisted_preview(tr.content, fp)
                decisions[tr.tool_use_id] = preview
                state.replacements[tr.tool_use_id] = preview
                state.seen_ids.add(tr.tool_use_id)
                new_records.append(ContentReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=preview,
                ))
                persisted_p1.add(tr.tool_use_id)

        # Pass 2：聚合超限
        remaining = [tr for tr in fresh if tr.tool_use_id not in persisted_p1]
        total = sum(len(c) for c in decisions.values()) + sum(
            len(tr.content) for tr in remaining
        )
        if total > AGGREGATE_CHAR_LIMIT:
            ranked = sorted(remaining, key=lambda tr: len(tr.content), reverse=True)
            for tr in ranked:
                if total <= AGGREGATE_CHAR_LIMIT:
                    break
                # Pass 2 按长度降序替换，优先处理最长结果，回收空间效率最高。
                fp = persist_tool_result(tr.tool_use_id, tr.content, session_dir)
                preview = make_persisted_preview(tr.content, fp)
                old_len = len(tr.content)
                decisions[tr.tool_use_id] = preview
                state.replacements[tr.tool_use_id] = preview
                state.seen_ids.add(tr.tool_use_id)
                new_records.append(ContentReplacementRecord(
                    tool_use_id=tr.tool_use_id, replacement=preview,
                ))
                total -= old_len - len(preview)

        # 剩余未替换的 fresh 标记为"已见但未替换"
        for tr in fresh:
            if tr.tool_use_id not in state.replacements:
                # 没被替换也要记成 seen，避免下一轮又把它当成全新结果重新决策。
                state.seen_ids.add(tr.tool_use_id)
                decisions[tr.tool_use_id] = tr.content

        # 生成新的 tool_results，保持原始顺序
        new_tool_results = [
            ToolResultBlock(
                tool_use_id=tr.tool_use_id,
                content=decisions[tr.tool_use_id],
                is_error=tr.is_error,
            )
            for tr in msg.tool_results
        ]
        new_history.append(_copy_message_with_results(msg, new_tool_results))

    # Pass 3：在新 history 上裁剪过期结果（无状态；边界漂移是已知 trade-off）
    new_history = _snip_stale_messages(new_history)

    new_conv = ConversationManager()
    new_conv.history = new_history
    new_conv.env_injected = conversation.env_injected
    new_conv.ltm_injected = conversation.ltm_injected
    new_conv.last_input_tokens = conversation.last_input_tokens
    new_conv.baseline_tokens = conversation.baseline_tokens
    new_conv.anchor_count = conversation.anchor_count

    return new_conv, new_records


# ---------------------------------------------------------------------------
# Layer 2：全对话摘要（Auto-Compact）
# ---------------------------------------------------------------------------

def compute_compact_threshold(context_window: int, manual: bool = False) -> int:
    """计算触发 Layer 2 压缩的 token 阈值。

    输入:
        context_window: 当前模型的上下文窗口大小。
        manual: 是否手动触发压缩。

    输出:
        返回触发压缩的 token 阈值。

    说明:
        先预留摘要输出空间，再减掉安全边距；自动模式更保守，手动模式更激进。
    """
    effective = context_window - SUMMARY_OUTPUT_RESERVE
    margin = MANUAL_COMPACT_SAFETY_MARGIN if manual else AUTO_COMPACT_SAFETY_MARGIN
    return effective - margin


def should_auto_compact(last_input_tokens: int, context_window: int) -> bool:
    """快速判断当前是否达到自动压缩阈值。"""
    return last_input_tokens >= compute_compact_threshold(context_window)


SUMMARY_PROMPT = """\
你是一个对话摘要助手。你只能输出纯文本，不能调用任何工具。

请对下面的对话生成一份结构化摘要。

先在 <analysis> 标签中梳理对话中发生了什么（这部分会被丢弃），然后在 <summary> 标签中输出正式摘要。

<summary> 必须包含以下 9 个部分：

1. **主要请求和意图**：用户到底想做什么
2. **关键技术概念**：讨论过的重要技术点
3. **文件和代码段**：涉及哪些文件，关键代码片段要保留
4. **错误和修复**：遇到了什么错，怎么解决的
5. **问题解决过程**：解决问题的思路和方法
6. **所有用户消息**：用户说过的所有非工具结果的话（原文保留，不可改写！）
7. **待办任务**：还没完成的事
8. **当前工作**：最近在做什么（要最详细）
9. **可能的下一步**：接下来打算做什么

提醒：不要调用任何工具。工具调用会被拒绝，你会失败。只输出纯文本。"""


def extract_summary(llm_output: str) -> str:
    """从 LLM 输出中提取正式摘要段。

    输入:
        llm_output: 摘要模型返回的原始文本。

    输出:
        若存在 `<summary>...</summary>`，返回其中内容；
        否则返回完整输出作为兜底。
    """
    start = llm_output.find("<summary>")
    end = llm_output.find("</summary>")
    if start == -1 or end == -1:
        return llm_output
    return llm_output[start + len("<summary>"):end].strip()


# 这里构造的是“压缩后的新会话开头”。
# 它不会把整段历史重新展开，而是只放：
# 1. 早期历史的摘要
# 2. 近期原文是否还保留的提示
# 3. 完整细节需要时去哪里重新读取
# 4. 可选的恢复块（最近文件/技能/工具）
def build_compact_messages(
    summary: str,
    attachment: str = "",
    has_keep_tail: bool = False,
    transcript_path: str = "",
) -> list[Message]:
    """构造压缩后的摘要入口消息。

    输入:
        summary: 早期对话摘要。
        attachment: 压缩后恢复块。
        has_keep_tail: 是否仍保留了最近原文尾部。
        transcript_path: 压缩前完整会话转录路径。

    输出:
        返回新的消息列表，目前包含一条承载摘要的 `user` 消息。

    说明:
        这条消息是压缩后的新起点，用来告诉模型：
        早期历史已经浓缩成摘要，近期原文可能还保留，完整细节需要时应重新读取。
    """
    content = "本次会话延续自之前的对话，因上下文空间不足进行了压缩。以下是早期对话的摘要：\n\n" + summary
    if has_keep_tail:
        content += "\n\n近期消息已原样保留。"
    if transcript_path:
        content += f"\n\n如果你需要压缩前的具体细节（代码片段、报错信息等），请用 ReadFile 读取完整会话记录：{transcript_path}"
    if attachment:
        content += "\n\n---\n\n" + attachment
    return [
        Message(role="user", content=content),
    ]


# ---------------------------------------------------------------------------
# 压缩后恢复状态
# ---------------------------------------------------------------------------

# 追加到摘要 user 消息的恢复附件限制。compact 会清空工作对话；
# 没有这些快照，模型会忘记刚读过哪些文件、正在执行哪个 skill 的 SOP。
RECOVERY_FILE_LIMIT = 5
RECOVERY_TOKENS_PER_FILE = 5_000
RECOVERY_SKILLS_BUDGET = 25_000
RECOVERY_TOKENS_PER_SKILL = 5_000
_RECOVERY_CHARS_PER_TOKEN = 3.5


@dataclass
class FileReadRecord:
    path: str
    content: str
    timestamp: float


@dataclass
class SkillInvocationRecord:
    name: str
    body: str
    timestamp: float


class RecoveryState:
    """能在 Layer 2 压缩中存活下来的 per-agent 快照。

    记录 ReadFile 返回的字节内容，以及各个 skill 被调用时附带的 SOP 正文。
    这些记录会被重新附加到摘要的 user 消息上，这样即便对话记录被压缩清空，
    模型仍然保有可用的工作上下文。
    """

    def __init__(self) -> None:
        """初始化压缩后恢复状态容器。

        输入:
            无。

        输出:
            无返回值。

        说明:
            内部记录最近文件快照和技能快照，并用锁保护，避免并发写入时状态错乱。
        """
        self._lock = threading.Lock()
        self._files: dict[str, FileReadRecord] = {}
        self._skills: dict[str, SkillInvocationRecord] = {}

    def record_file_read(self, path: str, content: str) -> None:
        """记录一次文件读取快照。"""
        if not path:
            return
        with self._lock:
            self._files[path] = FileReadRecord(
                path=path, content=content, timestamp=time.time()
            )

    def record_skill_invocation(self, name: str, body: str) -> None:
        """记录一次技能触发快照。"""
        if not name:
            return
        with self._lock:
            self._skills[name] = SkillInvocationRecord(
                name=name, body=body, timestamp=time.time()
            )

    def snapshot_files(self, limit: int) -> list[FileReadRecord]:
        """按最近时间倒序导出文件快照。"""
        with self._lock:
            records = list(self._files.values())
        records.sort(key=lambda r: r.timestamp, reverse=True)
        if limit > 0:
            records = records[:limit]
        return records

    def snapshot_skills(self) -> list[SkillInvocationRecord]:
        """按最近时间倒序导出技能快照。"""
        with self._lock:
            records = list(self._skills.values())
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records


def _approx_tokens(s: str) -> int:
    """用固定字符/token 比例做近似 token 估算。"""
    if not s:
        return 0
    return int(len(s) / _RECOVERY_CHARS_PER_TOKEN)


def _truncate_by_tokens(s: str, token_budget: int) -> str:
    """按近似 token 预算截断文本，并追加截断提示。"""
    if token_budget <= 0 or not s:
        return s
    if _approx_tokens(s) <= token_budget:
        return s
    max_chars = int(token_budget * _RECOVERY_CHARS_PER_TOKEN)
    if max_chars <= 0 or max_chars >= len(s):
        return s
    return s[:max_chars] + "\n… (内容已截断)"


def _first_line(s: str) -> str:
    """提取首个非空行。"""
    for line in s.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def build_recovery_attachment(
    state: RecoveryState | None,
    tool_schemas: list[Mapping[str, Any]] | None,
) -> str:
    """渲染压缩后附件的四个小节。

    没有任何值得附加的内容时返回 ""，让调用方保持摘要消息干净。
    `tool_schemas` 应当是 agent 在下一次请求中将要发送的 schema —— 这里用其中的
    名称和描述来提醒模型当前都接入了哪些工具。
    """
    # 这里生成的不是给人类看的长文档，而是给压缩后的模型做“工作记忆回填”的紧凑块。
    sections: list[str] = []

    if state is not None:
        files = state.snapshot_files(RECOVERY_FILE_LIMIT)
        if files:
            buf = ["## 最近读过的文件\n",
                   "以下快照是文件读取工具上次返回的内容。如需当前字节请重新读取。\n"]
            for rec in files:
                # 每个文件快照都要受预算限制，避免恢复块反过来把新上下文再撑满。
                content = _truncate_by_tokens(rec.content, RECOVERY_TOKENS_PER_FILE)
                ts = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec.timestamp)
                )
                buf.append(f"### {rec.path}  (read {ts})\n")
                buf.append("```\n")
                buf.append(content)
                if not content.endswith("\n"):
                    buf.append("\n")
                buf.append("```\n")
            sections.append("".join(buf))

        skills = state.snapshot_skills()
        if skills:
            buf = ["## 已激活的技能\n",
                   "下列技能在本会话中被调用过，其触发条件仍然适用。\n"]
            used = 0
            emitted = False
            for sk in skills:
                # 技能正文可能很长，因此一边累计预算一边决定是否继续输出。
                body = _truncate_by_tokens(sk.body, RECOVERY_TOKENS_PER_SKILL)
                tokens = _approx_tokens(body) + _approx_tokens(sk.name) + 8
                if used + tokens > RECOVERY_SKILLS_BUDGET:
                    break
                used += tokens
                buf.append(f"### {sk.name}\n\n{body}\n")
                emitted = True
            if emitted:
                sections.append("".join(buf))

    if tool_schemas:
        buf = ["## 可用工具\n",
               "你仍然可以调用以下工具，需要时直接发起调用即可：\n"]
        for t in tool_schemas:
            # 工具列表只保留名字和描述首行，目的是提醒“还能调用什么”，而不是再次展开完整 schema。
            name = t.get("name") if isinstance(t, Mapping) else None
            if not name:
                continue
            desc = t.get("description", "") if isinstance(t, Mapping) else ""
            desc = _first_line(desc or "")
            if desc:
                buf.append(f"- {name} — {desc}\n")
            else:
                buf.append(f"- {name}\n")
        sections.append("".join(buf))

    if not sections:
        return ""

    sections.append(
        "## 提示\n\n以上恢复的上下文是重建的。若需要原文代码、错误信息或用户原话，"
        "请用文件读取工具重新读取，不要根据摘要猜测细节。\n"
    )
    return "\n".join(sections)


def _group_messages_by_turn(messages: list[Message]) -> list[list[Message]]:
    """按“对话轮次”对消息分组。

    输入:
        messages: 待分组消息列表。

    输出:
        返回二维列表，每个子列表代表一轮对话。

    说明:
        摘要失败后做降级重试时，会按轮次整体丢掉最老的一部分历史，
        这样比随便砍几条消息更不容易把一轮工作拆散。
    """
    groups: list[list[Message]] = []
    current: list[Message] = []
    for msg in messages:
        current.append(msg)
        if msg.role == "assistant" and not msg.tool_uses:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _message_tokens(msg: Message) -> int:
    """估算单条消息的 token 数，复用共享的字符数启发式算法。"""
    return estimate_tokens([msg])


def _compute_keep_start_index(messages: list[Message]) -> int:
    """决定压缩时尾部要原样保留多少条消息。

    从尾部向头部遍历 `messages`，逐条累加 token 估算值。只要还有任一保底条件
    未满足——累计 token 尚未达到 KEEP_RECENT_TOKENS，或保留的消息数仍少于
    MIN_KEEP_MESSAGES——当前消息就会被纳入保留窗口；但一旦纳入下一条消息会使
    保留总量超过 KEEP_MAX_TOKENS，遍历立即停止（这样单条超大的尾部消息就不会把
    整个 history 都拖进窗口）。

    返回第一条被保留消息的下标（keepStartIndex）。原始遍历结束后，必要时会把这个
    下标往前挪，确保被保留的 tool_result 不会和它对应的 tool_use 被拆散——
    参见 `_align_keep_start_to_tool_pair`。
    """
    # 这个函数决定 Layer 2 压缩时“从哪里切开”：
    # 前缀做摘要，尾部原文保留。切得太靠前会回收太少空间，切得太靠后又会损失近期细节。
    n = len(messages)
    if n == 0:
        return 0

    kept_tokens = 0
    kept_count = 0
    keep_start = n  # 尚未保留任何消息

    for i in range(n - 1, -1, -1):
        tok = _message_tokens(messages[i])

        # 在已经保留了至少一条消息的前提下，如果纳入当前消息会突破硬上限则停止
        # （但绝不拒绝保留最后一条消息，即使它单独就超限）。
        if kept_count > 0 and kept_tokens + tok > KEEP_MAX_TOKENS:
            break

        kept_tokens += tok
        kept_count += 1
        keep_start = i

        # 保底条件已满足（token 下限或消息条数下限达到其一）：
        # 近期原文保留足够了，停止回溯。
        if kept_tokens >= KEEP_RECENT_TOKENS or kept_count >= MIN_KEEP_MESSAGES:
            break

    return _align_keep_start_to_tool_pair(messages, keep_start)


def _align_keep_start_to_tool_pair(messages: list[Message], keep_start: int) -> int:
    """把 keep_start 往前挪，确保我们绝不会保留一个孤立的 tool_result。

    携带 tool_results 的 user 消息，会和它前面那条发起对应 tool_uses 的 assistant
    消息配成一对。如果 keep_start 正好落在这样一条 user 消息上，就把它往前回退到
    （至少）配对的那条 assistant 消息，让 tool_use 与 tool_result 的配对关系保持完整。
    宁可多保留一对，也不要只保留半对（一个模型无法归属到任何调用的悬空 tool_result）。
    """
    # 工具调用和工具结果最好成对保留，避免摘要边界把它们拆开后留下悬空结果。
    while 0 < keep_start < len(messages):
        msg = messages[keep_start]
        if msg.role == "user" and msg.tool_results:
            prev = messages[keep_start - 1]
            if prev.role == "assistant" and prev.tool_uses:
                keep_start -= 1
                continue
        break
    return keep_start


def _prefix_too_small_to_compact(prefix: list[Message]) -> bool:
    """当摘要 `prefix` 能回收的空间太少、不值得做时返回 True。"""
    if not prefix:
        return True
    return estimate_tokens(prefix) < MIN_SUMMARIZE_PREFIX_TOKENS


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------


@dataclass
class CompactCircuitBreaker:
    """自动压缩熔断器。

    说明:
        Layer 2 要调用 LLM 生成整段摘要，这一步成本更高，也可能连续失败。
        熔断器的作用是在连续失败达到阈值后暂时停掉自动压缩，避免无意义重试。
    """
    max_failures: int = 3
    consecutive_failures: int = field(default=0, init=False)

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0


    def is_open(self) -> bool:
        return self.consecutive_failures >= self.max_failures


# ---------------------------------------------------------------------------
# Auto-compact 编排器
# ---------------------------------------------------------------------------

async def auto_compact(
    conversation: ConversationManager,
    client: Any,
    context_window: int,
    session_dir: Path,
    protocol: str = "anthropic",
    manual: bool = False,
    breaker: CompactCircuitBreaker | None = None,
    recovery: RecoveryState | None = None,
    tool_schemas: list[Mapping[str, Any]] | None = None,
    transcript_path: str = "",
) -> CompactEvent | str | None:
    """执行 Layer 2 整段会话压缩。

    输入:
        conversation: 当前会话管理器。
        client: 用于生成摘要的 LLM 客户端。
        context_window: 当前模型上下文窗口大小。
        session_dir: 工具结果落盘目录。
        protocol: 当前协议类型。
        manual: 是否手动触发压缩。
        breaker: 自动压缩熔断器。
        recovery: 压缩后恢复状态。
        tool_schemas: 当前可用工具列表。
        transcript_path: 压缩前完整会话记录路径。

    输出:
        成功时返回 `CompactEvent`；
        失败或熔断时返回说明字符串；
        当前无需压缩时返回 `None`。

    说明:
        这是整个模块最核心的编排函数：判断阈值、生成摘要、拼接恢复块、替换历史都在这里完成。
    """
    threshold = compute_compact_threshold(context_window, manual=manual)

    # 以真实 API 用量为锚点做阈值判断：current_tokens() 返回上次计费基准
    # （input + cache_read + cache_creation + output）加上锚点之后新增消息的
    # 字符估算。冷启动或刚压缩清空锚点时，退化为对整个 history 做字符估算。
    current = conversation.current_tokens()

    if not manual and current < threshold:
        # 自动模式没到阈值就直接跳过，避免过早用摘要替换仍然有价值的原文。
        return None

    if not manual and breaker is not None and breaker.is_open():
        return "自动压缩已熔断（连续失败 3 次），请手动处理或使用 /compact"

    before_tokens = current

    # 决定保留多少尾部消息原文。只有前缀 messages[:keep_start] 会被摘要；
    # messages[keep_start:] 原样保留，让模型看到近期原文而非靠有损摘要复述。
    keep_start = _compute_keep_start_index(conversation.history)
    # 这里只对较早前缀做摘要；`keep_tail` 代表最近的原文窗口，会原样保留。
    to_summarize = conversation.history[:keep_start]
    keep_tail = conversation.history[keep_start:]

    # 待摘要的前缀太小时退化为不压缩——要么全部消息都落在保留窗口内
    # （keep_start <= 0），要么摘要回收的 token 还不够摘要本身的开销。
    if keep_start <= 0 or _prefix_too_small_to_compact(to_summarize):
        return None

    messages_for_summary = build_messages(list(to_summarize), protocol)

    summary_messages: list[dict[str, Any]] = [
        {"role": "user", "content": SUMMARY_PROMPT},
    ]
    summary_messages.extend(messages_for_summary)
    summary_messages.append(
        {"role": "user", "content": "请根据以上对话生成结构化摘要。记住：不要调用任何工具。"}
    )

    summary_conv = ConversationManager()
    summary_conv.history = [
        Message(role="user", content=SUMMARY_PROMPT),
    ]
    # 只摘要前缀；保留的尾部在下面重建时原样拼回。
    for msg in to_summarize:
        summary_conv.history.append(msg)
    summary_conv.history.append(
        Message(role="user", content="请根据以上对话生成结构化摘要。记住：不要调用任何工具。")
    )

    max_retries = 3
    llm_output: str | None = None

    for attempt in range(max_retries):
        try:
            from mewcode.tools.base import StreamEnd, StreamEvent, TextDelta

            collected_text = ""
            async for event in client.stream(summary_conv, system=SUMMARY_PROMPT):
                if isinstance(event, TextDelta):
                    collected_text += event.text
                elif isinstance(event, StreamEnd):
                    pass
            llm_output = collected_text
            break

        except Exception as e:
            err_msg = str(e).lower()
            if "prompt" in err_msg and "long" in err_msg or "too many" in err_msg:
                # 失败时按轮次整体丢弃最老的 20%，避免把一轮工作拆散。
                groups = _group_messages_by_turn(summary_conv.history[1:-1])
                drop_count = max(1, len(groups) // 5)
                remaining = groups[drop_count:]
                summary_conv.history = (
                    [summary_conv.history[0]]
                    + [m for g in remaining for m in g]
                    + [summary_conv.history[-1]]
                )
                continue
            if breaker is not None:
                breaker.record_failure()
            return f"摘要生成失败: {e}"

    if llm_output is None:
        if breaker is not None:
            breaker.record_failure()
        return "摘要生成失败：多次重试后仍超出上下文限制"

    summary = extract_summary(llm_output)
    attachment = build_recovery_attachment(recovery, tool_schemas)
    # 重建 = 摘要(user) + 尾部原文。
    new_messages = build_compact_messages(
        summary,
        attachment=attachment,
        has_keep_tail=bool(keep_tail),
        transcript_path=transcript_path,
    )
    # keep_tail 是近期原文窗口，这部分不摘要，直接拼回压缩后的历史。
    new_messages = new_messages + list(keep_tail)

    # replace_history 替换为重建后的对话并将用量锚点清零
    # （baseline_tokens / anchor_count / last_input_tokens），这是必须的：
    # 旧的 anchor_count 对应压缩前的消息列表，现在已无意义，
    # 不清零会导致 current_tokens() 对增量的估算出错。
    # 下一次 API 响应会基于重建后的 history 重新锚定。
    conversation.replace_history(new_messages)
    cleanup_tool_results(session_dir)

    if breaker is not None:
        breaker.record_success()

    # 将结构化的 boundary（摘要 + 保留的尾部原文）交给 session 层，
    # 由它持久化为一条 compact_boundary 记录。keep tail 就是拼回重建 history 的那段。
    return CompactEvent(
        before_tokens=before_tokens,
        boundary=CompactBoundary(summary=summary, keep=list(keep_tail)),
    )
