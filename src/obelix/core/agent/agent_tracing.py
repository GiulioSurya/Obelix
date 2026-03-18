"""Tracing functions for BaseAgent.

Encapsulates all tracer span operations so that BaseAgent
does not need to know tracer internals (SpanType, span output
format, conversation history serialization, etc.).

All functions are no-ops when tracer is None.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from obelix.core.model.assistant_message import AssistantMessage
    from obelix.core.model.standard_message import StandardMessage
    from obelix.core.model.system_message import SystemMessage
    from obelix.core.model.tool_message import ToolCall, ToolResult
    from obelix.core.tool.tool_base import Tool
    from obelix.core.tracer.tracer import Tracer
    from obelix.ports.outbound.llm_provider import AbstractLLMProvider


async def start_agent_trace(
    tracer: Tracer | None,
    agent_class_name: str,
    query: str | list[StandardMessage],
) -> bool:
    """Start a trace and agent span. Returns True if this is the root trace."""
    if not tracer:
        return False

    from obelix.core.tracer.context import get_current_trace
    from obelix.core.tracer.models import SpanType

    is_root = get_current_trace() is None
    if is_root:
        await tracer.start_trace(
            name=agent_class_name,
            metadata={"agent_name": agent_class_name},
        )
    await tracer.start_span(
        SpanType.agent,
        agent_class_name,
        input=query
        if isinstance(query, str)
        else (query.content if hasattr(query, "content") else f"{len(query)} messages"),
        metadata={},
    )
    return is_root


async def emit_human_span(
    tracer: Tracer | None,
    query_text: str,
) -> None:
    """Emit a human input span."""
    if not tracer:
        return
    from obelix.core.tracer.models import SpanType

    await tracer.start_span(SpanType.human, "human.input", input=query_text)
    await tracer.end_span(output=query_text)


async def start_llm_span(
    tracer: Tracer | None,
    provider: AbstractLLMProvider,
    conversation_history: list[StandardMessage],
    registered_tools: list[Tool],
) -> None:
    """Start an LLM call span."""
    if not tracer:
        return
    from obelix.core.tracer.models import SpanType

    await tracer.start_span(
        SpanType.llm,
        f"llm.{provider.provider_type}",
        input={
            "message_count": len(conversation_history),
            "tool_count": len(registered_tools),
            "model_id": provider.model_id,
        },
    )


async def end_llm_span(
    tracer: Tracer | None,
    assistant_msg: AssistantMessage,
) -> None:
    """End an LLM call span with response metadata."""
    if not tracer:
        return

    span_output: dict[str, Any] = {
        "content_preview": (
            assistant_msg.content[:200] if assistant_msg.content else None
        ),
        "tool_calls": (
            len(assistant_msg.tool_calls) if assistant_msg.tool_calls else 0
        ),
        "usage": (assistant_msg.usage.model_dump() if assistant_msg.usage else None),
    }
    if assistant_msg.metadata.get("reasoning"):
        span_output["reasoning"] = assistant_msg.metadata["reasoning"]
    await tracer.end_span(output=span_output)


async def start_tool_span(
    tracer: Tracer | None,
    call: ToolCall,
    registered_tools: list[Tool],
) -> None:
    """Start a tool execution span, detecting sub-agent vs regular tool."""
    if not tracer:
        return
    from obelix.core.agent.subagent_wrapper import SubAgentWrapper
    from obelix.core.tracer.models import SpanType

    is_subagent = any(
        isinstance(t, SubAgentWrapper) and t.tool_name == call.name
        for t in registered_tools
    )
    span_type = SpanType.sub_agent if is_subagent else SpanType.tool
    await tracer.start_span(
        span_type,
        call.name,
        input={
            "tool_call_id": call.id,
            "arguments": call.arguments,
        },
    )


async def end_tool_span(
    tracer: Tracer | None,
    result: ToolResult,
) -> None:
    """End a tool execution span."""
    if not tracer:
        return
    from obelix.core.model.tool_message import ToolStatus
    from obelix.core.tracer.models import SpanStatus

    span_status = (
        SpanStatus.error if result.status == ToolStatus.ERROR else SpanStatus.ok
    )
    await tracer.end_span(
        output={
            "result": str(result.result)[:500] if result.result else None,
            "status": str(result.status),
        },
        status=span_status,
        error=result.error,
    )


async def emit_assistant_span(
    tracer: Tracer | None,
    assistant_msg: AssistantMessage,
) -> None:
    """Emit a short assistant response span."""
    if not tracer:
        return
    from obelix.core.tracer.models import SpanType

    await tracer.start_span(
        SpanType.assistant,
        "assistant.response",
        input={"has_tool_calls": bool(assistant_msg.tool_calls)},
    )
    await tracer.end_span(output={"content": assistant_msg.content or None})


async def end_agent_trace(
    tracer: Tracer | None,
    is_root_trace: bool,
    conversation_history: list[StandardMessage],
    system_message: SystemMessage,
    error: str | None = None,
) -> None:
    """End the agent span and (if root) the trace."""
    if not tracer:
        return
    from obelix.core.tracer.context import get_current_span
    from obelix.core.tracer.models import SpanStatus

    status = SpanStatus.error if error else SpanStatus.ok

    span = get_current_span()
    if span:
        conversation = []
        for msg in conversation_history:
            entry: dict[str, Any] = {"role": msg.role.value}
            if hasattr(msg, "content"):
                entry["content"] = msg.content
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if hasattr(msg, "tool_results") and msg.tool_results:
                entry["tool_results"] = [
                    {
                        "tool_name": tr.tool_name,
                        "result": str(tr.result)[:500],
                        "status": tr.status.value,
                    }
                    for tr in msg.tool_results
                ]
            conversation.append(entry)
        span.metadata["conversation_history"] = conversation
        span.metadata["system_prompt"] = system_message.content

    await tracer.end_span(status=status, error=error)
    if is_root_trace:
        await tracer.end_trace(status=status, error=error)
