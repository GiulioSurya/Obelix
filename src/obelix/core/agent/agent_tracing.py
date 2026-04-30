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


async def accumulate_llm_call(
    tracer: Tracer | None,
    assistant_msg: AssistantMessage,
    provider_type: str,
    model_id: str,
    duration_ms: float,
) -> None:
    """Fold a single LLM call into the current agent span's metadata.llm_usage.

    No-op if ``tracer`` is None or there is no current span. The current span
    is expected to be an ``agent`` span opened by :func:`start_agent_trace`.

    Replaces the per-call ``llm`` span (dropped in Task 10) — rather than
    emitting a span per LLM invocation, we roll every call's usage into an
    aggregate stored on the enclosing agent span.
    """
    if not tracer:
        return

    from obelix.core.tracer.context import get_current_span

    span = get_current_span()
    if span is None:
        return

    usage_dict = span.metadata.setdefault(
        "llm_usage",
        {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    usage_dict["calls"] += 1
    if assistant_msg.usage:
        tin = getattr(assistant_msg.usage, "input_tokens", 0) or 0
        tout = getattr(assistant_msg.usage, "output_tokens", 0) or 0
        usage_dict["input_tokens"] += tin
        usage_dict["output_tokens"] += tout
        usage_dict["total_tokens"] = (
            usage_dict["input_tokens"] + usage_dict["output_tokens"]
        )

    span.metadata.setdefault("model_id", model_id)
    span.metadata.setdefault("provider_type", provider_type)

    # Accumulate reasoning (per-call list) when the provider surfaces it.
    reasoning = (
        assistant_msg.metadata.get("reasoning") if assistant_msg.metadata else None
    )
    if reasoning:
        reasoning_list = span.metadata.setdefault("reasoning", [])
        reasoning_list.append(reasoning)

    # Track per-iteration duration so downstream consumers can inspect each call.
    span.metadata.setdefault("llm_durations_ms", []).append(duration_ms)


async def start_tool_span(
    tracer: Tracer | None,
    call: ToolCall,
    registered_tools: list[Tool],
) -> None:
    """Start a span for a tool call, dispatching on skill / sub_agent / tool.

    The SkillTool (``tool_name == SKILL_TOOL_NAME``) is a built-in tool that
    drives the skills subsystem; its calls get a ``SpanType.skill`` named
    after the invoked skill (not the SKILL_TOOL_NAME literal) plus
    ``mode``/``source`` metadata. SubAgentWrapper calls get
    ``SpanType.sub_agent``. Everything else gets ``SpanType.tool``.
    """
    if not tracer:
        return
    from obelix.core.agent.subagent_wrapper import SubAgentWrapper
    from obelix.core.tracer.models import SpanType

    # Deferred import: skill_tool imports BaseAgent (which imports this module
    # via base_agent), so importing at module scope would create a cycle.
    from obelix.plugins.builtin.skill_tool import SKILL_TOOL_NAME

    tool = next(
        (t for t in registered_tools if getattr(t, "tool_name", None) == call.name),
        None,
    )

    # Skill branch: SkillTool carries tool_name == SKILL_TOOL_NAME (the
    # decorator sets this) and exposes its SkillManager via the private
    # _manager attribute populated by make_skill_tool(). Arguments shape
    # produced by the LLM: {"name": "<skill_name>", "args": "<shell-args>"}.
    if tool is not None and getattr(tool, "tool_name", None) == SKILL_TOOL_NAME:
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        skill_name = arguments.get("name") or call.name
        skill_args = arguments.get("args")
        metadata: dict[str, Any] = {}
        skill = _load_skill(tool, skill_name)
        if skill is not None:
            mode = getattr(skill, "context", None)
            source = getattr(skill, "source", None)
            if mode is not None:
                metadata["mode"] = mode
            if source is not None:
                metadata["source"] = source
        await tracer.start_span(
            SpanType.skill,
            skill_name,
            input={"tool_call_id": call.id, "skill_args": skill_args},
            metadata=metadata,
        )
        return

    if isinstance(tool, SubAgentWrapper):
        await tracer.start_span(
            SpanType.sub_agent,
            call.name,
            input={"tool_call_id": call.id, "arguments": call.arguments},
        )
        return

    await tracer.start_span(
        SpanType.tool,
        call.name,
        input={"tool_call_id": call.id, "arguments": call.arguments},
    )


def _load_skill(skill_tool: Any, skill_name: str):
    """Resolve the ``Skill`` object for ``skill_name`` from a SkillTool's manager.

    Returns ``None`` when the tool is not a SkillTool (no ``_manager``), when
    the manager does not expose ``load()``, or when the skill is not found.
    Tracing must degrade gracefully: a miss here just omits the metadata
    (``mode`` / ``source``) rather than crashing the tool dispatch.
    """
    manager = getattr(skill_tool, "_manager", None)
    if manager is None:
        return None
    load = getattr(manager, "load", None)
    if not callable(load):
        return None
    try:
        return load(skill_name)
    except Exception:
        return None


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
