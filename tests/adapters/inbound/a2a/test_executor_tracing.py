"""Integration tests for the A2A executor's tracer instrumentation.

Verifies that ``ObelixAgentExecutor`` opens a ``SpanType.a2a_task`` span as
the root of the trace, with the agent span becoming a child of it.
"""

from __future__ import annotations

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_executor_opens_a2a_task_root_span(executor_with_tracer):
    """A2A executor opens an a2a_task span as the trace root.

    The fixture returns ``(send_message, spy_exporter)``. Calling
    ``send_message`` drives ``ObelixAgentExecutor.execute`` through a minimal
    ``RequestContext`` + ``EventQueue`` stub. We then assert:

    * exactly one ``a2a_task`` span was produced;
    * it has no parent (i.e. it is the trace root);
    * at least one ``agent`` span exists with the ``a2a_task`` span as its
      direct parent.
    """
    send_message, spy = executor_with_tracer
    await send_message("hello")

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert len(a2a_tasks) == 1, (
        f"expected exactly one a2a_task span, got {len(a2a_tasks)}"
    )
    a2a_root = a2a_tasks[0]

    # The a2a_task span has no parent (it's the trace root)
    assert a2a_root.parent_span_id is None, (
        "a2a_task span must be the trace root (parent_span_id is None); "
        f"got parent_span_id={a2a_root.parent_span_id!r}"
    )

    # At least one agent span exists with a2a_task as its direct parent
    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    assert agent_spans, "no agent span was produced"
    assert any(a.parent_span_id == a2a_root.span_id for a in agent_spans), (
        "no agent span is a direct child of the a2a_task root span"
    )


@pytest.mark.asyncio
async def test_executor_emits_human_span(executor_with_tracer):
    """Human span emitted once per turn as a direct child of a2a_task."""
    send_message, spy = executor_with_tracer
    await send_message("review my staged changes")

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assert len(a2a_tasks) == 1
    a2a_task = a2a_tasks[0]

    human_spans = [s for s in spy.spans if s.span_type.value == "human"]
    assert len(human_spans) == 1
    human = human_spans[0]
    assert human.parent_span_id == a2a_task.span_id
    # The human input must carry the query text (exact format TBD — accept
    # either string or dict with 'text'/'content' field)
    input_value = human.input
    if isinstance(input_value, dict):
        assert (
            input_value.get("text") == "review my staged changes"
            or input_value.get("content") == "review my staged changes"
        )
    else:
        assert input_value == "review my staged changes"


@pytest.mark.asyncio
async def test_executor_emits_assistant_span_on_final(executor_with_tracer):
    """Assistant span emitted once per turn at the final response."""
    send_message, spy = executor_with_tracer
    await send_message("hi")

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assistant_spans = [s for s in spy.spans if s.span_type.value == "assistant"]
    assert len(assistant_spans) == 1
    assistant = assistant_spans[0]
    assert assistant.parent_span_id == a2a_tasks[0].span_id
    # Output should carry the response content (the fixture's mock provider
    # returns "done")
    out = assistant.output
    if isinstance(out, dict):
        assert "done" in str(out.get("content", ""))
    else:
        assert "done" in str(out)


@pytest.mark.asyncio
async def test_nested_agent_does_not_emit_duplicate_human_assistant(
    executor_with_tracer,
):
    """BaseAgent (nested under a2a_task) must NOT emit its own human/assistant spans."""
    send_message, spy = executor_with_tracer
    await send_message("hi")

    # Only the executor should emit these — BaseAgent would have created duplicates
    # if Task 13's guard failed.
    human_spans = [s for s in spy.spans if s.span_type.value == "human"]
    assistant_spans = [s for s in spy.spans if s.span_type.value == "assistant"]
    assert len(human_spans) == 1, f"expected 1 human span, got {len(human_spans)}"
    assert len(assistant_spans) == 1, (
        f"expected 1 assistant span, got {len(assistant_spans)}"
    )
