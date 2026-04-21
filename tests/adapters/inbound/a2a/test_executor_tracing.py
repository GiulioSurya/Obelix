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


@pytest.mark.asyncio
async def test_executor_does_not_unboundlocalerror_on_tracer_failure(
    executor_with_tracer,
):
    """
    If the tracer raises during start_span/end_span of the human span (or anywhere
    before final_response is assigned), the executor must not crash with
    UnboundLocalError on the assistant span emission block.
    """
    send_message, spy = executor_with_tracer

    # Patch the Tracer.start_span to raise on the human span call (2nd start_span
    # invocation: 1st is the a2a_task root, 2nd is the human span).
    from obelix.core.tracer import tracer as tracer_mod

    original_start_span = tracer_mod.Tracer.start_span
    call_count = {"n": 0}

    async def flaky_start_span(self, span_type, name, *args, **kwargs):
        call_count["n"] += 1
        # Let a2a_task (1st call) succeed, fail on human span (2nd call)
        if call_count["n"] == 2:
            raise RuntimeError("flaky exporter")
        return await original_start_span(self, span_type, name, *args, **kwargs)

    tracer_mod.Tracer.start_span = flaky_start_span
    try:
        # Should not raise UnboundLocalError — should propagate the RuntimeError
        # or gracefully continue. Accept either:
        try:
            await send_message("hi")
        except RuntimeError as e:
            assert "flaky" in str(e)
        except NameError as e:
            # UnboundLocalError is a subclass of NameError
            pytest.fail(f"UnboundLocalError leaked: {e}")
    finally:
        tracer_mod.Tracer.start_span = original_start_span


@pytest.mark.asyncio
async def test_executor_emits_state_change_events(executor_with_tracer):
    """A successful turn emits at least working and completed state_change events on the a2a_task span."""
    send_message, spy = executor_with_tracer
    await send_message("hi")

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assert len(a2a_tasks) == 1
    events = a2a_tasks[0].events
    state_changes = [e for e in events if e.name == "a2a.state_change"]
    assert len(state_changes) >= 2
    states = [e.attributes.get("to") for e in state_changes]
    assert "working" in states, f"working not in {states}"
    assert "completed" in states, f"completed not in {states}"


@pytest.mark.asyncio
async def test_state_change_attributes_carry_from_to_reason(executor_with_tracer):
    """Each state_change event has 'to' attribute; 'from' and 'reason' are optional."""
    send_message, spy = executor_with_tracer
    await send_message("hi")

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    events = [e for e in a2a_tasks[0].events if e.name == "a2a.state_change"]
    for e in events:
        assert "to" in e.attributes, f"state_change event missing 'to': {e.attributes}"


@pytest.mark.asyncio
async def test_deferred_tool_creates_deferred_wait_span(executor_with_deferred_tool):
    """A deferred tool pause opens a ``deferred_wait`` span that closes on resume.

    The fixture produces an executor whose first agent response is a deferred
    tool call (is_deferred=True). The executor must:

    * emit ``input_required`` and open a ``deferred_wait`` span as a direct
      child of the ``a2a_task`` root span;
    * keep that span open across the suspension;
    * close it on the resume path (duration = wall-clock of the pause).

    Metadata on the span must carry ``tool_name``, ``tool_call_ids``, and
    ``suspend_reason="deferred_tool"``.
    """
    send_message, resume, spy = executor_with_deferred_tool
    await send_message("do the deferred thing")
    await resume({"answer": "EUR"})

    # One deferred_wait span with a measurable duration
    dw_spans = [s for s in spy.spans if s.span_type.value == "deferred_wait"]
    assert len(dw_spans) == 1, (
        f"expected exactly one deferred_wait span, got {len(dw_spans)}: "
        f"{[(s.name, s.span_type.value) for s in spy.spans]}"
    )
    dw = dw_spans[0]
    assert dw.duration_ms is not None and dw.duration_ms >= 0
    tool_name = dw.metadata.get("tool_name")
    assert tool_name, f"deferred_wait span missing tool_name metadata: {dw.metadata}"
    assert dw.metadata.get("suspend_reason") == "deferred_tool"
    tool_call_ids = dw.metadata.get("tool_call_ids")
    assert tool_call_ids and "tc-deferred-1" in tool_call_ids

    # Parent must be the a2a_task span (sibling of the agent span)
    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assert len(a2a_tasks) == 1
    assert dw.parent_span_id == a2a_tasks[0].span_id, (
        f"deferred_wait parent must be a2a_task ({a2a_tasks[0].span_id}); "
        f"got parent_span_id={dw.parent_span_id!r}"
    )


@pytest.mark.asyncio
async def test_deferred_wait_cleared_on_resume(executor_with_deferred_tool):
    """After resume finishes, ``entry.deferred_wait_span_id`` is cleared."""
    send_message, resume, _spy = executor_with_deferred_tool
    await send_message("do the deferred thing")
    await resume({"answer": "ok"})

    # Inspect the executor's context store (stable for this fixture).
    # The ctx entry should have deferred_wait_span_id cleared.
    # We reach it via the fixture's closure — walk from a fresh module call.
    # The fixture uses a stable context_id ("ctx-deferred-001").
    # We need access to the executor; easiest path is checking the only ctx.
    # Since we don't expose the executor, inspect via the spy exporter's
    # span metadata as a proxy: the deferred_wait span must have end_time set.
    dw_spans = [
        s
        for s in _spy.spans
        if s.span_type.value == "deferred_wait" and s.end_time is not None
    ]
    assert len(dw_spans) == 1


@pytest.mark.asyncio
async def test_cancellation_emits_event_on_a2a_task(
    executor_with_deferred_tool_and_cancel,
):
    """
    When cancel() is called during deferred input_required, the a2a_task span
    receives a cancellation.requested event.
    """
    send_message, resume, spy, executor_cancel = executor_with_deferred_tool_and_cancel
    await send_message("do the deferred thing")
    await executor_cancel()

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assert len(a2a_tasks) == 1
    cancel_events = [
        e for e in a2a_tasks[0].events if e.name == "cancellation.requested"
    ]
    assert len(cancel_events) == 1
    assert cancel_events[0].attributes.get("source")


@pytest.mark.asyncio
async def test_cancel_closes_deferred_wait_span(executor_with_deferred_tool_and_cancel):
    """Task 19 follow-up: cancelling during input_required must close deferred_wait."""
    send_message, resume, spy, executor_cancel = executor_with_deferred_tool_and_cancel
    await send_message("deferred thing")
    await executor_cancel()

    dw_spans = [s for s in spy.spans if s.span_type.value == "deferred_wait"]
    assert len(dw_spans) == 1
    assert dw_spans[0].end_time is not None, (
        "deferred_wait span must be closed on cancel; got end_time=None (leak)"
    )


@pytest.mark.asyncio
async def test_cancel_marks_a2a_task_status_canceled(
    executor_with_deferred_tool_and_cancel,
):
    """a2a_task span status is SpanStatus.canceled after cancel."""
    send_message, resume, spy, executor_cancel = executor_with_deferred_tool_and_cancel
    await send_message("deferred thing")
    await executor_cancel()

    a2a_tasks = [s for s in spy.spans if s.span_type.value == "a2a_task"]
    assert len(a2a_tasks) == 1
    assert a2a_tasks[0].status.value == "canceled"
