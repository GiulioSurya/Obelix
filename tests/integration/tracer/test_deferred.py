"""End-to-end trace shape for deferred tool scenario.

Exercises the full ``a2a_task`` + ``deferred_wait`` + resume lifecycle via the
``deferred_scenario_with_spy`` fixture (mirror of ``executor_with_deferred_tool``
in ``tests/adapters/inbound/a2a/conftest.py``).

Invariants asserted:

* exactly 1 ``a2a_task`` root span — stays open across suspend/resume and
  closes with ``ok`` status on completion;
* exactly 1 ``deferred_wait`` span — duration ≥ the simulated pause and
  parent is the ``a2a_task`` root (sibling of the agent span);
* multiple ``a2a.state_change`` events on ``a2a_task`` covering at least
  ``working`` (initial), ``input_required``, ``working`` (resume), and
  ``completed``;
* ``deferred_wait`` metadata carries ``suspend_reason="deferred_tool"`` and
  ``tool_name`` — the fixture's mock agent doesn't run the real BaseAgent
  tool loop so there is no ``SpanType.tool`` span, but the deferred_wait
  span itself is the canonical "deferred" span per Task 19's spec;
* final trace status is ``ok``.
"""

from __future__ import annotations

import asyncio

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_deferred_wait_span_between_suspend_and_resume(
    deferred_scenario_with_spy,
):
    """``deferred_wait`` duration covers the pause between suspend and resume."""
    send_message, resume, spy = deferred_scenario_with_spy
    await send_message("run the deferred thing")
    # Simulate a pause by sleeping before resume so deferred_wait has a
    # measurable wall-clock duration.
    await asyncio.sleep(0.3)
    await resume({"stdout": "ok"})

    dw_spans = [s for s in spy.spans if s.span_type == SpanType.deferred_wait]
    assert len(dw_spans) == 1, (
        f"expected exactly one deferred_wait span, got {len(dw_spans)}: "
        f"{[(s.name, s.span_type.value) for s in spy.spans]}"
    )
    dw = dw_spans[0]
    assert dw.duration_ms is not None
    assert dw.duration_ms >= 250, (
        f"deferred_wait duration {dw.duration_ms}ms should be >= 250ms "
        f"(slept ~300ms between suspend and resume)"
    )

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert len(a2a_tasks) == 1, (
        f"expected exactly 1 a2a_task root span, got {len(a2a_tasks)}"
    )
    assert a2a_tasks[0].status.value == "ok", (
        f"expected a2a_task status=ok on completion, got {a2a_tasks[0].status.value}"
    )


@pytest.mark.asyncio
async def test_deferred_state_transitions_recorded(deferred_scenario_with_spy):
    """``a2a_task`` records the full state_change sequence across suspend/resume.

    Expected transitions (order and multiplicity):
    working (initial) → input_required → working (resume) → completed
    """
    send_message, resume, spy = deferred_scenario_with_spy
    await send_message("run")
    await resume({"ok": True})

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert a2a_tasks, "no a2a_task span emitted"

    events = [e for e in a2a_tasks[0].events if e.name == "a2a.state_change"]
    transitions = [e.attributes.get("to") for e in events]

    assert "working" in transitions, f"missing 'working' transition: {transitions}"
    assert "input_required" in transitions, (
        f"missing 'input_required' transition: {transitions}"
    )
    assert "completed" in transitions, f"missing 'completed' transition: {transitions}"

    # At least two "working" transitions (initial + resume)
    assert transitions.count("working") >= 2, (
        f"expected >= 2 'working' transitions (initial + resume), "
        f"got {transitions.count('working')} in {transitions}"
    )


@pytest.mark.asyncio
async def test_deferred_wait_parent_is_a2a_task(deferred_scenario_with_spy):
    """``deferred_wait`` must be a direct child of the ``a2a_task`` root span.

    Per Task 19's spec, the deferred_wait span is a sibling of the agent span
    under a2a_task — covering the suspension window at the task-level.
    """
    send_message, resume, spy = deferred_scenario_with_spy
    await send_message("run")
    await resume({"ok": True})

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    dw_spans = [s for s in spy.spans if s.span_type == SpanType.deferred_wait]
    assert len(a2a_tasks) == 1 and len(dw_spans) == 1
    assert dw_spans[0].parent_span_id == a2a_tasks[0].span_id, (
        f"deferred_wait parent must be a2a_task ({a2a_tasks[0].span_id}); "
        f"got parent_span_id={dw_spans[0].parent_span_id!r}"
    )


@pytest.mark.asyncio
async def test_deferred_wait_carries_deferred_metadata(deferred_scenario_with_spy):
    """``deferred_wait`` metadata identifies the span as the deferred-tool pause.

    Per Task 19, the span metadata carries:
    - ``suspend_reason="deferred_tool"`` (equivalent to a ``deferred=true``
      flag — marks the span as the deferred-tool suspension window);
    - ``tool_name`` — the name of the deferred tool that triggered the pause;
    - ``tool_call_ids`` — the list of deferred tool-call ids in flight.
    """
    send_message, resume, spy = deferred_scenario_with_spy
    await send_message("run")
    await resume({"ok": True})

    dw_spans = [s for s in spy.spans if s.span_type == SpanType.deferred_wait]
    assert len(dw_spans) == 1
    dw = dw_spans[0]

    assert dw.metadata.get("suspend_reason") == "deferred_tool", (
        f"deferred_wait must carry suspend_reason=deferred_tool metadata; "
        f"got metadata={dw.metadata}"
    )
    tool_name = dw.metadata.get("tool_name")
    assert tool_name == "ask_user", (
        f"deferred_wait tool_name should be 'ask_user', got {tool_name!r}"
    )
    tool_call_ids = dw.metadata.get("tool_call_ids")
    assert tool_call_ids and "tc-deferred-int-1" in tool_call_ids, (
        f"deferred_wait tool_call_ids should include the in-flight call id; "
        f"got {tool_call_ids!r}"
    )
