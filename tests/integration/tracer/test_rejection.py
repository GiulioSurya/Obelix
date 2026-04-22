"""End-to-end trace shape for hook-based rejection scenario.

Exercises the full ``a2a_task`` + ``agent`` lifecycle when a
``BEFORE_LLM_CALL`` hook rejects the task via ``.reject(reason)``. The
scenario is driven by the ``rejection_scenario_with_spy`` fixture which
wires a real ``BaseAgent`` behind an ``ObelixAgentExecutor`` with a mocked
provider whose ``invoke`` must never be called.

Invariants asserted:

* exactly 1 ``a2a_task`` root span with ``status=rejected`` and ``error``
  carrying the rejection reason;
* trace ends with ``status=rejected``;
* at least 1 ``agent`` span carrying a ``hook.fired`` event with
  ``decision=reject`` (StrEnum ``HookDecision.REJECT.value``);
* zero LLM calls on every ``agent`` span (``metadata.llm_usage.calls == 0``
  — rejection fires before ``provider.invoke`` is ever reached);
* an ``a2a.state_change`` event transitioning into the ``rejected`` state
  is emitted on the ``a2a_task`` span.
"""

from __future__ import annotations

import pytest

from obelix.core.tracer.models import SpanType


@pytest.mark.asyncio
async def test_rejection_propagates_to_a2a_task(rejection_scenario_with_spy):
    send_message, spy = rejection_scenario_with_spy
    await send_message("please review")

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    assert len(a2a_tasks) == 1
    assert a2a_tasks[0].status.value == "rejected"
    assert a2a_tasks[0].error  # rejection reason present


@pytest.mark.asyncio
async def test_rejection_marks_trace_status(rejection_scenario_with_spy):
    send_message, spy = rejection_scenario_with_spy
    await send_message("please review")
    assert any(v == "rejected" for v in spy.trace_end_statuses.values())


@pytest.mark.asyncio
async def test_rejection_emits_hook_fired_event_on_agent(rejection_scenario_with_spy):
    send_message, spy = rejection_scenario_with_spy
    await send_message("please review")

    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    assert agent_spans, "expected at least one agent span"
    hook_events = [e for s in agent_spans for e in s.events if e.name == "hook.fired"]
    assert len(hook_events) >= 1
    reject = next(
        (e for e in hook_events if e.attributes.get("decision") == "reject"), None
    )
    assert reject is not None, (
        f"no REJECT event found: {[e.attributes for e in hook_events]}"
    )


@pytest.mark.asyncio
async def test_rejection_zero_llm_calls(rejection_scenario_with_spy):
    """The agent's LLM is never invoked after a BEFORE_LLM_CALL REJECT."""
    send_message, spy = rejection_scenario_with_spy
    await send_message("please review")

    agent_spans = [s for s in spy.spans if s.span_type == SpanType.agent]
    for agent in agent_spans:
        llm_usage = agent.metadata.get("llm_usage", {})
        calls = llm_usage.get("calls", 0)
        assert calls == 0, f"expected 0 LLM calls after rejection, got {calls}"


@pytest.mark.asyncio
async def test_rejection_state_change_on_a2a_task(rejection_scenario_with_spy):
    send_message, spy = rejection_scenario_with_spy
    await send_message("please review")

    a2a_tasks = [s for s in spy.spans if s.span_type == SpanType.a2a_task]
    events = [e for e in a2a_tasks[0].events if e.name == "a2a.state_change"]
    transitions = [e.attributes.get("to") for e in events]
    assert "rejected" in transitions, (
        f"expected 'rejected' state transition, got {transitions}"
    )
