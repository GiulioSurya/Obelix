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
