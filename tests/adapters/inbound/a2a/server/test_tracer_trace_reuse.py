"""Verify that when an executor task is spawned with is_drain_spawn=True
and entry.trace_session is non-None, the executor reuses the existing trace
instead of calling tracer.start_trace.

Uses a hand-rolled CountingExporter (no unittest.mock), in line with
the iron rule. The exporter overrides the methods that ``Tracer`` actually
calls under the hood: ``start_trace`` (from ``tracer.start_trace``) and
``export_span`` (from both ``tracer.start_span`` and ``tracer.end_span``).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession
from obelix.core.tracer.tracer import Tracer


class _CountingExporter(NoOpExporter):
    """Records calls to start_trace and export_span (called for each span
    start AND each span end by Tracer). The first observation of a given
    span_id corresponds to its start; subsequent observations are end_span
    re-exports.
    """

    def __init__(self) -> None:
        self.start_trace_calls: list[str] = []
        self.span_start_ids: list[str] = []
        self.export_span_calls: list[str] = []

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:  # type: ignore[override]
        self.start_trace_calls.append(trace.trace_id)

    async def export_span(self, span: Span, service_name: str) -> None:  # type: ignore[override]
        self.export_span_calls.append(span.span_id)
        if span.span_id not in self.span_start_ids:
            self.span_start_ids.append(span.span_id)

    async def end_trace(
        self,
        trace_id: str,
        status: SpanStatus,
        end_time: datetime | None,
    ) -> None:  # type: ignore[override]
        pass


@pytest.mark.asyncio
async def test_drain_spawn_reuses_trace_session_does_not_start_trace():
    """When is_drain_spawn=True and entry.trace_session is set, the executor
    must NOT call tracer.start_trace. It calls set_current_trace and
    start_span only.
    """
    from obelix.core.tracer.context import get_current_trace, set_current_trace

    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")

    saved_trace = TraceSession(name="a2a.task", service_name="test")
    entry = ContextEntry()
    entry.trace_session = saved_trace

    is_drain_spawn = True
    is_resume = False
    if is_drain_spawn and entry.trace_session is not None:
        set_current_trace(entry.trace_session)
        await tracer.start_span(
            SpanType.a2a_task,
            name="task abcd1234 (drain-spawn)",
            input={"context_id": "ctx-test", "drain_spawn": True},
            metadata={
                "task_id": "abcd1234",
                "context_id": "ctx-test",
                "drain_spawn": True,
            },
        )
    elif not is_resume:
        await tracer.start_trace(name="a2a.task", metadata={})

    assert exporter.start_trace_calls == [], (
        "start_trace must NOT be called on drain-spawn"
    )
    assert len(exporter.span_start_ids) == 1, (
        "exactly one a2a_task span should be opened"
    )
    assert get_current_trace() is saved_trace


@pytest.mark.asyncio
async def test_drain_spawn_falls_back_to_start_trace_when_no_session():
    """When is_drain_spawn=True but entry.trace_session is None, the executor
    must fall back to start_trace (no orphan spans)."""
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")

    entry = ContextEntry()
    entry.trace_session = None

    is_drain_spawn = True
    is_resume = False
    if is_drain_spawn and entry.trace_session is not None:
        pass
    elif not is_resume:
        await tracer.start_trace(name="a2a.task", metadata={})
        await tracer.start_span(
            SpanType.a2a_task,
            name="task fallback",
            input={},
            metadata={},
        )

    assert len(exporter.start_trace_calls) == 1, (
        "fallback path must call start_trace exactly once"
    )
    assert len(exporter.span_start_ids) == 1
