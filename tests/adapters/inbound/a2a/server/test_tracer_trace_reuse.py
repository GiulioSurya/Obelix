"""Verify tracer span/trace lifecycle for the three branches of
``ObelixAgentExecutor._open_a2a_task_span`` and the matching close
behavior in ``_run_agent``'s finally block.

Tests exercise the REAL helper / real finally code path, NOT an inline
re-implementation. The previous version rebuilt the branch logic in the
test body and asserted on the rebuilt code — a self-confirming pattern
that would let real refactors silently break the executor without test
failure. Here we instantiate the executor (via a thin subclass with an
empty ``__init__`` to skip the agent_factory dependency) and call the
real ``_open_a2a_task_span`` / ``_run_agent`` so the assertions cover
the production code.

Iron rule: no ``unittest.mock`` / ``pytest-mock`` / ``monkeypatch`` on
SDK code. ``_MinimalExec`` is a hand-written subclass that bypasses
``ObelixAgentExecutor.__init__`` (which requires a full agent_factory +
context store); not a mock — a Fake-via-subclass that isolates the
helper for direct invocation. ``_CountingExporter`` is a hand-written
``NoOpExporter`` subclass that records calls.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.core.tracer.context import get_current_trace
from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession
from obelix.core.tracer.tracer import Tracer


class _CountingExporter(NoOpExporter):
    """Records calls to start_trace / end_trace / export_span.

    ``export_span`` is invoked once per ``start_span`` AND once per
    ``end_span`` by ``Tracer``. The first observation of a given
    ``span_id`` corresponds to its start; subsequent observations to
    end_span re-exports. ``span_end_ids`` records ids whose
    ``end_time`` is non-None at export time, so we can assert close.
    """

    def __init__(self) -> None:
        self.start_trace_calls: list[str] = []
        self.end_trace_calls: list[str] = []
        self.span_start_ids: list[str] = []
        self.span_end_ids: list[str] = []
        self.export_span_calls: list[str] = []

    async def start_trace(self, trace: TraceSession, service_name: str) -> None:  # type: ignore[override]
        self.start_trace_calls.append(trace.trace_id)

    async def export_span(self, span: Span, service_name: str) -> None:  # type: ignore[override]
        self.export_span_calls.append(span.span_id)
        if span.span_id not in self.span_start_ids:
            self.span_start_ids.append(span.span_id)
        if span.end_time is not None and span.span_id not in self.span_end_ids:
            self.span_end_ids.append(span.span_id)

    async def end_trace(
        self,
        trace_id: str,
        status: SpanStatus,
        end_time: datetime | None,
    ) -> None:  # type: ignore[override]
        self.end_trace_calls.append(trace_id)


class _MinimalExec(ObelixAgentExecutor):
    """Fake-via-subclass: bypass ``ObelixAgentExecutor.__init__`` so we can
    instantiate the executor without an agent_factory / context store.
    Tests set ``_tracer`` directly. Not a mock; just an isolation harness.
    """

    def __init__(self) -> None:  # noqa: D401 - intentional bypass
        # Skip super().__init__: tests only exercise tracer-related methods,
        # which depend solely on ``self._tracer``.
        pass


# ---------- _open_a2a_task_span: branch coverage ----------


@pytest.mark.asyncio
async def test_drain_spawn_reuses_trace_session_does_not_start_trace():
    """When ``is_drain_spawn=True`` and ``entry.trace_session`` is non-None,
    the helper must reuse the saved trace and NOT call
    ``tracer.start_trace``. Returned span is non-None,
    ``trace_opened_here`` is False (entry already owns the trace).
    """
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")

    saved_trace = TraceSession(name="a2a.task", service_name="test")
    entry = ContextEntry()
    entry.trace_session = saved_trace

    executor = _MinimalExec()
    executor._tracer = tracer

    span, opened_here = await executor._open_a2a_task_span(
        task_id="abcd1234",
        context_id="ctx-test",
        entry=entry,
        is_resume=False,
        is_drain_spawn=True,
    )

    assert exporter.start_trace_calls == [], (
        "start_trace must NOT be called on drain-spawn with saved trace"
    )
    assert span is not None, "a2a_task span must be opened on drain-spawn"
    assert span.span_type == SpanType.a2a_task
    assert opened_here is False, "drain-spawn does not own the trace"
    assert get_current_trace() is saved_trace, (
        "current trace must be set to the saved trace_session"
    )
    # entry.trace_session is preserved (entry owns the trace).
    assert entry.trace_session is saved_trace
    # Exactly one a2a_task span started; no extra spans leaked.
    assert len(exporter.span_start_ids) == 1


@pytest.mark.asyncio
async def test_drain_spawn_falls_back_to_start_trace_when_no_session():
    """When ``is_drain_spawn=True`` but ``entry.trace_session`` is None,
    the helper falls back to opening a fresh trace AND span; returns
    ``opened_here=True`` so the finally closes both. Prevents orphan
    spans on first-run drain-spawn (e.g. server restart races).
    """
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")
    entry = ContextEntry()
    entry.trace_session = None

    executor = _MinimalExec()
    executor._tracer = tracer

    span, opened_here = await executor._open_a2a_task_span(
        task_id="t1",
        context_id="ctx-1",
        entry=entry,
        is_resume=False,
        is_drain_spawn=True,
    )

    assert len(exporter.start_trace_calls) == 1, (
        "fallback path must call start_trace exactly once"
    )
    assert span is not None
    assert opened_here is True, "fallback path owns the trace"
    assert entry.trace_session is not None, (
        "entry.trace_session must be saved on the new trace"
    )
    assert len(exporter.span_start_ids) == 1


@pytest.mark.asyncio
async def test_user_triggered_task_starts_new_trace():
    """``is_drain_spawn=False, is_resume=False`` (= user-triggered first
    turn) opens a fresh trace + a2a_task span; saves the new trace on
    the entry for later drain-spawns / cancellations. Verifies the
    pre-existing path is unchanged by the refactor.
    """
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")
    entry = ContextEntry()
    entry.trace_session = None

    executor = _MinimalExec()
    executor._tracer = tracer

    span, opened_here = await executor._open_a2a_task_span(
        task_id="t1",
        context_id="ctx-1",
        entry=entry,
        is_resume=False,
        is_drain_spawn=False,
    )

    assert len(exporter.start_trace_calls) == 1
    assert span is not None
    assert opened_here is True
    assert entry.trace_session is not None
    # Saved trace_session is the live trace.
    assert get_current_trace() is entry.trace_session


@pytest.mark.asyncio
async def test_resume_does_not_open_trace_or_span():
    """``is_resume=True``: helper is a no-op (trace + span are restored
    by ``_run_agent_impl``). Returns ``(None, False)`` so the finally
    closes the trace (close_trace_here = trace_opened_here OR is_resume).
    """
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")
    entry = ContextEntry()

    executor = _MinimalExec()
    executor._tracer = tracer

    span, opened_here = await executor._open_a2a_task_span(
        task_id="t1",
        context_id="ctx-1",
        entry=entry,
        is_resume=True,
        is_drain_spawn=False,
    )

    assert exporter.start_trace_calls == []
    assert exporter.span_start_ids == []
    assert span is None
    assert opened_here is False


@pytest.mark.asyncio
async def test_no_tracer_returns_noop_tuple():
    """When ``self._tracer is None`` the helper must return
    ``(None, False)`` and not raise."""
    executor = _MinimalExec()
    executor._tracer = None

    entry = ContextEntry()
    span, opened_here = await executor._open_a2a_task_span(
        task_id="t1",
        context_id="ctx-1",
        entry=entry,
        is_resume=False,
        is_drain_spawn=True,
    )
    assert span is None
    assert opened_here is False


# ---------- _run_agent finally: span lifecycle ----------


class _FakeEventQueue:
    """Hand-written EventQueue that records enqueue calls. Replaces the
    a2a-sdk EventQueue without mocking the SDK class."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


class _FakeAgent:
    """Hand-written stand-in for BaseAgent. ``_run_agent_impl`` reads a
    handful of attributes (system_message, conversation_history,
    cancel(), etc.); we patch ``_run_agent_impl`` instead of populating
    every contract surface — see ``_NoOpRunAgentImpl`` below.
    """

    def __init__(self) -> None:
        self.canceled = False

    def cancel(self) -> None:
        self.canceled = True


class _NoOpRunAgentImpl(_MinimalExec):
    """Subclass that overrides ``_run_agent_impl`` to a no-op so we can
    exercise ``_run_agent``'s finally without spinning up a real agent.
    Returns ``False`` (not deferred-suspended) so the finally runs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.run_impl_calls: list[dict] = []

    async def _run_agent_impl(self, **kwargs) -> bool:  # type: ignore[override]
        self.run_impl_calls.append(kwargs)
        return False  # not suspended → finally must close span


@pytest.mark.asyncio
async def test_run_agent_drain_spawn_closes_span_but_not_trace():
    """BUG REGRESSION: drain-spawn used to leave its a2a_task span
    dangling because the old finally guard
    ``trace_opened_here or is_resume`` was False on this branch.
    The fix closes the span whenever ``a2a_task_span is not None``
    and only ends the trace when ``trace_opened_here`` is True.

    Asserts:
      - ``end_trace`` is NOT called (trace stays open for the context).
      - The drain-spawn a2a_task span IS ended (export_span re-fires
        with ``end_time != None``; ``span_end_ids`` has 1 entry).
      - ``entry.trace_session`` is preserved (still owned by the
        original first turn).
    """
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")

    # Pre-existing trace owned by a prior first-turn task on this context.
    saved_trace = TraceSession(name="a2a.task", service_name="test")
    entry = ContextEntry()
    entry.trace_session = saved_trace

    executor = _NoOpRunAgentImpl()
    executor._tracer = tracer

    queue = _FakeEventQueue()
    await executor._run_agent(
        task_id="dd112233",
        context_id="ctx-drain",
        user_text="hi",
        attachments=[],
        entry=entry,
        event_queue=queue,
        is_resume=False,
        is_drain_spawn=True,
    )

    # _run_agent_impl was called exactly once (no-op stub).
    assert len(executor.run_impl_calls) == 1
    # Drain-spawn must NOT call start_trace or end_trace.
    assert exporter.start_trace_calls == []
    assert exporter.end_trace_calls == []
    # Span MUST have been opened AND closed.
    assert len(exporter.span_start_ids) == 1, (
        "exactly one a2a_task span opened on drain-spawn"
    )
    assert len(exporter.span_end_ids) == 1, (
        "drain-spawn span MUST be closed in finally (regression: was leaked)"
    )
    assert exporter.span_end_ids == exporter.span_start_ids
    # Entry's trace_session is preserved (drain-spawn does not own it).
    assert entry.trace_session is saved_trace


@pytest.mark.asyncio
async def test_run_agent_user_triggered_closes_span_and_ends_trace():
    """First-turn (user-triggered) path: finally closes the span AND
    ends the trace. Verifies the pre-existing behavior survived the
    refactor."""
    exporter = _CountingExporter()
    tracer = Tracer(exporter, service_name="test")

    entry = ContextEntry()
    entry.trace_session = None

    executor = _NoOpRunAgentImpl()
    executor._tracer = tracer

    queue = _FakeEventQueue()
    await executor._run_agent(
        task_id="ee998877",
        context_id="ctx-user",
        user_text="hi",
        attachments=[],
        entry=entry,
        event_queue=queue,
        is_resume=False,
        is_drain_spawn=False,
    )

    assert len(exporter.start_trace_calls) == 1
    assert len(exporter.end_trace_calls) == 1
    assert len(exporter.span_start_ids) == 1
    assert len(exporter.span_end_ids) == 1
    # entry.trace_session cleared on close (next turn opens a new trace).
    assert entry.trace_session is None
