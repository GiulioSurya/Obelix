"""Tests for ConsoleExporter rendering the new taxonomy."""

import io
from contextlib import redirect_stdout
from datetime import UTC, datetime

import pytest

from obelix.core.tracer.exporters import ConsoleExporter
from obelix.core.tracer.models import Span, SpanStatus, SpanType, TraceSession


@pytest.fixture(autouse=True)
def _silence_console_suppression(monkeypatch):
    """Prevent ConsoleExporter's suppress/restore_console from mutating global logger state.

    These helpers operate on the loguru global logger; when they are exercised
    without a prior setup_logging() call they either raise or leak state across
    tests. For unit tests of the exporter we don't need them to do anything.
    """
    monkeypatch.setattr("obelix.core.tracer.exporters.suppress_console", lambda: None)
    monkeypatch.setattr("obelix.core.tracer.exporters.restore_console", lambda: None)


@pytest.mark.asyncio
async def test_badge_mapping_covers_all_span_types():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    for t in SpanType:
        assert t.value in exp._ICONS, f"Missing icon for {t}"


@pytest.mark.asyncio
async def test_renders_a2a_task_span():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="task", trace_id="aaaaaaaaaaaa")

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        start = datetime.now(UTC)
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.a2a_task,
            name="task 7f3a",
            start_time=start,
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 123.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "TK" in out
    assert "task 7f3a" in out


@pytest.mark.asyncio
async def test_renders_skill_span():
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.skill,
            name="code-review",
            start_time=datetime.now(UTC),
            metadata={"mode": "fork"},
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 200.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "SK" in out
    assert "code-review" in out


@pytest.mark.asyncio
async def test_renders_deferred_wait_span_basic():
    """Verifies deferred_wait is formatted at all (full divider rendering in Task 9)."""
    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.deferred_wait,
            name="deferred_wait",
            start_time=datetime.now(UTC),
            metadata={"tool_name": "bash"},
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 7000.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "SUSPEND" in out


@pytest.mark.asyncio
async def test_renders_agent_span_with_llm_usage():
    """Agent span at verbosity=2 shows LLM usage chips from metadata.llm_usage."""
    exp = ConsoleExporter(verbosity=2, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)

    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")

        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.agent,
            name="ReviewerAgent",
            start_time=datetime.now(UTC),
            metadata={
                "llm_usage": {
                    "calls": 2,
                    "input_tokens": 1500,
                    "output_tokens": 250,
                    "total_tokens": 1750,
                }
            },
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 2700.0
        await exp.export_span(span, "svc")

        await exp.end_trace(trace.trace_id, SpanStatus.ok, span.end_time)
    out = buf.getvalue()
    assert "AG" in out
    assert "ReviewerAgent" in out
    assert "2 calls" in out  # calls chip
    # token counts: "1.5k->250 tok" or similar
    assert "1.5k" in out or "1500" in out


@pytest.mark.asyncio
async def test_event_printed_inline_at_verbosity_2():
    from obelix.core.tracer.models import SpanEvent

    exp = ConsoleExporter(verbosity=2, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.agent,
            name="A",
            start_time=datetime.now(UTC),
        )
        await exp.export_span(span, "svc")
        ev = SpanEvent(
            name="hook.fired", attributes={"decision": "REJECT", "reason": "nope"}
        )
        await exp.on_event(span=span, event=ev, service_name="svc")
    out = buf.getvalue()
    assert "hook.fired" in out
    assert "REJECT" in out


@pytest.mark.asyncio
async def test_event_not_printed_at_verbosity_1():
    from obelix.core.tracer.models import SpanEvent

    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.agent,
            name="A",
            start_time=datetime.now(UTC),
        )
        await exp.export_span(span, "svc")
        ev = SpanEvent(name="hook.fired", attributes={"decision": "REJECT"})
        await exp.on_event(span=span, event=ev, service_name="svc")
    out = buf.getvalue()
    assert "hook.fired" not in out


@pytest.mark.asyncio
async def test_chip_counter_shows_in_closed_span_line():
    """When a span closes with events, its rendered line includes chip counters at v1."""
    from obelix.core.tracer.models import SpanEvent

    exp = ConsoleExporter(verbosity=1, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.agent,
            name="A",
            start_time=datetime.now(UTC),
            metadata={
                "llm_usage": {
                    "calls": 1,
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                }
            },
        )
        await exp.export_span(span, "svc")
        span.events.append(
            SpanEvent(name="hook.fired", attributes={"decision": "REJECT"})
        )
        span.events.append(
            SpanEvent(name="memory.pull", attributes={"from_agent": "r"})
        )
        span.end_time = datetime.now(UTC)
        span.duration_ms = 10.0
        await exp.export_span(span, "svc")
    out = buf.getvalue()
    # The agent close-out line must include either numeric counters ("hk:1", "mp:1") or equivalent chip markers
    assert "hk:1" in out or "mp:1" in out or "⚡" in out or "⇩" in out


@pytest.mark.asyncio
async def test_deferred_wait_renders_as_suspend_divider():
    exp = ConsoleExporter(verbosity=2, use_color=False)
    trace = TraceSession(name="t", trace_id="x" * 12)
    buf = io.StringIO()
    with redirect_stdout(buf):
        await exp.start_trace(trace, "svc")
        span = Span(
            trace_id=trace.trace_id,
            span_type=SpanType.deferred_wait,
            name="deferred_wait",
            start_time=datetime.now(UTC),
            metadata={"tool_name": "bash"},
        )
        await exp.export_span(span, "svc")
        span.end_time = datetime.now(UTC)
        span.duration_ms = 7000.0
        await exp.export_span(span, "svc")
    out = buf.getvalue()
    assert "SUSPEND" in out
    assert "7.0s" in out
    assert "bash" in out
    # Divider markers (dashes) should frame the line
    assert "───" in out or "---" in out


def test_deferred_wait_line_encodes_on_windows_cp1252():
    """The deferred_wait divider must not crash on Windows's default cp1252 console."""
    exp = ConsoleExporter(verbosity=2, use_color=False)
    span = Span(
        trace_id="x" * 12,
        span_type=SpanType.deferred_wait,
        name="deferred_wait",
        start_time=datetime.now(UTC),
        metadata={"tool_name": "bash"},
    )
    span.duration_ms = 7000.0
    line = exp._fmt_deferred_wait_line(span)
    # Must round-trip through cp1252 without raising
    line.encode("cp1252")


def test_deferred_wait_line_no_duration_no_unicode_crash():
    """Edge case: if duration_ms is None, empty-duration marker must also encode."""
    exp = ConsoleExporter(verbosity=2, use_color=False)
    span = Span(
        trace_id="x" * 12,
        span_type=SpanType.deferred_wait,
        name="deferred_wait",
        start_time=datetime.now(UTC),
        metadata={"tool_name": "bash"},
    )
    line = exp._fmt_deferred_wait_line(span)
    line.encode("cp1252")
