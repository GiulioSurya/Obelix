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
    assert "DW" in out or "deferred_wait" in out


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
