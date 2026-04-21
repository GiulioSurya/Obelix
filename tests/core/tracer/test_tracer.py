"""Tests for Tracer API (add_event)."""

import pytest

from obelix.core.tracer.exporters import NoOpExporter
from obelix.core.tracer.models import SpanType
from obelix.core.tracer.tracer import Tracer


class SpyExporter(NoOpExporter):
    def __init__(self):
        self.events = []
        self.span_exports = []

    async def export_span(self, span, service_name):
        self.span_exports.append(
            (span.span_id, span.end_time is not None, list(span.events))
        )

    async def on_event(self, span, event, service_name):
        self.events.append((span.span_id, event.name, dict(event.attributes)))


@pytest.mark.asyncio
async def test_add_event_attaches_to_current_span():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    span = await tracer.start_span(SpanType.agent, "agent.Test")
    await tracer.add_event("hook.fired", {"decision": "REJECT"})
    await tracer.end_span()
    await tracer.end_trace()

    assert len(span.events) == 1
    assert span.events[0].name == "hook.fired"
    assert span.events[0].attributes == {"decision": "REJECT"}


@pytest.mark.asyncio
async def test_add_event_calls_exporter_hook():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    await tracer.start_span(SpanType.agent, "a")
    await tracer.add_event("memory.pull", {"from_agent": "x"})
    await tracer.end_span()
    await tracer.end_trace()

    assert len(exp.events) == 1
    assert exp.events[0][1] == "memory.pull"
    assert exp.events[0][2] == {"from_agent": "x"}


@pytest.mark.asyncio
async def test_add_event_without_span_is_noop():
    """Adding an event when no span is active must not raise."""
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    # no trace, no span
    await tracer.add_event("orphan", {})
    assert exp.events == []


@pytest.mark.asyncio
async def test_span_export_includes_events():
    exp = SpyExporter()
    tracer = Tracer(exporter=exp)
    await tracer.start_trace("t")
    await tracer.start_span(SpanType.agent, "a")
    await tracer.add_event("e1", {"k": "v"})
    await tracer.end_span()
    await tracer.end_trace()

    # export_span called at start (no events) and at end (with event)
    end_export = exp.span_exports[-1]
    assert end_export[1] is True  # has end_time
    assert len(end_export[2]) == 1
    assert end_export[2][0].name == "e1"
