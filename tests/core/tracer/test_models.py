"""Tests for tracer data models."""

from obelix.core.tracer.models import Span, SpanEvent, SpanStatus, SpanType


class TestSpanType:
    def test_new_types_present(self):
        assert SpanType.a2a_task == "a2a_task"
        assert SpanType.skill == "skill"
        assert SpanType.deferred_wait == "deferred_wait"

    def test_kept_types_present(self):
        assert SpanType.agent == "agent"
        assert SpanType.sub_agent == "sub_agent"
        assert SpanType.tool == "tool"
        assert SpanType.human == "human"
        assert SpanType.assistant == "assistant"

    def test_removed_types_absent(self):
        assert not hasattr(SpanType, "llm")
        assert not hasattr(SpanType, "memory")
        assert not hasattr(SpanType, "hook")


class TestSpanEvent:
    def test_constructs_with_defaults(self):
        ev = SpanEvent(name="hook.fired")
        assert ev.name == "hook.fired"
        assert ev.attributes == {}
        assert ev.timestamp is not None

    def test_carries_attributes(self):
        ev = SpanEvent(
            name="memory.pull", attributes={"from_agent": "reviewer", "bytes": 1240}
        )
        assert ev.attributes["from_agent"] == "reviewer"
        assert ev.attributes["bytes"] == 1240

    def test_is_serializable(self):
        ev = SpanEvent(
            name="a2a.state_change", attributes={"to": "rejected", "reason": "nope"}
        )
        dumped = ev.model_dump(mode="json")
        assert dumped["name"] == "a2a.state_change"
        assert dumped["attributes"]["to"] == "rejected"
        assert "timestamp" in dumped


class TestSpanEvents:
    def test_span_has_empty_events_by_default(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        assert sp.events == []

    def test_events_can_be_appended(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        sp.events.append(
            SpanEvent(name="hook.fired", attributes={"decision": "REJECT"})
        )
        assert len(sp.events) == 1
        assert sp.events[0].name == "hook.fired"

    def test_events_included_in_dump(self):
        sp = Span(trace_id="t1", span_type=SpanType.agent, name="a")
        sp.events.append(SpanEvent(name="memory.pull", attributes={"from_agent": "x"}))
        dumped = sp.model_dump(mode="json")
        assert len(dumped["events"]) == 1
        assert dumped["events"][0]["name"] == "memory.pull"


class TestSpanStatus:
    def test_existing_values(self):
        assert SpanStatus.ok == "ok"
        assert SpanStatus.error == "error"
        assert SpanStatus.timeout == "timeout"

    def test_a2a_values(self):
        assert SpanStatus.rejected == "rejected"
        assert SpanStatus.canceled == "canceled"
