"""Tests for tracer data models."""

from obelix.core.tracer.models import SpanType


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
