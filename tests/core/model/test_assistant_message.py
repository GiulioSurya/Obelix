"""Tests for obelix.core.model.assistant_message — AssistantMessage and AssistantResponse."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from obelix.core.model.assistant_message import AssistantMessage, AssistantResponse
from obelix.core.model.roles import MessageRole
from obelix.core.model.tool_message import ToolCall, ToolResult, ToolStatus
from obelix.core.model.usage import Usage

# ---------------------------------------------------------------------------
# AssistantMessage
# ---------------------------------------------------------------------------


class TestAssistantMessageConstruction:
    """Construction and default values."""

    def test_default_role(self):
        msg = AssistantMessage(content="Hi")
        assert msg.role == MessageRole.ASSISTANT

    def test_content_stored(self):
        msg = AssistantMessage(content="The answer is 42.")
        assert msg.content == "The answer is 42."

    def test_default_content_is_empty(self):
        msg = AssistantMessage()
        assert msg.content == ""

    def test_timestamp_auto_generated(self):
        before = datetime.now()
        msg = AssistantMessage(content="x")
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_default_tool_calls_is_empty_list(self):
        msg = AssistantMessage()
        assert msg.tool_calls == []

    def test_default_usage_is_none(self):
        msg = AssistantMessage()
        assert msg.usage is None

    def test_default_metadata_is_empty_dict(self):
        msg = AssistantMessage()
        assert msg.metadata == {}

    def test_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1})
        msg = AssistantMessage(content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "calc"

    def test_with_usage(self):
        usage = Usage(input_tokens=10, output_tokens=20, total_tokens=30)
        msg = AssistantMessage(content="x", usage=usage)
        assert msg.usage is not None
        assert msg.usage.total_tokens == 30

    def test_with_multiple_tool_calls(self):
        calls = [
            ToolCall(id="tc1", name="tool_a", arguments={}),
            ToolCall(id="tc2", name="tool_b", arguments={"k": "v"}),
        ]
        msg = AssistantMessage(content="", tool_calls=calls)
        assert len(msg.tool_calls) == 2


class TestAssistantMessageSerialization:
    """Round-trip serialization."""

    def test_round_trip_simple(self):
        msg = AssistantMessage(content="hello")
        restored = AssistantMessage.model_validate(msg.model_dump())
        assert restored.content == msg.content

    def test_round_trip_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1, "b": 2})
        msg = AssistantMessage(content="calling tool", tool_calls=[tc])
        restored = AssistantMessage.model_validate(msg.model_dump())
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "tc1"
        assert restored.tool_calls[0].arguments == {"a": 1, "b": 2}

    def test_round_trip_with_usage(self):
        usage = Usage(input_tokens=5, output_tokens=10, total_tokens=15)
        msg = AssistantMessage(content="x", usage=usage)
        restored = AssistantMessage.model_validate(msg.model_dump())
        assert restored.usage is not None
        assert restored.usage.input_tokens == 5

    def test_json_round_trip(self):
        msg = AssistantMessage(content="json test")
        restored = AssistantMessage.model_validate_json(msg.model_dump_json())
        assert restored.content == "json test"

    def test_dump_keys(self):
        data = AssistantMessage().model_dump()
        expected = {"role", "content", "timestamp", "metadata", "tool_calls", "usage"}
        assert set(data.keys()) == expected


# ---------------------------------------------------------------------------
# AssistantResponse
# ---------------------------------------------------------------------------


class TestAssistantResponseConstruction:
    """Construction and custom __init__ logic."""

    def test_required_fields(self):
        resp = AssistantResponse(agent_name="test_agent", content="Done")
        assert resp.agent_name == "test_agent"
        assert resp.content == "Done"

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            AssistantResponse(content="no agent name")

    def test_default_tool_results_is_none(self):
        resp = AssistantResponse(agent_name="a", content="x")
        assert resp.tool_results is None

    def test_default_error_is_none(self):
        resp = AssistantResponse(agent_name="a", content="x")
        assert resp.error is None

    def test_empty_tool_results_normalized_to_none(self):
        """Custom __init__ converts empty list to None."""
        resp = AssistantResponse(agent_name="a", content="x", tool_results=[])
        assert resp.tool_results is None

    def test_non_empty_tool_results_preserved(self):
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc1",
            result=42,
            status=ToolStatus.SUCCESS,
        )
        resp = AssistantResponse(agent_name="a", content="x", tool_results=[tr])
        assert resp.tool_results is not None
        assert len(resp.tool_results) == 1
        assert resp.tool_results[0].result == 42

    def test_with_error(self):
        resp = AssistantResponse(
            agent_name="a", content="failed", error="Something went wrong"
        )
        assert resp.error == "Something went wrong"


class TestAssistantResponseSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        resp = AssistantResponse(agent_name="bot", content="ok")
        restored = AssistantResponse.model_validate(resp.model_dump())
        assert restored.agent_name == "bot"
        assert restored.content == "ok"

    def test_json_round_trip_with_tool_results(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result={"key": "val"})
        resp = AssistantResponse(agent_name="a", content="done", tool_results=[tr])
        restored = AssistantResponse.model_validate_json(resp.model_dump_json())
        assert restored.tool_results is not None
        assert restored.tool_results[0].tool_name == "t"
