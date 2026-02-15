"""Tests for obelix.core.model.tool_message — ToolCall, ToolStatus, ToolRequirement, ToolResult, ToolMessage, MCPToolSchema."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from obelix.core.model.roles import MessageRole
from obelix.core.model.tool_message import (
    MCPToolSchema,
    ToolCall,
    ToolMessage,
    ToolRequirement,
    ToolResult,
    ToolStatus,
)

# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


class TestToolCall:
    """Tests for ToolCall model."""

    def test_construction(self):
        tc = ToolCall(id="tc1", name="calc", arguments={"a": 1})
        assert tc.id == "tc1"
        assert tc.name == "calc"
        assert tc.arguments == {"a": 1}

    def test_all_fields_required(self):
        with pytest.raises(ValidationError):
            ToolCall(id="tc1", name="calc")  # missing arguments

    def test_empty_arguments(self):
        tc = ToolCall(id="tc1", name="no_args", arguments={})
        assert tc.arguments == {}

    def test_complex_arguments(self):
        args = {"nested": {"list": [1, 2]}, "flag": True, "text": "hello"}
        tc = ToolCall(id="tc1", name="complex", arguments=args)
        assert tc.arguments["nested"]["list"] == [1, 2]

    def test_round_trip(self):
        tc = ToolCall(id="tc1", name="t", arguments={"x": 42})
        restored = ToolCall.model_validate(tc.model_dump())
        assert restored.id == tc.id
        assert restored.arguments == tc.arguments

    def test_json_round_trip(self):
        tc = ToolCall(id="tc1", name="t", arguments={"k": "v"})
        restored = ToolCall.model_validate_json(tc.model_dump_json())
        assert restored.name == "t"


# ---------------------------------------------------------------------------
# ToolStatus
# ---------------------------------------------------------------------------


class TestToolStatus:
    """Tests for ToolStatus enum."""

    def test_values(self):
        assert ToolStatus.SUCCESS == "success"
        assert ToolStatus.ERROR == "error"
        assert ToolStatus.TIMEOUT == "timeout"

    def test_member_count(self):
        assert len(ToolStatus) == 3

    def test_is_str_subclass(self):
        assert isinstance(ToolStatus.SUCCESS, str)

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ToolStatus("unknown")


# ---------------------------------------------------------------------------
# ToolRequirement
# ---------------------------------------------------------------------------


class TestToolRequirement:
    """Tests for ToolRequirement model."""

    def test_construction_minimal(self):
        req = ToolRequirement(tool_name="search")
        assert req.tool_name == "search"
        assert req.min_calls == 1
        assert req.require_success is False
        assert req.error_message is None

    def test_construction_full(self):
        req = ToolRequirement(
            tool_name="db_query",
            min_calls=3,
            require_success=True,
            error_message="Must query database at least 3 times",
        )
        assert req.min_calls == 3
        assert req.require_success is True
        assert "3 times" in req.error_message

    def test_missing_tool_name_raises(self):
        with pytest.raises(ValidationError):
            ToolRequirement()

    def test_round_trip(self):
        req = ToolRequirement(tool_name="t", min_calls=2)
        restored = ToolRequirement.model_validate(req.model_dump())
        assert restored.tool_name == "t"
        assert restored.min_calls == 2


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResultConstruction:
    """Construction and default values."""

    def test_minimal_construction(self):
        tr = ToolResult(tool_name="calc", tool_call_id="tc1", result=42)
        assert tr.tool_name == "calc"
        assert tr.tool_call_id == "tc1"
        assert tr.result == 42
        assert tr.status == ToolStatus.SUCCESS
        assert tr.error is None
        assert tr.execution_time is None

    def test_with_error(self):
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc1",
            result=None,
            status=ToolStatus.ERROR,
            error="Division by zero",
        )
        assert tr.status == ToolStatus.ERROR
        assert tr.error == "Division by zero"

    def test_with_execution_time(self):
        tr = ToolResult(
            tool_name="calc", tool_call_id="tc1", result=5, execution_time=0.123
        )
        assert tr.execution_time == pytest.approx(0.123)

    def test_result_can_be_any_type(self):
        """Result field accepts Any type."""
        for value in [42, "text", [1, 2], {"k": "v"}, None, True]:
            tr = ToolResult(tool_name="t", tool_call_id="tc1", result=value)
            assert tr.result == value


class TestToolResultErrorTruncation:
    """Tests for the error truncation logic in ToolResult.__init__."""

    def test_short_error_not_truncated(self):
        error = "Short error message"
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=None, error=error)
        assert tr.error == error

    def test_error_at_max_length_not_truncated(self):
        error = "x" * 2000
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=None, error=error)
        assert tr.error == error
        assert len(tr.error) == 2000

    def test_error_over_max_length_is_truncated(self):
        error = "A" * 3000
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=None, error=error)
        assert len(tr.error) <= 2000
        assert "[truncated]" in tr.error

    def test_truncated_error_preserves_head_and_tail(self):
        head = "HEAD" * 250  # 1000 chars
        tail = "TAIL" * 250  # 1000 chars
        middle = "M" * 1500
        error = head + middle + tail
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=None, error=error)
        assert tr.error.startswith("HEAD")
        assert tr.error.endswith("TAIL")

    def test_none_error_stays_none(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=42, error=None)
        assert tr.error is None

    def test_empty_error_stays_empty(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=42, error="")
        assert tr.error == ""

    def test_truncate_static_method_directly(self):
        """Test _truncate_error_if_needed as a standalone static method."""
        short = "short"
        assert ToolResult._truncate_error_if_needed(short) == short

        long_msg = "Z" * 5000
        truncated = ToolResult._truncate_error_if_needed(long_msg)
        assert len(truncated) <= 2000
        assert "[truncated]" in truncated

    def test_truncate_with_custom_max_length(self):
        msg = "X" * 500
        result = ToolResult._truncate_error_if_needed(msg, max_length=100)
        assert len(result) <= 100
        assert "[truncated]" in result

    def test_truncate_preserves_none_input(self):
        assert ToolResult._truncate_error_if_needed(None) is None

    def test_truncate_preserves_empty_string(self):
        assert ToolResult._truncate_error_if_needed("") == ""


class TestToolResultSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc1",
            result={"sum": 5},
            status=ToolStatus.SUCCESS,
            execution_time=1.5,
        )
        restored = ToolResult.model_validate(tr.model_dump())
        assert restored.tool_name == "calc"
        assert restored.result == {"sum": 5}
        assert restored.execution_time == pytest.approx(1.5)

    def test_json_round_trip(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result="ok")
        restored = ToolResult.model_validate_json(tr.model_dump_json())
        assert restored.result == "ok"


# ---------------------------------------------------------------------------
# ToolMessage
# ---------------------------------------------------------------------------


class TestToolMessageConstruction:
    """Construction and default values."""

    def test_construction_with_results(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=42)
        msg = ToolMessage(tool_results=[tr])
        assert msg.role == MessageRole.TOOL
        assert len(msg.tool_results) == 1

    def test_timestamp_auto_generated(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=42)
        before = datetime.now()
        msg = ToolMessage(tool_results=[tr])
        after = datetime.now()
        assert before <= msg.timestamp <= after

    def test_default_metadata_is_empty(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=42)
        msg = ToolMessage(tool_results=[tr])
        assert msg.metadata == {}

    def test_multiple_tool_results(self):
        results = [
            ToolResult(tool_name="t1", tool_call_id="tc1", result=1),
            ToolResult(tool_name="t2", tool_call_id="tc2", result=2),
        ]
        msg = ToolMessage(tool_results=results)
        assert len(msg.tool_results) == 2

    def test_tool_results_required(self):
        """ToolMessage requires tool_results (it is Field(...))."""
        with pytest.raises(TypeError):
            ToolMessage()


class TestToolMessageSerialization:
    """Round-trip serialization."""

    def test_round_trip(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result="ok")
        msg = ToolMessage(tool_results=[tr])
        restored = ToolMessage.model_validate(msg.model_dump())
        assert len(restored.tool_results) == 1
        assert restored.tool_results[0].result == "ok"

    def test_json_round_trip(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=99)
        msg = ToolMessage(tool_results=[tr], metadata={"k": "v"})
        restored = ToolMessage.model_validate_json(msg.model_dump_json())
        assert restored.metadata == {"k": "v"}
        assert restored.tool_results[0].result == 99

    def test_dump_keys(self):
        tr = ToolResult(tool_name="t", tool_call_id="tc1", result=1)
        data = ToolMessage(tool_results=[tr]).model_dump()
        assert set(data.keys()) == {"role", "timestamp", "metadata", "tool_results"}


# ---------------------------------------------------------------------------
# MCPToolSchema
# ---------------------------------------------------------------------------


class TestMCPToolSchema:
    """Tests for MCPToolSchema model."""

    def test_construction_minimal(self):
        schema = MCPToolSchema(
            name="my_tool",
            description="Does something",
            inputSchema={"type": "object", "properties": {}},
        )
        assert schema.name == "my_tool"
        assert schema.description == "Does something"
        assert schema.title is None
        assert schema.outputSchema is None
        assert schema.annotations is None

    def test_construction_full(self):
        schema = MCPToolSchema(
            name="my_tool",
            description="Does something",
            title="My Tool",
            inputSchema={"type": "object", "properties": {"x": {"type": "integer"}}},
            outputSchema={"type": "object"},
            annotations={"readOnlyHint": True},
        )
        assert schema.title == "My Tool"
        assert schema.outputSchema == {"type": "object"}
        assert schema.annotations["readOnlyHint"] is True

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            MCPToolSchema(name="t")  # missing description and inputSchema

    def test_round_trip(self):
        schema = MCPToolSchema(
            name="t",
            description="d",
            inputSchema={"type": "object"},
        )
        restored = MCPToolSchema.model_validate(schema.model_dump())
        assert restored.name == "t"
        assert restored.inputSchema == {"type": "object"}

    def test_validate_assignment_enabled(self):
        """Config has validate_assignment=True, so assignment triggers validation."""
        schema = MCPToolSchema(
            name="t", description="d", inputSchema={"type": "object"}
        )
        # Valid reassignment should work
        schema.name = "new_name"
        assert schema.name == "new_name"
