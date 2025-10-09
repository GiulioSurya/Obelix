"""
Unit tests for ToolCall, ToolResult, and ToolMessage classes.

Tests verify:
- ToolCall creation with various argument types
- ToolResult with success/error/timeout status
- ToolResult error truncation
- ToolMessage creation and content summary generation
"""

import pytest
from src.messages.base_message import MessageRole
from src.messages.tool_message import ToolCall, ToolResult, ToolMessage, ToolStatus


class TestToolCall:
    """Test suite for ToolCall class."""

    def test_tool_call_creation(self):
        """Test ToolCall creation with valid parameters."""
        tool_call = ToolCall(
            id="call_789",
            name="read_query",
            arguments={"query": "SELECT * FROM users"}
        )

        assert tool_call.id == "call_789"
        assert tool_call.name == "read_query"
        assert tool_call.arguments["query"] == "SELECT * FROM users"

    def test_tool_call_with_complex_arguments(self):
        """Test ToolCall with nested argument structure."""
        arguments = {
            "table": "orders",
            "filters": {"status": "active", "date": "2024-01-01"},
            "limit": 100
        }
        tool_call = ToolCall(
            id="call_complex",
            name="query_database",
            arguments=arguments
        )

        assert tool_call.arguments["filters"]["status"] == "active"
        assert tool_call.arguments["limit"] == 100

    def test_tool_call_empty_arguments(self):
        """Test ToolCall with empty arguments dict."""
        tool_call = ToolCall(
            id="call_empty",
            name="list_tables",
            arguments={}
        )

        assert tool_call.arguments == {}
        assert isinstance(tool_call.arguments, dict)

    def test_tool_call_with_list_arguments(self):
        """Test ToolCall with list in arguments."""
        tool_call = ToolCall(
            id="call_list",
            name="batch_process",
            arguments={"items": [1, 2, 3, 4, 5]}
        )

        assert len(tool_call.arguments["items"]) == 5
        assert tool_call.arguments["items"][0] == 1


class TestToolResult:
    """Test suite for ToolResult class."""

    def test_tool_result_success(self):
        """Test successful ToolResult creation."""
        result = ToolResult(
            tool_name="calculator",
            tool_call_id="call_123",
            result=42,
            status=ToolStatus.SUCCESS
        )

        assert result.tool_name == "calculator"
        assert result.tool_call_id == "call_123"
        assert result.result == 42
        assert result.status == ToolStatus.SUCCESS
        assert result.error is None

    def test_tool_result_error(self):
        """Test ToolResult with error status."""
        result = ToolResult(
            tool_name="database",
            tool_call_id="call_456",
            result=None,
            status=ToolStatus.ERROR,
            error="Connection refused"
        )

        assert result.status == ToolStatus.ERROR
        assert result.error == "Connection refused"
        assert result.result is None

    def test_tool_result_timeout(self):
        """Test ToolResult with timeout status."""
        result = ToolResult(
            tool_name="slow_api",
            tool_call_id="call_789",
            result=None,
            status=ToolStatus.TIMEOUT,
            error="Operation timed out after 30s"
        )

        assert result.status == ToolStatus.TIMEOUT
        assert "timed out" in result.error

    def test_tool_result_error_truncation(self):
        """Test that long error messages are truncated."""
        long_error = "Error: " + "x" * 1000  # Error longer than 500 chars
        result = ToolResult(
            tool_name="test",
            tool_call_id="call_trunc",
            result=None,
            status=ToolStatus.ERROR,
            error=long_error
        )

        # Error should be truncated to max 500 chars
        assert len(result.error) <= 500
        assert "truncated" in result.error

    def test_tool_result_with_execution_time(self):
        """Test ToolResult with execution time tracking."""
        result = ToolResult(
            tool_name="api_call",
            tool_call_id="call_time",
            result={"data": "response"},
            status=ToolStatus.SUCCESS,
            execution_time=0.234
        )

        assert result.execution_time == 0.234
        assert result.status == ToolStatus.SUCCESS

    def test_tool_result_with_dict_result(self):
        """Test ToolResult with dictionary result."""
        result_data = {"status": "ok", "count": 42, "items": [1, 2, 3]}
        result = ToolResult(
            tool_name="api",
            tool_call_id="call_dict",
            result=result_data,
            status=ToolStatus.SUCCESS
        )

        assert result.result["status"] == "ok"
        assert result.result["count"] == 42
        assert len(result.result["items"]) == 3

    def test_tool_result_with_list_result(self):
        """Test ToolResult with list result."""
        result = ToolResult(
            tool_name="query",
            tool_call_id="call_list",
            result=[{"id": 1}, {"id": 2}, {"id": 3}],
            status=ToolStatus.SUCCESS
        )

        assert isinstance(result.result, list)
        assert len(result.result) == 3


class TestToolMessage:
    """Test suite for ToolMessage class."""

    def test_tool_message_creation(self):
        """Test ToolMessage creation with tool results."""
        results = [
            ToolResult(
                tool_name="calc",
                tool_call_id="call_1",
                result=10,
                status=ToolStatus.SUCCESS
            )
        ]
        message = ToolMessage(tool_results=results)

        assert message.role == MessageRole.TOOL
        assert len(message.tool_results) == 1
        assert message.tool_results[0].result == 10

    def test_tool_message_content_summary_generation(self):
        """Test automatic content summary generation from tool results."""
        results = [
            ToolResult(
                tool_name="tool1",
                tool_call_id="call_1",
                result="success",
                status=ToolStatus.SUCCESS
            ),
            ToolResult(
                tool_name="tool2",
                tool_call_id="call_2",
                result=None,
                status=ToolStatus.ERROR,
                error="Failed"
            )
        ]
        message = ToolMessage(tool_results=results)

        # Content should contain summaries of both results
        assert "tool1" in message.content
        assert "tool2" in message.content
        assert "success" in message.content or "SUCCESS" in message.content.upper()

    def test_tool_message_empty_results(self):
        """Test ToolMessage with empty results list."""
        message = ToolMessage(tool_results=[])

        assert message.role == MessageRole.TOOL
        assert len(message.tool_results) == 0
        assert "No tool results" in message.content

    def test_tool_message_multiple_results(self):
        """Test ToolMessage with multiple tool results."""
        results = [
            ToolResult(
                tool_name=f"tool_{i}",
                tool_call_id=f"call_{i}",
                result=i * 10,
                status=ToolStatus.SUCCESS
            )
            for i in range(5)
        ]
        message = ToolMessage(tool_results=results)

        assert len(message.tool_results) == 5
        assert all(r.status == ToolStatus.SUCCESS for r in message.tool_results)

    def test_tool_message_mixed_status_results(self):
        """Test ToolMessage with mixed success/error results."""
        results = [
            ToolResult(
                tool_name="success_tool",
                tool_call_id="call_s",
                result="ok",
                status=ToolStatus.SUCCESS
            ),
            ToolResult(
                tool_name="error_tool",
                tool_call_id="call_e",
                result=None,
                status=ToolStatus.ERROR,
                error="Failed"
            ),
            ToolResult(
                tool_name="timeout_tool",
                tool_call_id="call_t",
                result=None,
                status=ToolStatus.TIMEOUT,
                error="Timed out"
            )
        ]
        message = ToolMessage(tool_results=results)

        assert len(message.tool_results) == 3
        assert message.tool_results[0].status == ToolStatus.SUCCESS
        assert message.tool_results[1].status == ToolStatus.ERROR
        assert message.tool_results[2].status == ToolStatus.TIMEOUT