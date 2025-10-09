"""
Unit tests for AssistantMessage and AssistantResponse classes.

Tests verify:
- Message creation with content only
- Message creation with tool calls only
- Message creation with both content and tool calls
- AssistantResponse structure and fields
"""

import pytest
from src.messages.base_message import BaseMessage, MessageRole
from src.messages.assistant_message import AssistantMessage, AssistantResponse
from src.messages.tool_message import ToolCall


class TestAssistantMessage:
    """Test suite for AssistantMessage class."""

    def test_assistant_message_with_content_only(self):
        """Test AssistantMessage with text content and no tool calls."""
        content = "The weather is sunny today."
        message = AssistantMessage(content=content)

        assert message.content == content
        assert message.role == MessageRole.ASSISTANT
        assert message.tool_calls == []  # Default is empty list, not None

    def test_assistant_message_with_tool_calls_only(self):
        """Test AssistantMessage with tool calls and empty content."""
        tool_calls = [
            ToolCall(
                id="call_123",
                name="get_weather",
                arguments={"location": "Rome"}
            )
        ]
        # Content cannot be None, use empty string instead
        message = AssistantMessage(content="", tool_calls=tool_calls)

        assert message.content == ""
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].name == "get_weather"

    def test_assistant_message_with_both_content_and_tool_calls(self):
        """Test AssistantMessage with both content and tool calls."""
        content = "Let me check the weather for you."
        tool_calls = [
            ToolCall(
                id="call_456",
                name="get_weather",
                arguments={"location": "Milan"}
            )
        ]
        message = AssistantMessage(content=content, tool_calls=tool_calls)

        assert message.content == content
        assert len(message.tool_calls) == 1

    def test_assistant_message_with_default_empty_content(self):
        """Test that AssistantMessage can be created with default empty content."""
        # Empty content with empty tool calls list is valid (defaults)
        message = AssistantMessage()

        # Both have default values
        assert message.content == ""  # Default empty string
        assert message.tool_calls == []  # Default empty list

    def test_assistant_message_with_multiple_tool_calls(self):
        """Test AssistantMessage with multiple tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={"arg": 1}),
            ToolCall(id="call_2", name="tool2", arguments={"arg": 2}),
            ToolCall(id="call_3", name="tool3", arguments={"arg": 3})
        ]
        message = AssistantMessage(content="Using multiple tools", tool_calls=tool_calls)

        assert len(message.tool_calls) == 3
        assert message.tool_calls[0].id == "call_1"
        assert message.tool_calls[2].id == "call_3"


class TestAssistantResponse:
    """Test suite for AssistantResponse class."""

    def test_assistant_response_basic(self):
        """Test basic AssistantResponse creation."""
        response = AssistantResponse(
            agent_name="test-agent",
            content="Task completed successfully"
        )

        assert response.agent_name == "test-agent"
        assert response.content == "Task completed successfully"
        assert response.tool_results is None
        assert response.error is None

    def test_assistant_response_with_tool_results(self):
        """Test AssistantResponse with tool execution results."""
        tool_results = [
            {"tool_name": "calculator", "result": 42}
        ]
        response = AssistantResponse(
            agent_name="math-agent",
            content="Calculation complete",
            tool_results=tool_results
        )

        assert len(response.tool_results) == 1
        assert response.tool_results[0]["tool_name"] == "calculator"

    def test_assistant_response_with_error(self):
        """Test AssistantResponse with error message."""
        response = AssistantResponse(
            agent_name="failing-agent",
            content="Task failed",
            error="Database connection timeout"
        )

        assert response.error == "Database connection timeout"
        assert response.content == "Task failed"

    def test_assistant_response_with_multiple_tool_results(self):
        """Test AssistantResponse with multiple tool results."""
        tool_results = [
            {"tool_name": "tool1", "result": "result1"},
            {"tool_name": "tool2", "result": "result2"},
            {"tool_name": "tool3", "result": "result3"}
        ]
        response = AssistantResponse(
            agent_name="multi-tool-agent",
            content="All tools executed",
            tool_results=tool_results
        )

        assert len(response.tool_results) == 3
        assert response.tool_results[1]["tool_name"] == "tool2"

    def test_assistant_response_empty_tool_results_becomes_none(self):
        """Test that empty tool_results list becomes None."""
        response = AssistantResponse(
            agent_name="agent",
            content="No tools used",
            tool_results=[]
        )

        # Empty list should be converted to None in __init__
        assert response.tool_results is None