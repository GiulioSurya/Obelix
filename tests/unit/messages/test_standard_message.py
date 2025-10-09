"""
Unit tests for StandardMessage union type.

Tests verify:
- Union type accepts all message types
- Type checking works correctly
- Conversation history can be built with mixed message types
"""

import pytest
from typing import List

from src.messages.base_message import MessageRole
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage, ToolResult, ToolStatus
from src.messages.standard_message import StandardMessage


class TestStandardMessage:
    """Test suite for StandardMessage union type."""

    def test_standard_message_accepts_human_message(self):
        """Test that StandardMessage union accepts HumanMessage."""
        msg: StandardMessage = HumanMessage(content="Hello")
        assert isinstance(msg, HumanMessage)

    def test_standard_message_accepts_system_message(self):
        """Test that StandardMessage union accepts SystemMessage."""
        msg: StandardMessage = SystemMessage(content="System prompt")
        assert isinstance(msg, SystemMessage)

    def test_standard_message_accepts_assistant_message(self):
        """Test that StandardMessage union accepts AssistantMessage."""
        msg: StandardMessage = AssistantMessage(content="Response")
        assert isinstance(msg, AssistantMessage)

    def test_standard_message_accepts_tool_message(self):
        """Test that StandardMessage union accepts ToolMessage."""
        results = [
            ToolResult(
                tool_name="test",
                tool_call_id="123",
                result="ok",
                status=ToolStatus.SUCCESS
            )
        ]
        msg: StandardMessage = ToolMessage(tool_results=results)
        assert isinstance(msg, ToolMessage)

    def test_standard_message_list_conversation(self):
        """Test StandardMessage list representing a conversation."""
        conversation: List[StandardMessage] = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AssistantMessage(content="Hi there!")
        ]

        assert len(conversation) == 3
        assert conversation[0].role == MessageRole.SYSTEM
        assert conversation[1].role == MessageRole.HUMAN
        assert conversation[2].role == MessageRole.ASSISTANT

    def test_standard_message_complex_conversation_with_tools(self):
        """Test complex conversation including tool messages."""
        conversation: List[StandardMessage] = [
            SystemMessage(content="You are a calculator"),
            HumanMessage(content="Calculate 2+2"),
            AssistantMessage(content="", tool_calls=[]),  # Would have tool calls in real scenario
            ToolMessage(tool_results=[
                ToolResult(
                    tool_name="calc",
                    tool_call_id="call_1",
                    result=4,
                    status=ToolStatus.SUCCESS
                )
            ]),
            AssistantMessage(content="The result is 4")
        ]

        assert len(conversation) == 5
        assert isinstance(conversation[0], SystemMessage)
        assert isinstance(conversation[1], HumanMessage)
        assert isinstance(conversation[2], AssistantMessage)
        assert isinstance(conversation[3], ToolMessage)
        assert isinstance(conversation[4], AssistantMessage)

    def test_standard_message_all_have_role(self):
        """Test that all StandardMessage types have role attribute."""
        messages: List[StandardMessage] = [
            SystemMessage(content="sys"),
            HumanMessage(content="human"),
            AssistantMessage(content="assistant"),
            ToolMessage(tool_results=[])
        ]

        for msg in messages:
            assert hasattr(msg, 'role')
            assert isinstance(msg.role, MessageRole)

    def test_standard_message_all_have_content(self):
        """Test that all StandardMessage types have content attribute."""
        messages: List[StandardMessage] = [
            SystemMessage(content="sys"),
            HumanMessage(content="human"),
            AssistantMessage(content="assistant"),
            ToolMessage(tool_results=[])  # Auto-generates content
        ]

        for msg in messages:
            assert hasattr(msg, 'content')
            assert isinstance(msg.content, str)