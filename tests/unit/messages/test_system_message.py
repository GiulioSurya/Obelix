"""
Unit tests for SystemMessage class.

Tests verify:
- Correct message creation with valid content
- Complex instruction handling
- System role assignment
"""

import pytest
from src.messages.base_message import BaseMessage, MessageRole
from src.messages.system_message import SystemMessage


class TestSystemMessage:
    """Test suite for SystemMessage class."""

    def test_system_message_creation_with_valid_content(self):
        """Test that SystemMessage is created correctly with valid content."""
        content = "You are a helpful AI assistant."
        message = SystemMessage(content=content)

        assert message.content == content
        assert message.role == MessageRole.SYSTEM
        assert isinstance(message, BaseMessage)

    def test_system_message_with_instructions(self):
        """Test SystemMessage with complex instructions."""
        content = """You are an expert SQLite analyst.

        Rules:
        - Always validate input
        - Return structured results
        """
        message = SystemMessage(content=content)

        assert "Rules:" in message.content
        assert message.role == MessageRole.SYSTEM

    def test_system_message_empty_content(self):
        """Test SystemMessage with empty content."""
        message = SystemMessage(content="")

        assert message.content == ""
        assert message.role == MessageRole.SYSTEM

    def test_system_message_with_special_characters(self):
        """Test SystemMessage handles special characters."""
        content = "System prompt with special chars: @#$%^&*()"
        message = SystemMessage(content=content)

        assert "@#$%^&*()" in message.content
        assert message.role == MessageRole.SYSTEM