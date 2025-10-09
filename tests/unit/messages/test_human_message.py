"""
Unit tests for HumanMessage class.

Tests verify:
- Correct message creation with valid content
- Empty content handling
- Multiline content handling
"""

import pytest
from src.messages.base_message import BaseMessage, MessageRole
from src.messages.human_message import HumanMessage


class TestHumanMessage:
    """Test suite for HumanMessage class."""

    def test_human_message_creation_with_valid_content(self):
        """Test that HumanMessage is created correctly with valid content."""
        content = "What is the weather today?"
        message = HumanMessage(content=content)

        assert message.content == content
        assert message.role == MessageRole.HUMAN
        assert isinstance(message, BaseMessage)

    def test_human_message_empty_content(self):
        """Test that HumanMessage handles empty content correctly."""
        message = HumanMessage(content="")

        assert message.content == ""
        assert message.role == MessageRole.HUMAN

    def test_human_message_multiline_content(self):
        """Test that HumanMessage handles multiline content."""
        content = """This is a multiline
        message with multiple
        lines of text"""
        message = HumanMessage(content=content)

        assert message.content == content
        assert "\n" in message.content

    def test_human_message_has_timestamp(self):
        """Test that HumanMessage has automatic timestamp."""
        message = HumanMessage(content="Test")

        assert hasattr(message, 'timestamp')
        assert message.timestamp is not None

    def test_human_message_has_metadata(self):
        """Test that HumanMessage has metadata field."""
        message = HumanMessage(content="Test")

        assert hasattr(message, 'metadata')
        assert isinstance(message.metadata, dict)
        assert message.metadata == {}