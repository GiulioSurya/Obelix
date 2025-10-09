"""
Pytest configuration and shared fixtures.

This file contains pytest fixtures and configurations that are shared
across all test modules, following best practices from the PPT guidelines.

Fixtures provided:
- sample_human_message: Sample HumanMessage for testing
- sample_system_message: Sample SystemMessage for testing
- sample_assistant_message: Sample AssistantMessage for testing
- sample_tool_call: Sample ToolCall for testing
- sample_tool_result_success: Sample successful ToolResult
- sample_tool_result_error: Sample error ToolResult
- sample_conversation: Sample conversation history
"""

import pytest
import sys
from pathlib import Path
from typing import List

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolCall, ToolResult, ToolMessage, ToolStatus
from src.messages.standard_message import StandardMessage


# ============================================================================
# Message Fixtures
# ============================================================================

@pytest.fixture
def sample_human_message():
    """Provides a sample HumanMessage for testing."""
    return HumanMessage(content="What is the weather today?")


@pytest.fixture
def sample_system_message():
    """Provides a sample SystemMessage for testing."""
    return SystemMessage(content="You are a helpful AI assistant.")


@pytest.fixture
def sample_assistant_message():
    """Provides a sample AssistantMessage for testing."""
    return AssistantMessage(content="I can help you with that!")


@pytest.fixture
def sample_assistant_message_with_tools():
    """Provides an AssistantMessage with tool calls."""
    tool_calls = [
        ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Rome", "units": "celsius"}
        )
    ]
    return AssistantMessage(content="Let me check the weather", tool_calls=tool_calls)


# ============================================================================
# Tool Fixtures
# ============================================================================

@pytest.fixture
def sample_tool_call():
    """Provides a sample ToolCall for testing."""
    return ToolCall(
        id="call_test_123",
        name="test_tool",
        arguments={"param1": "value1", "param2": 42}
    )


@pytest.fixture
def sample_tool_result_success():
    """Provides a sample successful ToolResult."""
    return ToolResult(
        tool_name="test_tool",
        tool_call_id="call_test_123",
        result={"status": "success", "data": [1, 2, 3]},
        status=ToolStatus.SUCCESS,
        execution_time=0.125
    )


@pytest.fixture
def sample_tool_result_error():
    """Provides a sample error ToolResult."""
    return ToolResult(
        tool_name="test_tool",
        tool_call_id="call_test_456",
        result=None,
        status=ToolStatus.ERROR,
        error="Connection timeout after 30 seconds"
    )


@pytest.fixture
def sample_tool_message():
    """Provides a sample ToolMessage with results."""
    results = [
        ToolResult(
            tool_name="calculator",
            tool_call_id="call_calc",
            result=42,
            status=ToolStatus.SUCCESS
        )
    ]
    return ToolMessage(tool_results=results)


# ============================================================================
# Conversation Fixtures
# ============================================================================

@pytest.fixture
def sample_conversation():
    """Provides a sample conversation history."""
    return [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content="Hello!"),
        AssistantMessage(content="Hi! How can I help you?"),
        HumanMessage(content="What's 2+2?"),
        AssistantMessage(content="2+2 equals 4")
    ]


@pytest.fixture
def sample_conversation_with_tools():
    """Provides a conversation including tool usage."""
    return [
        SystemMessage(content="You are a calculator assistant"),
        HumanMessage(content="Calculate 15 * 23"),
        AssistantMessage(
            content="",
            tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expr": "15*23"})]
        ),
        ToolMessage(tool_results=[
            ToolResult(
                tool_name="calculator",
                tool_call_id="call_1",
                result=345,
                status=ToolStatus.SUCCESS
            )
        ]),
        AssistantMessage(content="15 * 23 = 345")
    ]


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Mark integration tests
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)