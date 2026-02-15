"""Shared fixtures for Obelix test suite.

Provides mock providers, sample messages, sample tools, and helper factories
used across all test modules. Fixtures are organized by domain concept.
"""

from unittest.mock import AsyncMock

import pytest

from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import (
    ToolCall,
    ToolMessage,
    ToolResult,
    ToolStatus,
)

# ---------------------------------------------------------------------------
# Mock Provider
# ---------------------------------------------------------------------------


class MockProvider:
    """Minimal mock that satisfies AbstractLLMProvider contract for BaseAgent.

    Attributes:
        model_id: Fake model identifier.
        provider_type: Always ``"mock"``.
        invoke: ``AsyncMock`` returning an ``AssistantMessage``.
    """

    def __init__(self, response_content: str = "Mock response") -> None:
        self.model_id = "mock-model"
        self.provider_type = "mock"
        self._response_content = response_content
        self.invoke = AsyncMock(return_value=AssistantMessage(content=response_content))


@pytest.fixture
def mock_provider() -> MockProvider:
    """A ``MockProvider`` that returns ``'Mock response'``."""
    return MockProvider()


@pytest.fixture
def mock_provider_factory():
    """Factory fixture for creating MockProvider with custom responses."""

    def _create(response_content: str = "Mock response") -> MockProvider:
        return MockProvider(response_content=response_content)

    return _create


@pytest.fixture
def mock_provider_with_tool_calls():
    """Factory fixture for creating a MockProvider whose response contains tool calls.

    Usage::

        def test_example(mock_provider_with_tool_calls):
            provider = mock_provider_with_tool_calls(
                tool_calls=[ToolCall(id="tc_1", name="my_tool", arguments={"x": 1})]
            )
    """

    def _create(
        tool_calls: list[ToolCall] | None = None,
        content: str = "",
    ) -> MockProvider:
        provider = MockProvider(response_content=content)
        provider.invoke = AsyncMock(
            return_value=AssistantMessage(content=content, tool_calls=tool_calls or [])
        )
        return provider

    return _create


# ---------------------------------------------------------------------------
# Sample Messages
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_system_message() -> SystemMessage:
    return SystemMessage(content="You are a helpful assistant.")


@pytest.fixture
def sample_human_message() -> HumanMessage:
    return HumanMessage(content="Hello, how are you?")


@pytest.fixture
def sample_assistant_message() -> AssistantMessage:
    return AssistantMessage(content="I am fine, thank you!")


@pytest.fixture
def sample_tool_call() -> ToolCall:
    return ToolCall(
        id="tc_test_001",
        name="calculator",
        arguments={"a": 2, "b": 3},
    )


@pytest.fixture
def sample_tool_result() -> ToolResult:
    return ToolResult(
        tool_name="calculator",
        tool_call_id="tc_test_001",
        result={"sum": 5},
        status=ToolStatus.SUCCESS,
    )


@pytest.fixture
def sample_tool_result_error() -> ToolResult:
    return ToolResult(
        tool_name="calculator",
        tool_call_id="tc_test_002",
        result=None,
        status=ToolStatus.ERROR,
        error="Division by zero",
    )


@pytest.fixture
def sample_tool_message(sample_tool_result: ToolResult) -> ToolMessage:
    return ToolMessage(tool_results=[sample_tool_result])


# ---------------------------------------------------------------------------
# BaseAgent helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def make_agent(mock_provider):
    """Factory fixture to build a BaseAgent with sensible defaults.

    Returns a callable that accepts keyword overrides::

        agent = make_agent(system_message="custom prompt", max_iterations=3)
    """
    from obelix.core.agent.base_agent import BaseAgent

    def _create(**kwargs):
        defaults = {
            "system_message": "You are a test agent.",
            "provider": mock_provider,
            "max_iterations": 5,
        }
        defaults.update(kwargs)
        return BaseAgent(**defaults)

    return _create


# ---------------------------------------------------------------------------
# Sample Tool (decorated) for integration-style tests
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator_tool_class():
    """Return a decorated Tool subclass suitable for registration on agents.

    This fixture returns the *class*, not an instance, so tests can instantiate
    it multiple times if needed.
    """
    from pydantic import Field

    from obelix.core.tool.tool_base import Tool
    from obelix.core.tool.tool_decorator import tool

    @tool(name="calculator", description="Adds two numbers")
    class CalculatorTool(Tool):
        a: float = Field(..., description="First operand")
        b: float = Field(..., description="Second operand")

        def execute(self) -> dict:
            return {"result": self.a + self.b}

    return CalculatorTool


@pytest.fixture
def calculator_tool(calculator_tool_class):
    """A ready-to-use instance of CalculatorTool."""
    return calculator_tool_class()


@pytest.fixture
def async_calculator_tool_class():
    """Return a decorated async Tool subclass."""
    from pydantic import Field

    from obelix.core.tool.tool_base import Tool
    from obelix.core.tool.tool_decorator import tool

    @tool(name="async_calculator", description="Adds two numbers (async)")
    class AsyncCalculatorTool(Tool):
        a: float = Field(..., description="First operand")
        b: float = Field(..., description="Second operand")

        async def execute(self) -> dict:
            return {"result": self.a + self.b}

    return AsyncCalculatorTool


@pytest.fixture
def async_calculator_tool(async_calculator_tool_class):
    """A ready-to-use instance of AsyncCalculatorTool."""
    return async_calculator_tool_class()


@pytest.fixture
def failing_tool_class():
    """A tool whose execute() always raises an exception."""
    from pydantic import Field

    from obelix.core.tool.tool_base import Tool
    from obelix.core.tool.tool_decorator import tool

    @tool(name="failing_tool", description="Always fails")
    class FailingTool(Tool):
        input_value: str = Field(..., description="Any input")

        def execute(self) -> dict:
            raise RuntimeError("Intentional failure for testing")

    return FailingTool
