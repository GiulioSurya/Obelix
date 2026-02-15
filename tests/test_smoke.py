"""Smoke tests to verify the test environment is properly configured.

These tests validate that:
- Fixtures from conftest.py load correctly
- Basic model classes instantiate without errors
- The mock provider satisfies the BaseAgent contract
- Async tests execute via pytest-asyncio (auto mode)
"""

from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.roles import MessageRole
from obelix.core.model.tool_message import ToolCall, ToolStatus

# ---------------------------------------------------------------------------
# Fixture sanity checks
# ---------------------------------------------------------------------------


class TestFixturesLoad:
    """Verify that all conftest fixtures are importable and return valid objects."""

    def test_mock_provider_has_invoke(self, mock_provider):
        assert hasattr(mock_provider, "invoke")
        assert mock_provider.model_id == "mock-model"

    def test_mock_provider_factory_creates_custom_response(self, mock_provider_factory):
        provider = mock_provider_factory(response_content="custom")
        assert provider._response_content == "custom"

    def test_sample_messages(
        self,
        sample_system_message,
        sample_human_message,
        sample_assistant_message,
    ):
        assert sample_system_message.role == MessageRole.SYSTEM
        assert sample_human_message.role == MessageRole.HUMAN
        assert sample_assistant_message.role == MessageRole.ASSISTANT

    def test_sample_tool_call(self, sample_tool_call):
        assert sample_tool_call.name == "calculator"
        assert sample_tool_call.arguments == {"a": 2, "b": 3}

    def test_sample_tool_result(self, sample_tool_result):
        assert sample_tool_result.status == ToolStatus.SUCCESS

    def test_sample_tool_result_error(self, sample_tool_result_error):
        assert sample_tool_result_error.status == ToolStatus.ERROR
        assert "Division by zero" in sample_tool_result_error.error

    def test_sample_tool_message(self, sample_tool_message):
        assert len(sample_tool_message.tool_results) == 1

    def test_calculator_tool_class(self, calculator_tool_class):
        assert calculator_tool_class.tool_name == "calculator"
        schema = calculator_tool_class.create_schema()
        assert schema.name == "calculator"

    def test_calculator_tool_instance(self, calculator_tool):
        assert calculator_tool.tool_name == "calculator"

    def test_async_calculator_tool_class(self, async_calculator_tool_class):
        assert async_calculator_tool_class.tool_name == "async_calculator"

    def test_failing_tool_class(self, failing_tool_class):
        assert failing_tool_class.tool_name == "failing_tool"


# ---------------------------------------------------------------------------
# Async execution check
# ---------------------------------------------------------------------------


class TestAsyncSupport:
    """Verify that async tests work with pytest-asyncio auto mode."""

    async def test_async_mock_provider_invoke(self, mock_provider):
        result = await mock_provider.invoke([], [])
        assert isinstance(result, AssistantMessage)
        assert result.content == "Mock response"

    async def test_async_calculator_executes(self, async_calculator_tool):
        tool_call = ToolCall(
            id="tc_async_1", name="async_calculator", arguments={"a": 10, "b": 20}
        )
        result = await async_calculator_tool.execute(tool_call)
        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"result": 30.0}

    async def test_sync_calculator_executes_via_to_thread(self, calculator_tool):
        tool_call = ToolCall(
            id="tc_sync_1", name="calculator", arguments={"a": 7, "b": 3}
        )
        result = await calculator_tool.execute(tool_call)
        assert result.status == ToolStatus.SUCCESS
        assert result.result == {"result": 10.0}


# ---------------------------------------------------------------------------
# BaseAgent construction
# ---------------------------------------------------------------------------


class TestBaseAgentConstruction:
    """Verify BaseAgent can be instantiated with the mock provider."""

    def test_make_agent_default(self, make_agent):
        agent = make_agent()
        assert agent.system_message.content == "You are a test agent."
        assert agent.max_iterations == 5

    def test_make_agent_custom(self, make_agent):
        agent = make_agent(system_message="Custom prompt", max_iterations=2)
        assert agent.system_message.content == "Custom prompt"
        assert agent.max_iterations == 2

    def test_agent_registers_tool(self, make_agent, calculator_tool):
        agent = make_agent()
        agent.register_tool(calculator_tool)
        assert len(agent.registered_tools) == 1
        assert agent.registered_tools[0].tool_name == "calculator"
