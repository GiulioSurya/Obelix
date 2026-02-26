"""Tests for obelix.core.agent.base_agent.BaseAgent.

Covers constructor, register_tool, _validate_query_input, execute_query (sync),
execute_query_async, hook integration (STOP/FAIL/RETRY), tool execution,
exit_on_success, max_iterations, _build_final_response, _collect_tool_results,
clear_conversation_history, register_agent, shared memory no-ops, and tool_policy.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, HookDecision
from obelix.core.model.assistant_message import AssistantMessage, AssistantResponse
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import (
    ToolCall,
    ToolMessage,
    ToolRequirement,
    ToolResult,
    ToolStatus,
)

# ---------------------------------------------------------------------------
# Constructor and initialization
# ---------------------------------------------------------------------------


class TestBaseAgentConstructor:
    """Tests for BaseAgent.__init__ and initial state."""

    def test_system_message_converted_to_system_message(self, mock_provider):
        """String system_message is wrapped in SystemMessage."""
        agent = BaseAgent(system_message="Hello", provider=mock_provider)
        assert isinstance(agent.system_message, SystemMessage)
        assert agent.system_message.content == "Hello"

    def test_conversation_history_starts_with_system_message(self, mock_provider):
        """conversation_history starts with the system message."""
        agent = BaseAgent(system_message="Hi", provider=mock_provider)
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0] is agent.system_message

    def test_max_iterations_default(self, mock_provider):
        """Default max_iterations is 15."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        assert agent.max_iterations == 15

    def test_max_iterations_custom(self, mock_provider):
        """Custom max_iterations is respected."""
        agent = BaseAgent(system_message="x", provider=mock_provider, max_iterations=3)
        assert agent.max_iterations == 3

    def test_tools_none_gives_empty_list(self, mock_provider):
        """tools=None results in empty registered_tools."""
        agent = BaseAgent(system_message="x", provider=mock_provider, tools=None)
        assert agent.registered_tools == []

    def test_tools_single_class_auto_instantiated(
        self, mock_provider, calculator_tool_class
    ):
        """tools=<class> is instantiated and registered."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=calculator_tool_class,
        )
        assert len(agent.registered_tools) == 1
        assert agent.registered_tools[0].tool_name == "calculator"

    def test_tools_single_instance_registered(self, mock_provider, calculator_tool):
        """tools=<instance> is registered directly."""
        agent = BaseAgent(
            system_message="x", provider=mock_provider, tools=calculator_tool
        )
        assert len(agent.registered_tools) == 1
        assert agent.registered_tools[0] is calculator_tool

    def test_tools_list_mixed_class_and_instance(
        self, mock_provider, calculator_tool_class, async_calculator_tool
    ):
        """tools=[class, instance] registers both."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=[calculator_tool_class, async_calculator_tool],
        )
        assert len(agent.registered_tools) == 2
        names = {t.tool_name for t in agent.registered_tools}
        assert names == {"calculator", "async_calculator"}

    def test_tool_policy_none_gives_empty_list(self, mock_provider):
        """tool_policy=None results in empty _tool_policy."""
        agent = BaseAgent(system_message="x", provider=mock_provider, tool_policy=None)
        assert agent._tool_policy == []

    def test_exit_on_success_none_gives_empty_set(self, mock_provider):
        """exit_on_success=None results in empty set."""
        agent = BaseAgent(
            system_message="x", provider=mock_provider, exit_on_success=None
        )
        assert agent._exit_on_success == set()

    def test_exit_on_success_list_converted_to_set(self, mock_provider):
        """exit_on_success=['a', 'b'] is stored as a set."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            exit_on_success=["calc", "search"],
        )
        assert agent._exit_on_success == {"calc", "search"}

    def test_hooks_dict_initialized_for_all_events(self, mock_provider):
        """_hooks dict has an entry for every AgentEvent."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        for event in AgentEvent:
            assert event in agent._hooks

    def test_memory_hooks_registered_in_constructor(self, mock_provider):
        """Memory hooks are registered in the constructor (BEFORE_LLM_CALL, BEFORE_FINAL_RESPONSE)."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        # Memory hooks are the first ones registered (before any tool_policy hooks)
        assert len(agent._hooks[AgentEvent.BEFORE_LLM_CALL]) >= 1
        assert len(agent._hooks[AgentEvent.BEFORE_FINAL_RESPONSE]) >= 1

    def test_memory_graph_and_agent_id_default_to_none(self, mock_provider):
        """memory_graph and agent_id default to None."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        assert agent.memory_graph is None
        assert agent.agent_id is None

    def test_provider_stored(self, mock_provider):
        """provider attribute stores the given provider."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        assert agent.provider is mock_provider


# ---------------------------------------------------------------------------
# register_tool
# ---------------------------------------------------------------------------


class TestRegisterTool:
    """Tests for BaseAgent.register_tool."""

    def test_adds_tool_to_registered_tools(self, mock_provider, calculator_tool):
        """register_tool appends the tool."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.register_tool(calculator_tool)
        assert calculator_tool in agent.registered_tools

    def test_no_duplicates(self, mock_provider, calculator_tool):
        """Registering the same tool instance twice does not create duplicates."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.register_tool(calculator_tool)
        agent.register_tool(calculator_tool)
        assert len(agent.registered_tools) == 1

    def test_different_tools_both_registered(
        self, mock_provider, calculator_tool, async_calculator_tool
    ):
        """Two different tools are both registered."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.register_tool(calculator_tool)
        agent.register_tool(async_calculator_tool)
        assert len(agent.registered_tools) == 2


# ---------------------------------------------------------------------------
# _validate_query_input
# ---------------------------------------------------------------------------


class TestValidateQueryInput:
    """Tests for BaseAgent._validate_query_input."""

    def test_string_is_valid(self, mock_provider):
        """A plain string is accepted."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent._validate_query_input("hello")  # should not raise

    def test_list_with_one_human_message_is_valid(self, mock_provider):
        """A list containing exactly 1 HumanMessage is accepted."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent._validate_query_input([HumanMessage(content="hi")])

    def test_list_with_zero_human_messages_raises_value_error(self, mock_provider):
        """A list with no HumanMessage raises ValueError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        with pytest.raises(ValueError, match="exactly 1 HumanMessage"):
            agent._validate_query_input([SystemMessage(content="sys")])

    def test_list_with_two_human_messages_raises_value_error(self, mock_provider):
        """A list with 2 HumanMessages raises ValueError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msgs = [HumanMessage(content="a"), HumanMessage(content="b")]
        with pytest.raises(ValueError, match="exactly 1 HumanMessage"):
            agent._validate_query_input(msgs)

    def test_int_raises_type_error(self, mock_provider):
        """An int input raises TypeError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        with pytest.raises(TypeError, match="query must be str"):
            agent._validate_query_input(42)

    def test_none_raises_type_error(self, mock_provider):
        """None input raises TypeError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        with pytest.raises(TypeError, match="query must be str"):
            agent._validate_query_input(None)


# ---------------------------------------------------------------------------
# execute_query (sync wrapper)
# ---------------------------------------------------------------------------


class TestExecuteQuerySync:
    """Tests for BaseAgent.execute_query (synchronous)."""

    def test_simple_response(self, mock_provider):
        """Sync execute_query returns AssistantResponse with content."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        response = agent.execute_query("hello")
        assert isinstance(response, AssistantResponse)
        assert response.content == "Mock response"

    def test_history_updated_after_call(self, mock_provider):
        """conversation_history grows after execute_query."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.execute_query("hello")
        # system + human + assistant = 3
        assert len(agent.conversation_history) >= 3
        types = [type(m).__name__ for m in agent.conversation_history]
        assert "HumanMessage" in types
        assert "AssistantMessage" in types


# ---------------------------------------------------------------------------
# execute_query_async
# ---------------------------------------------------------------------------


class TestExecuteQueryAsync:
    """Tests for BaseAgent.execute_query_async (async core loop)."""

    @pytest.mark.asyncio
    async def test_simple_response(self, mock_provider):
        """Async query returns AssistantResponse."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        response = await agent.execute_query_async("hello")
        assert isinstance(response, AssistantResponse)
        assert response.content == "Mock response"

    @pytest.mark.asyncio
    async def test_provider_invoked_with_history_and_tools(self, mock_provider):
        """provider.invoke is called with conversation_history, tools, and response_schema."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        await agent.execute_query_async("hello")
        mock_provider.invoke.assert_awaited_once()
        args = mock_provider.invoke.call_args
        assert isinstance(args[0][0], list)  # conversation_history
        assert isinstance(args[0][1], list)  # registered_tools

    @pytest.mark.asyncio
    async def test_before_llm_call_hook_stop_skips_provider(self, mock_provider):
        """BEFORE_LLM_CALL hook returning STOP bypasses provider.invoke."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        stop_msg = AssistantMessage(content="Stopped by hook")
        agent.on(AgentEvent.BEFORE_LLM_CALL).handle(HookDecision.STOP, value=stop_msg)
        response = await agent.execute_query_async("hello")
        assert response.content == "Stopped by hook"
        mock_provider.invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_before_llm_call_hook_fail_raises(self, mock_provider):
        """BEFORE_LLM_CALL hook returning FAIL raises RuntimeError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.on(AgentEvent.BEFORE_LLM_CALL).handle(HookDecision.FAIL)
        with pytest.raises(RuntimeError, match="BEFORE_LLM_CALL requested FAIL"):
            await agent.execute_query_async("hello")

    @pytest.mark.asyncio
    async def test_after_llm_call_hook_retry_calls_provider_twice(self, mock_provider):
        """AFTER_LLM_CALL hook returning RETRY causes a second provider.invoke call."""
        call_count = 0

        async def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return AssistantMessage(content=f"response_{call_count}")

        mock_provider.invoke = AsyncMock(side_effect=invoke_side_effect)

        agent = BaseAgent(system_message="x", provider=mock_provider, max_iterations=3)
        # RETRY only on the first call
        agent.on(AgentEvent.AFTER_LLM_CALL).when(lambda s: s.iteration == 1).handle(
            HookDecision.RETRY
        )

        response = await agent.execute_query_async("hello")
        assert call_count == 2
        assert response.content == "response_2"

    @pytest.mark.asyncio
    async def test_after_llm_call_hook_stop_returns_response(self, mock_provider):
        """AFTER_LLM_CALL hook returning STOP returns the hook's value as response."""
        stop_msg = AssistantMessage(content="Stopped after LLM")
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.on(AgentEvent.AFTER_LLM_CALL).handle(HookDecision.STOP, value=stop_msg)
        response = await agent.execute_query_async("hello")
        assert response.content == "Stopped after LLM"

    @pytest.mark.asyncio
    async def test_after_llm_call_hook_fail_raises(self, mock_provider):
        """AFTER_LLM_CALL hook returning FAIL raises RuntimeError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.on(AgentEvent.AFTER_LLM_CALL).handle(HookDecision.FAIL)
        with pytest.raises(RuntimeError, match="AFTER_LLM_CALL requested FAIL"):
            await agent.execute_query_async("hello")

    @pytest.mark.asyncio
    async def test_before_final_response_hook_retry_re_calls_provider(
        self, mock_provider
    ):
        """BEFORE_FINAL_RESPONSE hook returning RETRY causes another LLM call."""
        call_count = 0

        async def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return AssistantMessage(content=f"resp_{call_count}")

        mock_provider.invoke = AsyncMock(side_effect=invoke_side_effect)

        agent = BaseAgent(system_message="x", provider=mock_provider, max_iterations=3)
        agent.on(AgentEvent.BEFORE_FINAL_RESPONSE).when(
            lambda s: s.iteration == 1
        ).handle(HookDecision.RETRY)

        await agent.execute_query_async("hello")
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_before_final_response_hook_fail_raises(self, mock_provider):
        """BEFORE_FINAL_RESPONSE hook returning FAIL raises RuntimeError."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.on(AgentEvent.BEFORE_FINAL_RESPONSE).handle(HookDecision.FAIL)
        with pytest.raises(RuntimeError, match="Required tool call missing"):
            await agent.execute_query_async("hello")

    @pytest.mark.asyncio
    async def test_query_as_message_list(self, mock_provider):
        """execute_query_async accepts a list with one HumanMessage."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        response = await agent.execute_query_async([HumanMessage(content="from list")])
        assert isinstance(response, AssistantResponse)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


class TestToolExecution:
    """Tests for tool call processing in the agent loop."""

    @pytest.mark.asyncio
    async def test_tool_called_and_result_in_history(
        self, mock_provider, calculator_tool
    ):
        """Provider returns tool_calls, tool is executed, history updated."""
        tc = ToolCall(id="tc_1", name="calculator", arguments={"a": 1, "b": 2})
        tool_response = AssistantMessage(content="", tool_calls=[tc])
        final_response = AssistantMessage(content="Result is 3")
        mock_provider.invoke = AsyncMock(side_effect=[tool_response, final_response])

        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=calculator_tool,
            max_iterations=5,
        )
        response = await agent.execute_query_async("add 1+2")
        assert response.content == "Result is 3"
        # History should contain ToolMessage
        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        assert len(tool_msgs) >= 1

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_error_result(self, mock_provider):
        """When LLM calls a tool that is not registered, ToolResult has ERROR status."""
        tc = ToolCall(id="tc_1", name="nonexistent", arguments={})
        tool_response = AssistantMessage(content="", tool_calls=[tc])
        final_response = AssistantMessage(content="Done")
        mock_provider.invoke = AsyncMock(side_effect=[tool_response, final_response])

        agent = BaseAgent(system_message="x", provider=mock_provider, max_iterations=5)
        await agent.execute_query_async("do something")
        # Should still complete -- the error is captured in ToolResult
        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        assert len(tool_msgs) >= 1
        error_results = [
            r
            for tm in tool_msgs
            for r in tm.tool_results
            if r.status == ToolStatus.ERROR
        ]
        assert len(error_results) >= 1

    @pytest.mark.asyncio
    async def test_tool_exception_captured_in_result(
        self, mock_provider, failing_tool_class
    ):
        """When a tool's execute() raises, ToolResult captures the error."""
        failing_tool = failing_tool_class()
        tc = ToolCall(id="tc_1", name="failing_tool", arguments={"input_value": "test"})
        tool_response = AssistantMessage(content="", tool_calls=[tc])
        final_response = AssistantMessage(content="Error handled")
        mock_provider.invoke = AsyncMock(side_effect=[tool_response, final_response])

        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=failing_tool,
            max_iterations=5,
        )
        await agent.execute_query_async("fail please")
        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        error_results = [
            r
            for tm in tool_msgs
            for r in tm.tool_results
            if r.status == ToolStatus.ERROR
        ]
        assert len(error_results) >= 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_executed_in_parallel(
        self, mock_provider, calculator_tool
    ):
        """Multiple tool_calls in one message are all executed."""
        tc1 = ToolCall(id="tc_1", name="calculator", arguments={"a": 1, "b": 2})
        tc2 = ToolCall(id="tc_2", name="calculator", arguments={"a": 3, "b": 4})
        tool_response = AssistantMessage(content="", tool_calls=[tc1, tc2])
        final_response = AssistantMessage(content="Both done")
        mock_provider.invoke = AsyncMock(side_effect=[tool_response, final_response])

        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=calculator_tool,
            max_iterations=5,
        )
        await agent.execute_query_async("compute")
        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        assert len(tool_msgs) >= 1
        # The tool message should have results for both calls
        all_results = [r for tm in tool_msgs for r in tm.tool_results]
        assert len(all_results) >= 2


# ---------------------------------------------------------------------------
# exit_on_success
# ---------------------------------------------------------------------------


class TestExitOnSuccess:
    """Tests for _exit_on_success and _should_exit_on_success."""

    def test_empty_set_always_false(self, mock_provider):
        """_should_exit_on_success returns False when set is empty."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        results = [
            ToolResult(
                tool_name="calc",
                tool_call_id="tc_1",
                result=5,
                status=ToolStatus.SUCCESS,
            )
        ]
        assert agent._should_exit_on_success(results) is False

    def test_tool_in_set_returns_true(self, mock_provider):
        """Returns True when called tool is in exit_on_success set."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            exit_on_success=["calc"],
        )
        results = [
            ToolResult(
                tool_name="calc",
                tool_call_id="tc_1",
                result=5,
                status=ToolStatus.SUCCESS,
            )
        ]
        assert agent._should_exit_on_success(results) is True

    def test_tool_not_in_set_returns_false(self, mock_provider):
        """Returns False when called tool is not in exit_on_success set."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            exit_on_success=["search"],
        )
        results = [
            ToolResult(
                tool_name="calc",
                tool_call_id="tc_1",
                result=5,
                status=ToolStatus.SUCCESS,
            )
        ]
        assert agent._should_exit_on_success(results) is False

    @pytest.mark.asyncio
    async def test_exit_on_success_triggers_early_return(
        self, mock_provider, calculator_tool
    ):
        """When exit_on_success is satisfied, agent returns after tool execution."""
        tc = ToolCall(id="tc_1", name="calculator", arguments={"a": 1, "b": 2})
        tool_response = AssistantMessage(content="", tool_calls=[tc])
        mock_provider.invoke = AsyncMock(return_value=tool_response)

        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=calculator_tool,
            exit_on_success=["calculator"],
            max_iterations=5,
        )
        response = await agent.execute_query_async("add")
        # Provider should only be called once (no second LLM call)
        assert mock_provider.invoke.await_count == 1
        assert isinstance(response, AssistantResponse)


# ---------------------------------------------------------------------------
# _build_final_response
# ---------------------------------------------------------------------------


class TestBuildFinalResponse:
    """Tests for BaseAgent._build_final_response."""

    def test_content_from_assistant_message(self, mock_provider):
        """Uses assistant_msg.content when available."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="Final answer")
        response = agent._build_final_response(msg, [], None)
        assert response.content == "Final answer"

    def test_fallback_content_when_empty(self, mock_provider):
        """Uses 'Execution completed.' when content is empty."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="")
        response = agent._build_final_response(msg, [], None)
        assert response.content == "Execution completed."

    def test_empty_tool_results_become_none(self, mock_provider):
        """Empty tool_results list becomes None in response."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="ok")
        response = agent._build_final_response(msg, [], None)
        assert response.tool_results is None

    def test_tool_results_passed_through(self, mock_provider):
        """Non-empty tool_results are preserved in response."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="ok")
        results = [
            ToolResult(
                tool_name="calc",
                tool_call_id="tc_1",
                result=42,
                status=ToolStatus.SUCCESS,
            )
        ]
        response = agent._build_final_response(msg, results, None)
        assert response.tool_results == results

    def test_avoids_duplicate_in_history(self, mock_provider):
        """Does not append assistant_msg if already in history (identity check)."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="already there")
        agent.conversation_history.append(msg)
        history_len_before = len(agent.conversation_history)
        agent._build_final_response(msg, [], None)
        assert len(agent.conversation_history) == history_len_before

    def test_appends_to_history_if_not_present(self, mock_provider):
        """Appends assistant_msg to history when not already there."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="new")
        history_len_before = len(agent.conversation_history)
        agent._build_final_response(msg, [], None)
        assert len(agent.conversation_history) == history_len_before + 1
        assert agent.conversation_history[-1] is msg

    def test_error_passed_through(self, mock_provider):
        """execution_error is reflected in response.error."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="ok")
        response = agent._build_final_response(msg, [], "some error")
        assert response.error == "some error"

    def test_agent_name_is_class_name(self, mock_provider):
        """response.agent_name matches the agent class name."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        msg = AssistantMessage(content="ok")
        response = agent._build_final_response(msg, [], None)
        assert response.agent_name == "BaseAgent"


# ---------------------------------------------------------------------------
# _collect_tool_results
# ---------------------------------------------------------------------------


class TestCollectToolResults:
    """Tests for BaseAgent._collect_tool_results."""

    def test_empty_history(self, mock_provider):
        """Empty history returns empty list."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.conversation_history = []
        assert agent._collect_tool_results() == []

    def test_history_with_tool_message(self, mock_provider):
        """Extracts results from ToolMessages in history."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        tr = ToolResult(
            tool_name="calc",
            tool_call_id="tc_1",
            result=5,
            status=ToolStatus.SUCCESS,
        )
        tm = ToolMessage(tool_results=[tr])
        agent.conversation_history.append(tm)
        results = agent._collect_tool_results()
        assert len(results) == 1
        assert results[0].tool_name == "calc"

    def test_history_without_tool_message(self, mock_provider):
        """History with no ToolMessage returns empty list."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.conversation_history.append(HumanMessage(content="hello"))
        assert agent._collect_tool_results() == []

    def test_custom_history_parameter(self, mock_provider):
        """Can pass a custom history list to scan."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        tr = ToolResult(
            tool_name="search",
            tool_call_id="tc_2",
            result="found",
            status=ToolStatus.SUCCESS,
        )
        custom_history = [ToolMessage(tool_results=[tr])]
        results = agent._collect_tool_results(history=custom_history)
        assert len(results) == 1
        assert results[0].tool_name == "search"


# ---------------------------------------------------------------------------
# _async_execute_tool
# ---------------------------------------------------------------------------


class TestAsyncExecuteTool:
    """Tests for BaseAgent._async_execute_tool."""

    @pytest.mark.asyncio
    async def test_tool_found_and_executed(self, mock_provider, calculator_tool):
        """When tool is registered, execute returns a ToolResult."""
        agent = BaseAgent(
            system_message="x", provider=mock_provider, tools=calculator_tool
        )
        tc = ToolCall(id="tc_1", name="calculator", arguments={"a": 1, "b": 2})
        result = await agent._async_execute_tool(tc)
        assert result is not None
        assert result.status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_tool_not_found_returns_none(self, mock_provider):
        """When tool is not registered, returns None."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        tc = ToolCall(id="tc_1", name="nonexistent", arguments={})
        result = await agent._async_execute_tool(tc)
        assert result is None

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_result(
        self, mock_provider, failing_tool_class
    ):
        """When tool.execute() raises, returns ToolResult with error."""
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tools=failing_tool_class(),
        )
        tc = ToolCall(id="tc_1", name="failing_tool", arguments={"input_value": "test"})
        result = await agent._async_execute_tool(tc)
        assert result is not None
        assert result.status == ToolStatus.ERROR
        # The @tool decorator catches exceptions internally and returns str(e) as the
        # error message — it never re-raises, so _async_execute_tool's own except
        # block is never reached and the "Tool execution error: " prefix is not added.
        assert "Intentional failure for testing" in result.error


# ---------------------------------------------------------------------------
# clear_conversation_history
# ---------------------------------------------------------------------------


class TestClearConversationHistory:
    """Tests for BaseAgent.clear_conversation_history."""

    def test_keep_system_message_true(self, mock_provider):
        """keep_system_message=True (default) retains only system_message."""
        agent = BaseAgent(system_message="sys", provider=mock_provider)
        agent.conversation_history.append(HumanMessage(content="hello"))
        agent.clear_conversation_history()
        assert len(agent.conversation_history) == 1
        assert agent.conversation_history[0] is agent.system_message

    def test_keep_system_message_false(self, mock_provider):
        """keep_system_message=False empties history completely."""
        agent = BaseAgent(system_message="sys", provider=mock_provider)
        agent.clear_conversation_history(keep_system_message=False)
        assert agent.conversation_history == []

    def test_default_is_keep(self, mock_provider):
        """Default behavior keeps the system message."""
        agent = BaseAgent(system_message="sys", provider=mock_provider)
        agent.conversation_history.append(HumanMessage(content="hello"))
        agent.clear_conversation_history()
        assert len(agent.conversation_history) == 1


# ---------------------------------------------------------------------------
# register_agent
# ---------------------------------------------------------------------------


class TestRegisterAgent:
    """Tests for BaseAgent.register_agent (sub-agent registration)."""

    def test_creates_subagent_wrapper_in_tools(self, mock_provider):
        """register_agent appends a SubAgentWrapper to registered_tools."""
        from obelix.core.agent.subagent_wrapper import SubAgentWrapper

        parent = BaseAgent(system_message="parent", provider=mock_provider)
        child = BaseAgent(system_message="child", provider=mock_provider)
        parent.register_agent(child, name="helper", description="Helps with stuff")
        assert len(parent.registered_tools) == 1
        assert isinstance(parent.registered_tools[0], SubAgentWrapper)

    def test_wrapper_has_correct_name(self, mock_provider):
        """SubAgentWrapper.tool_name matches the given name."""
        parent = BaseAgent(system_message="parent", provider=mock_provider)
        child = BaseAgent(system_message="child", provider=mock_provider)
        parent.register_agent(child, name="analyzer", description="Analyzes data")
        assert parent.registered_tools[0].tool_name == "analyzer"

    def test_wrapper_has_correct_description(self, mock_provider):
        """SubAgentWrapper.tool_description matches the given description."""
        parent = BaseAgent(system_message="parent", provider=mock_provider)
        child = BaseAgent(system_message="child", provider=mock_provider)
        parent.register_agent(child, name="writer", description="Writes text")
        assert parent.registered_tools[0].tool_description == "Writes text"

    def test_stateless_parameter_passed(self, mock_provider):
        """stateless parameter is propagated to SubAgentWrapper."""
        parent = BaseAgent(system_message="parent", provider=mock_provider)
        child = BaseAgent(system_message="child", provider=mock_provider)
        parent.register_agent(child, name="worker", description="Works", stateless=True)
        assert parent.registered_tools[0]._stateless is True


# ---------------------------------------------------------------------------
# Shared memory hooks (no-op when memory_graph=None)
# ---------------------------------------------------------------------------


class TestSharedMemoryNoOp:
    """Tests for shared memory hooks when memory_graph is None."""

    @pytest.mark.asyncio
    async def test_inject_shared_memory_noop_without_graph(self, mock_provider):
        """_inject_shared_memory does nothing when memory_graph is None."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        assert agent.memory_graph is None
        history_len = len(agent.conversation_history)
        status = MagicMock()
        status.agent = agent
        agent._inject_shared_memory(status)
        assert len(agent.conversation_history) == history_len

    @pytest.mark.asyncio
    async def test_publish_to_memory_noop_without_graph(self, mock_provider):
        """_publish_to_memory does nothing when memory_graph is None."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        assert agent.memory_graph is None
        status = MagicMock()
        status.assistant_message = AssistantMessage(content="test")
        # Should not raise
        await agent._publish_to_memory(status)

    @pytest.mark.asyncio
    async def test_inject_shared_memory_noop_without_agent_id(self, mock_provider):
        """_inject_shared_memory does nothing when agent_id is None."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        agent.memory_graph = MagicMock()
        agent.agent_id = None
        history_len = len(agent.conversation_history)
        status = MagicMock()
        status.agent = agent
        agent._inject_shared_memory(status)
        assert len(agent.conversation_history) == history_len


# ---------------------------------------------------------------------------
# tool_policy
# ---------------------------------------------------------------------------


class TestToolPolicy:
    """Tests for tool_policy enforcement via BEFORE_FINAL_RESPONSE hooks."""

    @pytest.mark.asyncio
    async def test_policy_retry_when_tool_not_called(self, mock_provider):
        """tool_policy causes RETRY when required tool not called yet."""
        call_count = 0

        async def invoke_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AssistantMessage(content="I can answer directly")
            # Second call: simulate tool usage
            return AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="tc_1",
                        name="calculator",
                        arguments={"a": 1, "b": 2},
                    )
                ],
            )

        mock_provider.invoke = AsyncMock(side_effect=invoke_side_effect)

        policy = [ToolRequirement(tool_name="calculator", min_calls=1)]
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tool_policy=policy,
            max_iterations=3,
        )
        # The agent will RETRY on the first response (no tool call), then get
        # tool_calls on the second response, but we won't register the tool so it
        # will error. Still, the point is that RETRY happened (call_count >= 2).
        try:
            await agent.execute_query_async("compute 1+2")
        except RuntimeError:
            pass  # May fail at final response due to still no successful tool call
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_policy_fail_at_last_iteration(self, mock_provider):
        """tool_policy causes FAIL at last iteration when tool never called."""
        mock_provider.invoke = AsyncMock(
            return_value=AssistantMessage(content="no tool")
        )
        policy = [ToolRequirement(tool_name="calculator", min_calls=1)]
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            tool_policy=policy,
            max_iterations=1,
        )
        with pytest.raises(RuntimeError, match="Required tool call missing"):
            await agent.execute_query_async("compute")

    def test_policy_hooks_registered(self, mock_provider):
        """tool_policy causes BEFORE_FINAL_RESPONSE hooks to be registered."""
        policy = [ToolRequirement(tool_name="calc", min_calls=1)]
        agent = BaseAgent(
            system_message="x", provider=mock_provider, tool_policy=policy
        )
        # Should have memory hooks + 2 policy hooks on BEFORE_FINAL_RESPONSE
        assert len(agent._hooks[AgentEvent.BEFORE_FINAL_RESPONSE]) >= 3


# ---------------------------------------------------------------------------
# Max iterations (timeout)
# ---------------------------------------------------------------------------


class TestMaxIterations:
    """Tests for max_iterations exhaustion behavior."""

    @pytest.mark.asyncio
    async def test_max_iterations_reached_returns_timeout_response(self, mock_provider):
        """When max_iterations exhausted, _build_timeout_response is called."""
        tc = ToolCall(id="tc_1", name="calc", arguments={})
        mock_provider.invoke = AsyncMock(
            return_value=AssistantMessage(content="", tool_calls=[tc])
        )
        agent = BaseAgent(system_message="x", provider=mock_provider, max_iterations=1)
        result = await agent.execute_query_async("loop forever")
        assert "stopped after 1 iterations" in result.content.lower()
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_max_iterations_reached_returns_timeout_with_tools(
        self, mock_provider, calculator_tool_class
    ):
        """When max_iterations exhausted with successful tools, returns timeout response.

        Previously this triggered a NameError (warning_msg undefined), now fixed.
        """
        tc = ToolCall(id="tc_1", name="calculator", arguments={"a": 1, "b": 2})
        mock_provider.invoke = AsyncMock(
            return_value=AssistantMessage(content="", tool_calls=[tc])
        )
        agent = BaseAgent(
            system_message="x",
            provider=mock_provider,
            max_iterations=1,
            tools=calculator_tool_class(),
        )
        result = await agent.execute_query_async("loop")
        assert "stopped after 1 iterations" in result.content.lower()
        assert result.error is not None


# ---------------------------------------------------------------------------
# get_conversation_history property
# ---------------------------------------------------------------------------


class TestGetConversationHistory:
    """Tests for the get_conversation_history property."""

    def test_returns_copy(self, mock_provider):
        """get_conversation_history returns a copy, not the original list."""
        agent = BaseAgent(system_message="x", provider=mock_provider)
        history = agent.get_conversation_history
        assert history is not agent.conversation_history
        assert history == agent.conversation_history
