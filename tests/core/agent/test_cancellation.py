"""Tests for BaseAgent cooperative cancellation.

Covers:
- cancel() sets the event
- _is_canceled() default and after cancel
- Cancel before execution (Point A)
- Cancel between iterations (Point A, multi-iteration)
- Cancel during streaming (Point B)
- Cancel during tool dispatch (Point C)
- Cancellation message content and history preservation
- StreamEvent.canceled field
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import (
    AssistantMessage,
    StreamEvent,
)
from obelix.core.model.tool_message import ToolCall, ToolMessage, ToolStatus


def _make_provider_with_invoke(invoke_side_effect):
    """Create a MagicMock provider with invoke and invoke_stream (NotImplementedError)."""
    provider = MagicMock()
    provider.model_id = "mock"
    provider.provider_type = "mock"
    provider.invoke = AsyncMock(side_effect=invoke_side_effect)

    def _no_stream(*a, **kw):
        raise NotImplementedError

    provider.invoke_stream = MagicMock(side_effect=_no_stream)
    return provider


# ---------------------------------------------------------------------------
# StreamEvent.canceled field
# ---------------------------------------------------------------------------


class TestStreamEventCanceledField:
    def test_canceled_defaults_to_false(self):
        event = StreamEvent()
        assert event.canceled is False

    def test_canceled_true(self):
        event = StreamEvent(canceled=True, is_final=True)
        assert event.canceled is True
        assert event.is_final is True

    def test_canceled_event_has_no_token_or_response(self):
        event = StreamEvent(canceled=True, is_final=True)
        assert event.token is None
        assert event.assistant_message is None
        assert event.assistant_response is None


# ---------------------------------------------------------------------------
# cancel() and _is_canceled()
# ---------------------------------------------------------------------------


class TestCancelFlag:
    def test_is_canceled_default_false(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        assert agent._is_canceled() is False

    def test_cancel_sets_event(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()
        assert agent._cancel_event.is_set() is True

    def test_is_canceled_true_after_cancel(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()
        assert agent._is_canceled() is True


# ---------------------------------------------------------------------------
# Cancel before execution (Point A — before first LLM call)
# ---------------------------------------------------------------------------


class TestCancelBeforeExecution:
    @pytest.mark.asyncio
    async def test_cancel_before_execute_yields_canceled_event(self, mock_provider):
        """Canceling before execute_query_stream yields a canceled StreamEvent."""
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        events = []
        async for event in agent.execute_query_stream("hello"):
            events.append(event)

        assert len(events) == 1
        assert events[0].canceled is True
        assert events[0].is_final is True

    @pytest.mark.asyncio
    async def test_cancel_before_execute_provider_not_called(self, mock_provider):
        """When canceled before execution, the provider should never be invoked."""
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        async for _ in agent.execute_query_stream("hello"):
            pass

        mock_provider.invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_before_execute_history_has_cancel_message(
        self, mock_provider
    ):
        """After cancel, conversation_history contains the cancellation message."""
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        async for _ in agent.execute_query_stream("hello"):
            pass

        # History: SystemMessage + HumanMessage + AssistantMessage(cancel)
        assert len(agent.conversation_history) == 3
        cancel_msg = agent.conversation_history[-1]
        assert isinstance(cancel_msg, AssistantMessage)


# ---------------------------------------------------------------------------
# Cancel message content
# ---------------------------------------------------------------------------


class TestCancelMessageContent:
    @pytest.mark.asyncio
    async def test_cancel_message_contains_interrupted(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        async for _ in agent.execute_query_stream("hello"):
            pass

        cancel_msg = agent.conversation_history[-1]
        assert "interrupted" in cancel_msg.content.lower()

    @pytest.mark.asyncio
    async def test_cancel_message_contains_canceled(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        async for _ in agent.execute_query_stream("hello"):
            pass

        cancel_msg = agent.conversation_history[-1]
        assert "canceled" in cancel_msg.content.lower()

    @pytest.mark.asyncio
    async def test_cancel_message_mentions_user(self, mock_provider):
        agent = BaseAgent(system_message="test", provider=mock_provider)
        agent.cancel()

        async for _ in agent.execute_query_stream("hello"):
            pass

        cancel_msg = agent.conversation_history[-1]
        assert "user" in cancel_msg.content.lower()


# ---------------------------------------------------------------------------
# Cancel between iterations (Point A — agent does tool call then gets canceled)
# ---------------------------------------------------------------------------


class TestCancelBetweenIterations:
    @pytest.mark.asyncio
    async def test_cancel_after_first_iteration_stops_before_second_llm_call(self):
        """Agent with a tool call: cancel after first iteration prevents second LLM call."""
        from pydantic import Field

        from obelix.core.tool.tool_base import Tool
        from obelix.core.tool.tool_decorator import tool

        @tool(name="trigger_cancel", description="A tool that triggers cancel")
        class TriggerCancelTool(Tool):
            query: str = Field(..., description="Input")

            async def execute(self) -> dict:
                return {"ok": True}

        tool_instance = TriggerCancelTool()

        # First invoke returns tool call, second would return text
        tool_call = ToolCall(id="tc_1", name="trigger_cancel", arguments={"query": "x"})
        first_response = AssistantMessage(content="", tool_calls=[tool_call])
        second_response = AssistantMessage(content="Final answer")

        provider = MagicMock()
        provider.model_id = "mock"
        provider.provider_type = "mock"

        call_count = 0

        async def counting_invoke(history, tools, schema=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return first_response
            return second_response

        provider.invoke = AsyncMock(side_effect=counting_invoke)

        def _no_stream(*a, **kw):
            raise NotImplementedError

        provider.invoke_stream = MagicMock(side_effect=_no_stream)

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tools=tool_instance,
            max_iterations=5,
        )

        # Cancel after the first tool execution completes — the cancel
        # check at the top of the next iteration (Point A) will fire.
        # We hook into AFTER_TOOL_EXECUTION to set cancel at the right moment.
        from obelix.core.agent.hooks import AgentEvent as AE
        from obelix.core.agent.hooks import HookDecision

        def cancel_effect(status):
            agent.cancel()

        agent.on(AE.AFTER_TOOL_EXECUTION).handle(
            HookDecision.CONTINUE, effects=[cancel_effect]
        )

        events = []
        async for event in agent.execute_query_stream("do something"):
            events.append(event)

        # Should have been canceled after first iteration, so only 1 LLM call
        assert call_count == 1
        # Final event should be canceled
        assert events[-1].canceled is True
        assert events[-1].is_final is True


# ---------------------------------------------------------------------------
# Cancel during streaming (Point B — mid-stream)
# ---------------------------------------------------------------------------


class TestCancelDuringStreaming:
    @pytest.mark.asyncio
    async def test_cancel_mid_stream_yields_canceled_event(self):
        """Cancel during streaming closes the stream and yields canceled event."""
        provider = MagicMock()
        provider.model_id = "mock"
        provider.provider_type = "mock"

        token_count = 0

        async def fake_stream(history, tools, schema=None):
            nonlocal token_count
            for i in range(10):
                token_count += 1
                yield StreamEvent(token=f"token_{i}")
                # Simulate a small delay so cancel can be detected
                await asyncio.sleep(0)
            yield StreamEvent(
                is_final=True,
                assistant_message=AssistantMessage(content="full response"),
            )

        provider.invoke_stream = MagicMock(side_effect=fake_stream)

        agent = BaseAgent(system_message="test", provider=provider, max_iterations=5)

        collected_events = []

        async for event in agent.execute_query_stream("hello"):
            collected_events.append(event)
            # Cancel after receiving 3 tokens
            if len([e for e in collected_events if e.token]) >= 3:
                agent.cancel()

        # The last event should be canceled
        final = collected_events[-1]
        assert final.canceled is True
        assert final.is_final is True

        # We should have received some tokens but not all 10
        token_events = [e for e in collected_events if e.token]
        assert len(token_events) >= 3  # at least the 3 we waited for
        assert len(token_events) < 10  # but not all of them

    @pytest.mark.asyncio
    async def test_cancel_mid_stream_history_preserved(self):
        """After cancel during streaming, history contains cancel message."""
        provider = MagicMock()
        provider.model_id = "mock"
        provider.provider_type = "mock"

        async def fake_stream(history, tools, schema=None):
            for i in range(5):
                yield StreamEvent(token=f"tok_{i}")
                await asyncio.sleep(0)
            yield StreamEvent(
                is_final=True,
                assistant_message=AssistantMessage(content="done"),
            )

        provider.invoke_stream = MagicMock(side_effect=fake_stream)

        agent = BaseAgent(system_message="test", provider=provider, max_iterations=5)

        async for event in agent.execute_query_stream("hello"):
            if event.token:
                agent.cancel()

        # History should end with cancellation AssistantMessage
        cancel_msg = agent.conversation_history[-1]
        assert isinstance(cancel_msg, AssistantMessage)
        assert "interrupted" in cancel_msg.content.lower()
        assert (
            "streaming" in cancel_msg.content.lower()
            or "generated" in cancel_msg.content.lower()
        )


# ---------------------------------------------------------------------------
# Cancel during tool dispatch (Point C)
# ---------------------------------------------------------------------------


class TestCancelDuringToolDispatch:
    @pytest.mark.asyncio
    async def test_cancel_during_slow_tool(self):
        """Canceling during a slow tool yields canceled event with tool results."""
        from pydantic import Field

        from obelix.core.tool.tool_base import Tool
        from obelix.core.tool.tool_decorator import tool

        @tool(name="slow_tool", description="A slow tool")
        class SlowTool(Tool):
            query: str = Field(..., description="Input")

            async def execute(self) -> dict:
                await asyncio.sleep(10)  # very slow
                return {"done": True}

        tool_instance = SlowTool()

        tool_call = ToolCall(id="tc_slow", name="slow_tool", arguments={"query": "x"})
        provider = _make_provider_with_invoke(
            lambda *a, **kw: AssistantMessage(content="", tool_calls=[tool_call])
        )

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tools=tool_instance,
            max_iterations=5,
        )

        # Cancel after a short delay (tool is sleeping for 10s)
        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            agent.cancel()

        cancel_task = asyncio.create_task(cancel_after_delay())

        events = []
        async for event in agent.execute_query_stream("do something"):
            events.append(event)

        await cancel_task

        # Final event should be canceled
        final = events[-1]
        assert final.canceled is True
        assert final.is_final is True

    @pytest.mark.asyncio
    async def test_cancel_during_tool_dispatch_tool_results_in_history(self):
        """After cancel during tools, history contains assistant_msg + tool results."""
        from pydantic import Field

        from obelix.core.tool.tool_base import Tool
        from obelix.core.tool.tool_decorator import tool

        @tool(name="slow_tool_2", description="Another slow tool")
        class SlowTool2(Tool):
            query: str = Field(..., description="Input")

            async def execute(self) -> dict:
                await asyncio.sleep(10)
                return {"done": True}

        tool_instance = SlowTool2()

        tool_call = ToolCall(
            id="tc_slow2", name="slow_tool_2", arguments={"query": "x"}
        )
        provider = _make_provider_with_invoke(
            lambda *a, **kw: AssistantMessage(content="", tool_calls=[tool_call])
        )

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tools=tool_instance,
            max_iterations=5,
        )

        async def cancel_after_delay():
            await asyncio.sleep(0.1)
            agent.cancel()

        cancel_task = asyncio.create_task(cancel_after_delay())

        async for _ in agent.execute_query_stream("do something"):
            pass

        await cancel_task

        # History should contain:
        # SystemMessage + HumanMessage + AssistantMessage(tool_calls)
        # + ToolMessage(results from _process_tool_calls)
        # + AssistantMessage(cancel message)
        assert len(agent.conversation_history) >= 5

        # At least one ToolMessage should contain the canceled tool result
        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        assert len(tool_msgs) >= 1
        # Find the tool result for slow_tool_2
        all_results = [r for tm in tool_msgs for r in tm.tool_results]
        slow_results = [r for r in all_results if r.tool_name == "slow_tool_2"]
        assert len(slow_results) >= 1
        tool_result = slow_results[0]

        # The canceled tool should have an error about cancellation
        assert tool_result.status == ToolStatus.ERROR
        assert "canceled" in tool_result.error.lower()

        # Last message should be the cancel AssistantMessage
        cancel_msg = agent.conversation_history[-1]
        assert isinstance(cancel_msg, AssistantMessage)
        assert "interrupted" in cancel_msg.content.lower()

    @pytest.mark.asyncio
    async def test_canceled_tool_result_mentions_tool_name(self):
        """CancelledError on tool produces ToolResult that names the tool."""
        from pydantic import Field

        from obelix.core.tool.tool_base import Tool
        from obelix.core.tool.tool_decorator import tool

        @tool(name="named_tool", description="Tool with a name")
        class NamedTool(Tool):
            query: str = Field(..., description="Input")

            async def execute(self) -> dict:
                await asyncio.sleep(10)
                return {"ok": True}

        tool_instance = NamedTool()

        tool_call = ToolCall(id="tc_named", name="named_tool", arguments={"query": "x"})
        provider = _make_provider_with_invoke(
            lambda *a, **kw: AssistantMessage(content="", tool_calls=[tool_call])
        )

        agent = BaseAgent(
            system_message="test",
            provider=provider,
            tools=tool_instance,
            max_iterations=5,
        )

        async def cancel_soon():
            await asyncio.sleep(0.05)
            agent.cancel()

        cancel_task = asyncio.create_task(cancel_soon())

        async for _ in agent.execute_query_stream("go"):
            pass

        await cancel_task

        tool_msgs = [
            m for m in agent.conversation_history if isinstance(m, ToolMessage)
        ]
        assert len(tool_msgs) >= 1
        # Find results for named_tool across all ToolMessages
        all_results = [r for tm in tool_msgs for r in tm.tool_results]
        named_results = [r for r in all_results if r.tool_name == "named_tool"]
        assert len(named_results) >= 1
        assert "named_tool" in named_results[0].error
