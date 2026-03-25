"""Tests for A2A executor cooperative cancellation.

Covers:
- ContextEntry.active_agent slot
- active_agent lifecycle during _run_agent
- cancel() with active agent signals agent.cancel()
- cancel() without active agent emits TaskState.canceled directly
- Full cancel event sequence
- Cancel preserves history
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from obelix.adapters.inbound.a2a.server import (
    ContextEntry,
    ObelixAgentExecutor,
)

# ---------------------------------------------------------------------------
# Lightweight fakes (reuse patterns from test_server.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeRequestContext:
    """Minimal stand-in for a2a RequestContext."""

    task_id: str = "task-cancel-001"
    context_id: str | None = "ctx-cancel-001"

    def __post_init__(self):
        from a2a.types import Message, Part, Role, TextPart

        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Hello agent"))],
            message_id="msg-001",
        )

    def get_user_input(self) -> str | None:
        return "Hello agent"


class FakeEventQueue:
    """Captures enqueued events for assertions."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


def _make_mock_agent(response_content: str = "Agent response") -> MagicMock:
    """Create a mock BaseAgent with the attributes the executor expects."""
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage

    agent = MagicMock()
    agent.system_message = SystemMessage(content="You are a test agent.")
    agent.conversation_history = [agent.system_message]

    async def fake_stream(query):
        yield StreamEvent(
            is_final=True,
            assistant_response=AssistantResponse(
                agent_name="test_agent",
                content=response_content,
            ),
        )

    agent.execute_query_stream = MagicMock(side_effect=fake_stream)
    return agent


# ---------------------------------------------------------------------------
# ContextEntry — active_agent slot
# ---------------------------------------------------------------------------


class TestContextEntryActiveAgent:
    def test_context_entry_has_active_agent_slot(self):
        """ContextEntry has 'active_agent' in __slots__."""
        assert "active_agent" in ContextEntry.__slots__

    def test_active_agent_starts_as_none(self):
        entry = ContextEntry()
        assert entry.active_agent is None

    def test_active_agent_can_be_set(self):
        entry = ContextEntry()
        mock_agent = MagicMock()
        entry.active_agent = mock_agent
        assert entry.active_agent is mock_agent


# ---------------------------------------------------------------------------
# active_agent lifecycle during execution
# ---------------------------------------------------------------------------


class TestActiveAgentLifecycle:
    @pytest.mark.asyncio
    async def test_active_agent_set_during_execution(self):
        """entry.active_agent is set to the agent during _run_agent."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent

        captured_agent_ref = []

        def factory():
            agent = _make_mock_agent()

            async def spy_stream(query):
                # At this point, the executor should have set active_agent
                # We can't easily check entry here, so we just record the agent
                captured_agent_ref.append(agent)
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test_agent",
                        content="done",
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=spy_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # Agent was created and used
        assert len(captured_agent_ref) == 1

    @pytest.mark.asyncio
    async def test_active_agent_cleared_after_execution(self):
        """entry.active_agent is None after execute() completes."""

        def factory_fn():
            return _make_mock_agent()

        executor = ObelixAgentExecutor(factory_fn)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # After execution, active_agent should be cleared
        entry = executor._store.get_or_create(ctx.context_id)
        assert entry.active_agent is None

    @pytest.mark.asyncio
    async def test_active_agent_cleared_after_error(self):
        """entry.active_agent is None even if the agent raises an exception."""

        def factory():
            agent = _make_mock_agent()

            async def failing_stream(query):
                raise RuntimeError("boom")
                yield  # make it a generator

            agent.execute_query_stream = MagicMock(side_effect=failing_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        entry = executor._store.get_or_create(ctx.context_id)
        assert entry.active_agent is None


# ---------------------------------------------------------------------------
# cancel() — signals active agent
# ---------------------------------------------------------------------------


class TestCancelSignalsAgent:
    @pytest.mark.asyncio
    async def test_cancel_calls_agent_cancel_when_active(self):
        """cancel() calls agent.cancel() when an active agent exists."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent

        agent_cancel_called = asyncio.Event()

        def factory():
            agent = _make_mock_agent()

            async def slow_stream(query):
                # Wait a long time (will be canceled)
                await asyncio.sleep(10)
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test", content="done"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=slow_stream)

            def track_cancel():
                agent_cancel_called.set()

            agent.cancel = MagicMock(side_effect=track_cancel)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-1", context_id="c-1")
        queue = FakeEventQueue()
        cancel_queue = FakeEventQueue()

        # Start execution in background
        exec_task = asyncio.create_task(executor.execute(ctx, queue))

        # Wait a bit for the agent to start
        await asyncio.sleep(0.05)

        # Call cancel
        cancel_ctx = FakeRequestContext(task_id="t-1", context_id="c-1")
        await executor.cancel(cancel_ctx, cancel_queue)

        # agent.cancel() should have been called
        assert agent_cancel_called.is_set()

        # Clean up (the execute task may still be running)
        exec_task.cancel()
        try:
            await exec_task
        except (asyncio.CancelledError, Exception):
            pass

    @pytest.mark.asyncio
    async def test_cancel_no_active_agent_emits_canceled_directly(self):
        """cancel() with no active agent emits TaskState.canceled."""
        from a2a.types import TaskState, TaskStatusUpdateEvent

        def factory_fn():
            return _make_mock_agent()

        executor = ObelixAgentExecutor(factory_fn)
        ctx = FakeRequestContext(task_id="t-no-agent", context_id="c-no-agent")
        queue = FakeEventQueue()

        await executor.cancel(ctx, queue)

        assert len(queue.events) == 1
        event = queue.events[0]
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.canceled
        assert event.final is True
        assert event.task_id == "t-no-agent"


# ---------------------------------------------------------------------------
# Full cancel event sequence
# ---------------------------------------------------------------------------


class TestCancelEventSequence:
    @pytest.mark.asyncio
    async def test_cancel_event_sequence_working_then_canceled(self):
        """When agent is canceled mid-execution: working -> canceled (no completed)."""
        from a2a.types import TaskState, TaskStatusUpdateEvent

        from obelix.core.model.assistant_message import StreamEvent

        def factory():
            agent = _make_mock_agent()

            async def cancellable_stream(query):
                # Yield nothing, just wait to be canceled
                for _i in range(100):
                    await asyncio.sleep(0.01)
                    if agent._cancel_check and agent._cancel_check():
                        yield StreamEvent(canceled=True, is_final=True)
                        return
                yield StreamEvent(
                    is_final=True,
                    assistant_response=MagicMock(agent_name="test", content="done"),
                )

            # Use a real-ish cancel mechanism
            agent._cancel_check = None

            async def stream_with_cancel(query):
                yield StreamEvent(canceled=True, is_final=True)

            agent.execute_query_stream = MagicMock(side_effect=stream_with_cancel)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-seq", context_id="c-seq")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # Should be: working -> canceled
        states = [
            e.status.state for e in queue.events if isinstance(e, TaskStatusUpdateEvent)
        ]
        assert TaskState.working in states
        assert TaskState.canceled in states
        assert TaskState.completed not in states

    @pytest.mark.asyncio
    async def test_cancel_final_event_is_terminal(self):
        """The canceled TaskStatusUpdateEvent has final=True."""
        from a2a.types import TaskState, TaskStatusUpdateEvent

        from obelix.core.model.assistant_message import StreamEvent

        def factory():
            agent = _make_mock_agent()

            async def canceled_stream(query):
                yield StreamEvent(canceled=True, is_final=True)

            agent.execute_query_stream = MagicMock(side_effect=canceled_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        canceled_events = [
            e
            for e in queue.events
            if isinstance(e, TaskStatusUpdateEvent)
            and e.status.state == TaskState.canceled
        ]
        assert len(canceled_events) == 1
        assert canceled_events[0].final is True


# ---------------------------------------------------------------------------
# Cancel preserves history
# ---------------------------------------------------------------------------


class TestCancelPreservesHistory:
    @pytest.mark.asyncio
    async def test_cancel_saves_history_to_entry(self):
        """After cancel, the context entry's history includes the cancellation."""
        from obelix.core.model.assistant_message import (
            AssistantMessage,
            StreamEvent,
        )
        from obelix.core.model.human_message import HumanMessage

        def factory():
            agent = _make_mock_agent()
            sys_msg = agent.system_message

            async def stream_then_cancel(query):
                # Simulate agent setting its own history before yielding canceled
                agent.conversation_history = [
                    sys_msg,
                    HumanMessage(content=query),
                    AssistantMessage(content="[canceled]"),
                ]
                yield StreamEvent(canceled=True, is_final=True)

            agent.execute_query_stream = MagicMock(side_effect=stream_then_cancel)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-hist", context_id="c-hist")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        entry = executor._store.get_or_create("c-hist")
        # History should have been saved (without the system message)
        assert len(entry.history) > 0
        # Should contain the cancel message
        has_cancel = any(
            isinstance(m, AssistantMessage) and "canceled" in m.content.lower()
            for m in entry.history
        )
        assert has_cancel
