"""Tests for ObelixAgentExecutor — the bridge between a2a-sdk and Obelix BaseAgent.

Covers:
- Factory callable usage (fresh agent per request)
- Per-context conversation history (multi-turn)
- LRU eviction when max_contexts exceeded
- Per-context asyncio.Lock serialization
- Parallel execution across different contexts
- CancelledError handling
- Empty user input
- Successful execution event sequence (working -> artifact -> completed)
- Failed execution (agent raises)
- Cancel method
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from obelix.adapters.inbound.a2a.executor import (
    DEFAULT_MAX_CONTEXTS,
    ObelixAgentExecutor,
    _ContextEntry,
    _error_message,
)

# ---------------------------------------------------------------------------
# Lightweight fakes for a2a-sdk types (avoid importing the real SDK in tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeRequestContext:
    """Minimal stand-in for a2a RequestContext."""

    task_id: str = "task-001"
    context_id: str | None = "ctx-001"

    def get_user_input(self) -> str | None:
        return "Hello agent"


class FakeEventQueue:
    """Captures enqueued events for assertions."""

    def __init__(self) -> None:
        self.events: list = []

    async def enqueue_event(self, event) -> None:
        self.events.append(event)


def _make_mock_agent(response_content: str = "Agent response") -> MagicMock:
    """Create a mock BaseAgent with the attributes executor expects."""
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage

    agent = MagicMock()
    agent.system_message = SystemMessage(content="You are a test agent.")
    agent.conversation_history = [agent.system_message]

    # Mock execute_query_stream as an async generator (non-streaming fallback path)
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


def _make_factory(
    response_content: str = "Agent response",
) -> tuple[Callable[[], MagicMock], list[MagicMock]]:
    """Return (factory_callable, list_of_created_agents).

    The list accumulates every agent the factory creates, so tests can
    inspect how many agents were created and their state.
    """
    created: list[MagicMock] = []

    def factory() -> MagicMock:
        agent = _make_mock_agent(response_content)
        created.append(agent)
        return agent

    return factory, created


def _assert_error_message(status, expected_text: str) -> None:
    """Assert that a TaskStatus has a proper A2A Message with expected text."""
    from a2a.types import Message, Role

    assert status.message is not None
    assert isinstance(status.message, Message)
    assert status.message.role == Role.agent
    assert len(status.message.parts) == 1
    assert status.message.parts[0].root.text == expected_text


# ---------------------------------------------------------------------------
# _error_message helper
# ---------------------------------------------------------------------------


class TestErrorMessage:
    def test_returns_message_with_agent_role(self):
        from a2a.types import Message, Role

        msg = _error_message("some error")
        assert isinstance(msg, Message)
        assert msg.role == Role.agent

    def test_has_single_text_part_with_given_text(self):
        from a2a.types import TextPart

        msg = _error_message("specific error text")
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0].root, TextPart)
        assert msg.parts[0].root.text == "specific error text"

    def test_message_id_is_valid_uuid(self):
        import uuid

        msg = _error_message("test")
        # Should not raise
        parsed = uuid.UUID(msg.message_id)
        assert str(parsed) == msg.message_id


# ---------------------------------------------------------------------------
# _ContextEntry
# ---------------------------------------------------------------------------


class TestContextEntry:
    def test_initial_state(self):
        entry = _ContextEntry()
        assert entry.history == []
        assert isinstance(entry.lock, asyncio.Lock)

    def test_history_is_mutable_list(self):
        entry = _ContextEntry()
        entry.history.append("msg")
        assert len(entry.history) == 1


# ---------------------------------------------------------------------------
# ObelixAgentExecutor — construction
# ---------------------------------------------------------------------------


class TestExecutorConstruction:
    def test_stores_factory_callable(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        assert executor._agent_factory is factory

    def test_default_max_contexts(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        assert executor._max_contexts == DEFAULT_MAX_CONTEXTS

    def test_custom_max_contexts(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=5)
        assert executor._max_contexts == 5

    def test_empty_contexts_on_init(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        assert len(executor._contexts) == 0


# ---------------------------------------------------------------------------
# _get_or_create_context
# ---------------------------------------------------------------------------


class TestGetOrCreateContext:
    def test_creates_new_context(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        entry = executor._get_or_create_context("ctx-1")
        assert isinstance(entry, _ContextEntry)
        assert "ctx-1" in executor._contexts

    def test_returns_existing_context(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        entry1 = executor._get_or_create_context("ctx-1")
        entry2 = executor._get_or_create_context("ctx-1")
        assert entry1 is entry2

    def test_moves_existing_to_end(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        executor._get_or_create_context("ctx-1")
        executor._get_or_create_context("ctx-2")
        executor._get_or_create_context("ctx-1")  # should move to end
        keys = list(executor._contexts.keys())
        assert keys == ["ctx-2", "ctx-1"]

    def test_lru_eviction_at_capacity(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=2)
        executor._get_or_create_context("ctx-1")
        executor._get_or_create_context("ctx-2")
        executor._get_or_create_context("ctx-3")  # should evict ctx-1
        assert "ctx-1" not in executor._contexts
        assert "ctx-2" in executor._contexts
        assert "ctx-3" in executor._contexts

    def test_lru_eviction_preserves_recently_used(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=2)
        executor._get_or_create_context("ctx-1")
        executor._get_or_create_context("ctx-2")
        executor._get_or_create_context("ctx-1")  # refresh ctx-1
        executor._get_or_create_context("ctx-3")  # should evict ctx-2 (oldest)
        assert "ctx-1" in executor._contexts
        assert "ctx-2" not in executor._contexts
        assert "ctx-3" in executor._contexts

    def test_eviction_history_is_lost(self):
        """Once a context is evicted, its history is gone."""
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=1)
        entry = executor._get_or_create_context("ctx-1")
        entry.history.append("some message")
        executor._get_or_create_context("ctx-2")  # evicts ctx-1
        new_entry = executor._get_or_create_context("ctx-1")  # re-created fresh
        assert new_entry.history == []


# ---------------------------------------------------------------------------
# execute() — successful flow
# ---------------------------------------------------------------------------


class TestExecuteSuccess:
    @pytest.mark.asyncio
    async def test_creates_fresh_agent_via_factory(self):
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(created) == 1

    @pytest.mark.asyncio
    async def test_second_call_creates_second_agent(self):
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)
        await executor.execute(ctx, queue)

        assert len(created) == 2

    @pytest.mark.asyncio
    async def test_event_sequence_working_artifact_completed(self):
        """Successful execution emits: working -> artifact -> completed."""
        from a2a.types import (
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatusUpdateEvent,
        )

        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 3
        # Event 1: working
        assert isinstance(queue.events[0], TaskStatusUpdateEvent)
        assert queue.events[0].status.state == TaskState.working
        assert queue.events[0].final is False
        # Event 2: artifact (non-streaming fallback: append=False, last_chunk=True)
        assert isinstance(queue.events[1], TaskArtifactUpdateEvent)
        assert queue.events[1].artifact.parts[0].root.text == "Agent response"
        assert queue.events[1].append is False
        assert queue.events[1].last_chunk is True
        # Event 3: completed
        assert isinstance(queue.events[2], TaskStatusUpdateEvent)
        assert queue.events[2].status.state == TaskState.completed
        assert queue.events[2].final is True

    @pytest.mark.asyncio
    async def test_task_id_and_context_id_propagated(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-42", context_id="c-99")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        for event in queue.events:
            assert event.task_id == "t-42"
            assert event.context_id == "c-99"

    @pytest.mark.asyncio
    async def test_default_context_id_when_none(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(context_id=None)
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        for event in queue.events:
            assert event.context_id == "default"

    @pytest.mark.asyncio
    async def test_agent_receives_user_text(self):
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        created[0].execute_query_stream.assert_called_once_with("Hello agent")

    @pytest.mark.asyncio
    async def test_empty_response_content_yields_empty_string_artifact(self):
        factory, _ = _make_factory(response_content="")
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        from a2a.types import TaskArtifactUpdateEvent

        artifact_events = [
            e for e in queue.events if isinstance(e, TaskArtifactUpdateEvent)
        ]
        assert len(artifact_events) == 1
        assert artifact_events[0].artifact.parts[0].root.text == ""


# ---------------------------------------------------------------------------
# execute() — multi-turn conversation history
# ---------------------------------------------------------------------------


class TestMultiTurnHistory:
    @pytest.mark.asyncio
    async def test_first_request_no_history_injected(self):
        """First request on a context should NOT set conversation_history (empty history)."""
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        agent = created[0]
        # conversation_history should NOT have been reassigned (history was empty)
        # The mock's default conversation_history is [system_message]
        assert agent.conversation_history == [agent.system_message]

    @pytest.mark.asyncio
    async def test_second_request_gets_history_from_first(self):
        """Second request on same context_id gets history saved from first request."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
        from obelix.core.model.human_message import HumanMessage

        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            agent = _make_mock_agent(f"Response {call_count}")

            # Simulate the agent populating conversation_history during execution
            sys_msg = agent.system_message
            current_count = call_count

            async def fake_stream(query):
                agent.conversation_history = [
                    sys_msg,
                    HumanMessage(content=query),
                    MagicMock(content=f"Response {current_count}"),
                ]
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test_agent", content=f"Response {current_count}"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=fake_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(context_id="ctx-mt")
        queue = FakeEventQueue()

        # First request
        await executor.execute(ctx, queue)

        # Second request — new agent should get history injected
        ctx2 = FakeRequestContext(context_id="ctx-mt")
        ctx2.get_user_input = lambda: "Follow-up question"
        queue2 = FakeEventQueue()
        await executor.execute(ctx2, queue2)

        # The context should now have history from both turns
        entry = executor._contexts["ctx-mt"]
        assert len(entry.history) >= 2  # At least the messages from both turns

    @pytest.mark.asyncio
    async def test_different_contexts_have_independent_history(self):
        """Different context_ids maintain separate conversation histories."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
        from obelix.core.model.human_message import HumanMessage

        def factory():
            agent = _make_mock_agent()
            sys_msg = agent.system_message

            async def fake_stream(query):
                agent.conversation_history = [
                    sys_msg,
                    HumanMessage(content=query),
                    MagicMock(content="reply"),
                ]
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test", content="reply"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=fake_stream)
            return agent

        executor = ObelixAgentExecutor(factory)

        # Execute on ctx-A
        ctx_a = FakeRequestContext(context_id="ctx-A")
        await executor.execute(ctx_a, FakeEventQueue())

        # Execute on ctx-B
        ctx_b = FakeRequestContext(context_id="ctx-B")
        await executor.execute(ctx_b, FakeEventQueue())

        entry_a = executor._contexts["ctx-A"]
        entry_b = executor._contexts["ctx-B"]
        assert entry_a is not entry_b
        assert entry_a.history is not entry_b.history


# ---------------------------------------------------------------------------
# execute() — empty user input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    @pytest.mark.asyncio
    async def test_empty_input_emits_failed_state(self):
        from a2a.types import TaskState, TaskStatusUpdateEvent

        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        ctx.get_user_input = lambda: ""  # Empty input
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 1
        event = queue.events[0]
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.failed
        assert event.final is True
        _assert_error_message(event.status, "No user input provided")

    @pytest.mark.asyncio
    async def test_none_input_emits_failed_state(self):
        from a2a.types import TaskState

        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        ctx.get_user_input = lambda: None
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 1
        assert queue.events[0].status.state == TaskState.failed
        _assert_error_message(queue.events[0].status, "No user input provided")

    @pytest.mark.asyncio
    async def test_empty_input_does_not_create_agent(self):
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        ctx.get_user_input = lambda: ""
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(created) == 0


# ---------------------------------------------------------------------------
# execute() — agent failure
# ---------------------------------------------------------------------------


class TestExecuteFailure:
    @pytest.mark.asyncio
    async def test_agent_exception_emits_failed_state(self):
        from a2a.types import TaskState, TaskStatusUpdateEvent

        def factory():
            agent = _make_mock_agent()

            async def failing_stream(query):
                raise RuntimeError("LLM provider down")
                yield  # make it a generator

            agent.execute_query_stream = MagicMock(side_effect=failing_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # working + failed
        assert len(queue.events) == 2
        assert queue.events[0].status.state == TaskState.working
        failed_event = queue.events[1]
        assert isinstance(failed_event, TaskStatusUpdateEvent)
        assert failed_event.status.state == TaskState.failed
        assert failed_event.final is True
        _assert_error_message(
            failed_event.status, "Execution failed: LLM provider down"
        )

    @pytest.mark.asyncio
    async def test_agent_exception_includes_error_in_message(self):
        def factory():
            agent = _make_mock_agent()

            async def failing_stream(query):
                raise ValueError("Bad input format")
                yield  # make it a generator

            agent.execute_query_stream = MagicMock(side_effect=failing_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        failed_event = queue.events[-1]
        _assert_error_message(failed_event.status, "Execution failed: Bad input format")


# ---------------------------------------------------------------------------
# execute() — CancelledError
# ---------------------------------------------------------------------------


class TestCancelledError:
    @pytest.mark.asyncio
    async def test_cancelled_error_emits_canceled_state_and_reraises(self):
        from a2a.types import TaskState

        def factory():
            agent = _make_mock_agent()

            async def cancelled_stream(query):
                raise asyncio.CancelledError()
                yield  # make it a generator

            agent.execute_query_stream = MagicMock(side_effect=cancelled_stream)
            return agent

        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        with pytest.raises(asyncio.CancelledError):
            await executor.execute(ctx, queue)

        # working + canceled
        assert len(queue.events) == 2
        assert queue.events[1].status.state == TaskState.canceled
        assert queue.events[1].final is True
        _assert_error_message(queue.events[1].status, "Task canceled by client")


# ---------------------------------------------------------------------------
# Per-context locking — serialization
# ---------------------------------------------------------------------------


class TestPerContextLocking:
    @pytest.mark.asyncio
    async def test_same_context_requests_are_serialized(self):
        """Two concurrent requests on the same context_id should NOT overlap."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent

        execution_log: list[tuple[str, str]] = []

        def factory():
            agent = _make_mock_agent()

            async def slow_stream(query):
                execution_log.append(("start", query))
                await asyncio.sleep(0.05)
                execution_log.append(("end", query))
                agent.conversation_history = [agent.system_message]
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test", content="ok"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=slow_stream)
            return agent

        executor = ObelixAgentExecutor(factory)

        ctx1 = FakeRequestContext(context_id="same-ctx")
        ctx1.get_user_input = lambda: "request-1"
        ctx2 = FakeRequestContext(context_id="same-ctx")
        ctx2.get_user_input = lambda: "request-2"

        await asyncio.gather(
            executor.execute(ctx1, FakeEventQueue()),
            executor.execute(ctx2, FakeEventQueue()),
        )

        # With serialization: start-1, end-1, start-2, end-2
        # Without serialization: start-1, start-2, end-1, end-2
        starts = [i for i, (action, _) in enumerate(execution_log) if action == "start"]
        ends = [i for i, (action, _) in enumerate(execution_log) if action == "end"]
        # First end must come before second start (serialized)
        assert ends[0] < starts[1]

    @pytest.mark.asyncio
    async def test_different_contexts_can_run_in_parallel(self):
        """Two concurrent requests on different context_ids CAN overlap."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent

        execution_log: list[tuple[str, str]] = []

        def factory():
            agent = _make_mock_agent()

            async def slow_stream(query):
                execution_log.append(("start", query))
                await asyncio.sleep(0.05)
                execution_log.append(("end", query))
                agent.conversation_history = [agent.system_message]
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test", content="ok"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=slow_stream)
            return agent

        executor = ObelixAgentExecutor(factory)

        ctx1 = FakeRequestContext(context_id="ctx-A")
        ctx1.get_user_input = lambda: "request-A"
        ctx2 = FakeRequestContext(context_id="ctx-B")
        ctx2.get_user_input = lambda: "request-B"

        await asyncio.gather(
            executor.execute(ctx1, FakeEventQueue()),
            executor.execute(ctx2, FakeEventQueue()),
        )

        # With parallel execution: start-A, start-B, end-A, end-B (interleaved)
        # Both starts should happen before both ends
        start_indices = [
            i for i, (action, _) in enumerate(execution_log) if action == "start"
        ]
        end_indices = [
            i for i, (action, _) in enumerate(execution_log) if action == "end"
        ]
        assert start_indices[1] < end_indices[0]


# ---------------------------------------------------------------------------
# cancel() method
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_emits_canceled_state(self):
        from a2a.types import TaskState, TaskStatusUpdateEvent

        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-cancel", context_id="c-cancel")
        queue = FakeEventQueue()

        await executor.cancel(ctx, queue)

        assert len(queue.events) == 1
        event = queue.events[0]
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.canceled
        assert event.final is True
        assert event.task_id == "t-cancel"
        _assert_error_message(event.status, "Task canceled by client")


# ---------------------------------------------------------------------------
# LRU eviction under load
# ---------------------------------------------------------------------------


class TestLRUEvictionIntegration:
    @pytest.mark.asyncio
    async def test_lru_eviction_during_execute(self):
        """Execute on 3 contexts with max_contexts=2, oldest is evicted."""
        from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
        from obelix.core.model.human_message import HumanMessage

        def factory():
            agent = _make_mock_agent()
            sys_msg = agent.system_message

            async def fake_stream(query):
                agent.conversation_history = [
                    sys_msg,
                    HumanMessage(content=query),
                    MagicMock(content="reply"),
                ]
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test", content="reply"
                    ),
                )

            agent.execute_query_stream = MagicMock(side_effect=fake_stream)
            return agent

        executor = ObelixAgentExecutor(factory, max_contexts=2)

        for cid in ["ctx-1", "ctx-2", "ctx-3"]:
            ctx = FakeRequestContext(context_id=cid)
            await executor.execute(ctx, FakeEventQueue())

        assert "ctx-1" not in executor._contexts
        assert "ctx-2" in executor._contexts
        assert "ctx-3" in executor._contexts
        assert len(executor._contexts) == 2
