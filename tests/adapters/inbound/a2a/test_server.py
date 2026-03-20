"""Tests for ObelixAgentExecutor — the bridge between a2a-sdk and Obelix BaseAgent.

Covers:
- Factory callable usage (fresh agent per request)
- Per-context conversation history (multi-turn)
- LRU eviction when max_contexts exceeded
- Per-context idle gate serialization
- Parallel execution across different contexts
- CancelledError handling
- Empty user input
- Successful execution event sequence (working -> artifact -> completed)
- Failed execution (agent raises)
- Cancel method
- Deferred tool (input-required) flow with DataPart
- Resume after deferred with DataPart
- Multi-part inbound (attachments)
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from obelix.adapters.inbound.a2a.server import (
    DEFAULT_MAX_CONTEXTS,
    ContextEntry,
    ObelixAgentExecutor,
    agent_message,
    parse_deferred_result,
)
from obelix.core.model.tool_message import ToolCall

# ---------------------------------------------------------------------------
# Lightweight fakes for a2a-sdk types (avoid importing the real SDK in tests)
# ---------------------------------------------------------------------------


@dataclass
class FakeRequestContext:
    """Minimal stand-in for a2a RequestContext."""

    task_id: str = "task-001"
    context_id: str | None = "ctx-001"

    def __post_init__(self):
        from a2a.types import Message, Part, Role, TextPart

        self.message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Hello agent"))],
            message_id="msg-001",
        )

    def get_user_input(self) -> str | None:
        return "Hello agent"


def _make_context_with_text(text: str, **kwargs) -> FakeRequestContext:
    """Create a FakeRequestContext with a specific text message."""
    from a2a.types import Message, Part, Role, TextPart

    ctx = FakeRequestContext(**kwargs)
    ctx.message = Message(
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        message_id="msg-001",
    )
    return ctx


def _make_empty_context(**kwargs) -> FakeRequestContext:
    """Create a FakeRequestContext with empty parts."""
    from a2a.types import Message, Role

    ctx = FakeRequestContext(**kwargs)
    ctx.message = Message(
        role=Role.user,
        parts=[],
        message_id="msg-001",
    )
    return ctx


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
    """Return (factory_callable, list_of_created_agents)."""
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
# agent_message helper
# ---------------------------------------------------------------------------


class TestAgentMessage:
    def test_returns_message_with_agent_role(self):
        from a2a.types import Message, Role

        msg = agent_message("some error")
        assert isinstance(msg, Message)
        assert msg.role == Role.agent

    def test_has_single_text_part_with_given_text(self):
        from a2a.types import TextPart

        msg = agent_message("specific error text")
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0].root, TextPart)
        assert msg.parts[0].root.text == "specific error text"

    def test_message_id_is_valid_uuid(self):
        import uuid

        msg = agent_message("test")
        parsed = uuid.UUID(msg.message_id)
        assert str(parsed) == msg.message_id


# ---------------------------------------------------------------------------
# ContextEntry
# ---------------------------------------------------------------------------


class TestContextEntry:
    def test_initial_state(self):
        entry = ContextEntry()
        assert entry.history == []
        assert isinstance(entry.idle, asyncio.Event)
        assert entry.idle.is_set()  # starts idle (ready)
        assert entry.deferred_tool_calls is None
        assert entry.trace_session is None
        assert entry.trace_span is None

    def test_history_is_mutable_list(self):
        entry = ContextEntry()
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
        assert executor._store._max_contexts == DEFAULT_MAX_CONTEXTS

    def test_custom_max_contexts(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=5)
        assert executor._store._max_contexts == 5

    def test_empty_contexts_on_init(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        assert len(executor._store._contexts) == 0


# ---------------------------------------------------------------------------
# _get_or_create_context
# ---------------------------------------------------------------------------


class TestGetOrCreateContext:
    def test_creates_new_context(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        entry = executor._store.get_or_create("ctx-1")
        assert isinstance(entry, ContextEntry)
        assert "ctx-1" in executor._store._contexts

    def test_returns_existing_context(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        entry1 = executor._store.get_or_create("ctx-1")
        entry2 = executor._store.get_or_create("ctx-1")
        assert entry1 is entry2

    def test_moves_existing_to_end(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        executor._store.get_or_create("ctx-1")
        executor._store.get_or_create("ctx-2")
        executor._store.get_or_create("ctx-1")  # should move to end
        keys = list(executor._store._contexts.keys())
        assert keys == ["ctx-2", "ctx-1"]

    def test_lru_eviction_at_capacity(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=2)
        executor._store.get_or_create("ctx-1")
        executor._store.get_or_create("ctx-2")
        executor._store.get_or_create("ctx-3")  # should evict ctx-1
        assert "ctx-1" not in executor._store._contexts
        assert "ctx-2" in executor._store._contexts
        assert "ctx-3" in executor._store._contexts

    def test_lru_eviction_preserves_recently_used(self):
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=2)
        executor._store.get_or_create("ctx-1")
        executor._store.get_or_create("ctx-2")
        executor._store.get_or_create("ctx-1")  # refresh ctx-1
        executor._store.get_or_create("ctx-3")  # should evict ctx-2 (oldest)
        assert "ctx-1" in executor._store._contexts
        assert "ctx-2" not in executor._store._contexts
        assert "ctx-3" in executor._store._contexts

    def test_eviction_history_is_lost(self):
        """Once a context is evicted, its history is gone."""
        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory, max_contexts=1)
        entry = executor._store.get_or_create("ctx-1")
        entry.history.append("some message")
        executor._store.get_or_create("ctx-2")  # evicts ctx-1
        new_entry = executor._store.get_or_create("ctx-1")  # re-created fresh
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
            Role,
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatusUpdateEvent,
        )

        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 4
        # Event 1: working (initial)
        assert isinstance(queue.events[0], TaskStatusUpdateEvent)
        assert queue.events[0].status.state == TaskState.working
        assert queue.events[0].final is False
        # Event 2: artifact (non-streaming: append=False, last_chunk=True)
        assert isinstance(queue.events[1], TaskArtifactUpdateEvent)
        assert queue.events[1].artifact.parts[0].root.text == "Agent response"
        assert queue.events[1].append is False
        assert queue.events[1].last_chunk is True
        # Event 3: working with agent message (for task history promotion)
        assert isinstance(queue.events[2], TaskStatusUpdateEvent)
        assert queue.events[2].status.state == TaskState.working
        assert queue.events[2].status.message is not None
        assert queue.events[2].status.message.role == Role.agent
        assert queue.events[2].status.message.parts[0].root.text == "Agent response"
        assert queue.events[2].final is False
        # Event 4: completed
        assert isinstance(queue.events[3], TaskStatusUpdateEvent)
        assert queue.events[3].status.state == TaskState.completed
        assert queue.events[3].final is True

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

            sys_msg = agent.system_message
            current_count = call_count

            async def fake_stream(query):
                agent.conversation_history = [
                    sys_msg,
                    HumanMessage(
                        content=query if isinstance(query, str) else query.content
                    ),
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
        ctx2 = _make_context_with_text("Follow-up question", context_id="ctx-mt")
        queue2 = FakeEventQueue()
        await executor.execute(ctx2, queue2)

        entry = executor._store._contexts["ctx-mt"]
        assert len(entry.history) >= 2

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
                    HumanMessage(
                        content=query if isinstance(query, str) else query.content
                    ),
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

        ctx_a = FakeRequestContext(context_id="ctx-A")
        await executor.execute(ctx_a, FakeEventQueue())

        ctx_b = FakeRequestContext(context_id="ctx-B")
        await executor.execute(ctx_b, FakeEventQueue())

        entry_a = executor._store._contexts["ctx-A"]
        entry_b = executor._store._contexts["ctx-B"]
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
        ctx = _make_context_with_text("")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 1
        event = queue.events[0]
        assert isinstance(event, TaskStatusUpdateEvent)
        assert event.status.state == TaskState.failed
        assert event.final is True
        _assert_error_message(event.status, "No user input provided")

    @pytest.mark.asyncio
    async def test_empty_parts_emits_failed_state(self):
        from a2a.types import TaskState

        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = _make_empty_context()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 1
        assert queue.events[0].status.state == TaskState.failed
        _assert_error_message(queue.events[0].status, "No user input provided")

    @pytest.mark.asyncio
    async def test_empty_input_does_not_create_agent(self):
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = _make_context_with_text("")
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

        ctx1 = _make_context_with_text("request-1", context_id="same-ctx")
        ctx2 = _make_context_with_text("request-2", context_id="same-ctx")

        await asyncio.gather(
            executor.execute(ctx1, FakeEventQueue()),
            executor.execute(ctx2, FakeEventQueue()),
        )

        starts = [i for i, (action, _) in enumerate(execution_log) if action == "start"]
        ends = [i for i, (action, _) in enumerate(execution_log) if action == "end"]
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

        ctx1 = _make_context_with_text("request-A", context_id="ctx-A")
        ctx2 = _make_context_with_text("request-B", context_id="ctx-B")

        await asyncio.gather(
            executor.execute(ctx1, FakeEventQueue()),
            executor.execute(ctx2, FakeEventQueue()),
        )

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
                    HumanMessage(
                        content=query if isinstance(query, str) else query.content
                    ),
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

        assert "ctx-1" not in executor._store._contexts
        assert "ctx-2" in executor._store._contexts
        assert "ctx-3" in executor._store._contexts
        assert len(executor._store._contexts) == 2


# ---------------------------------------------------------------------------
# Deferred tool (input-required) flow — now with DataPart
# ---------------------------------------------------------------------------


def _make_deferred_factory():
    """Create a factory whose agent yields deferred_tool_calls, then on resume completes."""
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage
    from obelix.core.model.tool_message import ToolCall

    created: list[MagicMock] = []

    def factory():
        agent = MagicMock()
        agent.system_message = SystemMessage(content="Test agent")
        agent.conversation_history = [agent.system_message]
        agent._tracer = None

        call_count = len(created)

        if call_count == 0:

            async def first_stream(query):
                yield StreamEvent(
                    deferred_tool_calls=[
                        ToolCall(
                            id="tc-deferred-1",
                            name="request_user_input",
                            arguments={"question": "Which currency?"},
                        )
                    ],
                    is_final=True,
                )

            agent.execute_query_stream = MagicMock(side_effect=first_stream)
        else:

            async def resume_stream():
                yield StreamEvent(
                    is_final=True,
                    assistant_response=AssistantResponse(
                        agent_name="test_agent",
                        content="Result: EUR",
                    ),
                )

            agent.resume_after_deferred = MagicMock(side_effect=resume_stream)

        created.append(agent)
        return agent

    return factory, created


class TestDeferredInputRequired:
    @pytest.mark.asyncio
    async def test_emits_input_required_with_data_part(self):
        """When agent yields deferred_tool_calls, executor emits input-required with DataPart."""
        from a2a.types import DataPart, TaskState

        factory, _ = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(task_id="t-ir", context_id="ctx-ir")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # Events: working -> input-required
        assert len(queue.events) == 2
        assert queue.events[0].status.state == TaskState.working
        assert queue.events[1].status.state == TaskState.input_required
        assert queue.events[1].final is True

        # The message contains a DataPart with deferred_tool_calls
        msg = queue.events[1].status.message
        assert len(msg.parts) == 1
        part_root = msg.parts[0].root
        assert isinstance(part_root, DataPart)
        assert "deferred_tool_calls" in part_root.data
        calls = part_root.data["deferred_tool_calls"]
        assert len(calls) == 1
        assert calls[0]["tool_name"] == "request_user_input"
        assert calls[0]["arguments"]["question"] == "Which currency?"

    @pytest.mark.asyncio
    async def test_deferred_calls_saved_in_context(self):
        factory, _ = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(context_id="ctx-saved")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        entry = executor._store._contexts["ctx-saved"]
        assert entry.deferred_tool_calls is not None
        assert len(entry.deferred_tool_calls) == 1
        assert entry.deferred_tool_calls[0].name == "request_user_input"

    @pytest.mark.asyncio
    async def test_context_becomes_idle_after_input_required(self):
        factory, _ = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext(context_id="ctx-idle")
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        entry = executor._store._contexts["ctx-idle"]
        assert entry.idle.is_set()


class TestResumeAfterDeferred:
    @pytest.mark.asyncio
    async def test_resume_completes_with_answer(self):
        """After input-required, second message resumes and completes."""
        from a2a.types import (
            DataPart,
            Message,
            Part,
            Role,
            TaskArtifactUpdateEvent,
            TaskState,
            TaskStatusUpdateEvent,
        )

        factory, _ = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)

        # First request -> input-required
        ctx1 = FakeRequestContext(task_id="t-1", context_id="ctx-resume")
        queue1 = FakeEventQueue()
        await executor.execute(ctx1, queue1)

        assert queue1.events[-1].status.state == TaskState.input_required

        # Second request -> delivers the answer as DataPart
        ctx2 = FakeRequestContext(task_id="t-2", context_id="ctx-resume")
        ctx2.message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data={"answer": "EUR"}))],
            message_id="msg-resume",
        )
        queue2 = FakeEventQueue()
        await executor.execute(ctx2, queue2)

        # Events on queue2: working -> artifact -> completed
        states = [
            e.status.state
            for e in queue2.events
            if isinstance(e, TaskStatusUpdateEvent)
        ]
        assert TaskState.working in states
        assert TaskState.completed in states

        artifacts = [e for e in queue2.events if isinstance(e, TaskArtifactUpdateEvent)]
        assert len(artifacts) >= 1
        artifact_text = artifacts[0].artifact.parts[0].root.text
        assert "EUR" in artifact_text

    @pytest.mark.asyncio
    async def test_deferred_calls_cleared_after_resume(self):
        from a2a.types import DataPart, Message, Part, Role

        factory, _ = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)

        # First request -> input-required
        ctx1 = FakeRequestContext(context_id="ctx-clear")
        await executor.execute(ctx1, FakeEventQueue())

        # Second request -> resume
        ctx2 = FakeRequestContext(context_id="ctx-clear")
        ctx2.message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data={"answer": "answer"}))],
            message_id="msg-resume",
        )
        await executor.execute(ctx2, FakeEventQueue())

        entry = executor._store._contexts["ctx-clear"]
        assert entry.deferred_tool_calls is None
        assert entry.idle.is_set()

    @pytest.mark.asyncio
    async def test_tool_message_injected_on_resume(self):
        from a2a.types import DataPart, Message, Part, Role

        factory, created = _make_deferred_factory()
        executor = ObelixAgentExecutor(factory)

        # First request -> input-required
        ctx1 = FakeRequestContext(context_id="ctx-inject")
        await executor.execute(ctx1, FakeEventQueue())

        # Second request -> resume with DataPart
        ctx2 = FakeRequestContext(context_id="ctx-inject")
        ctx2.message = Message(
            role=Role.user,
            parts=[Part(root=DataPart(data={"answer": "USD"}))],
            message_id="msg-resume",
        )
        await executor.execute(ctx2, FakeEventQueue())

        resume_agent = created[1]
        resume_agent.resume_after_deferred.assert_called_once()


class TestNormalFlowUnchanged:
    @pytest.mark.asyncio
    async def test_normal_execution_without_deferred_tool(self):
        from a2a.types import TaskState

        factory, _ = _make_factory()
        executor = ObelixAgentExecutor(factory)
        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        assert len(queue.events) == 3  # working -> artifact -> completed
        assert queue.events[0].status.state == TaskState.working
        assert queue.events[2].status.state == TaskState.completed

        entry = executor._store._contexts["ctx-001"]
        assert entry.idle.is_set()
        assert entry.deferred_tool_calls is None


# ---------------------------------------------------------------------------
# parse_deferred_result (now takes dict, not str)
# ---------------------------------------------------------------------------


class _MockOutputSchema(BaseModel):
    """Pydantic schema for a bash-like tool output."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


class _MockBashTool:
    """Fake tool with _output_schema (mimics what @tool decorator sets)."""

    tool_name = "bash"
    _output_schema = _MockOutputSchema


class _MockSimpleTool:
    """Fake tool without _output_schema (like RequestUserInputTool)."""

    tool_name = "request_user_input"


class TestParseDeferredResult:
    def test_with_output_schema_valid_data(self):
        """Tool has _output_schema, client sends valid dict -> validated dict."""
        call = ToolCall(id="tc-1", name="bash", arguments={})
        tools = [_MockBashTool(), _MockSimpleTool()]
        data = {"stdout": "hello world", "exit_code": 0}

        result = parse_deferred_result(call, tools, data)

        assert result["stdout"] == "hello world"
        assert result["exit_code"] == 0
        assert result["stderr"] == ""  # default filled in by schema

    def test_with_output_schema_extra_fields_ignored(self):
        """Tool has _output_schema, extra fields in data -> validated with defaults."""
        call = ToolCall(id="tc-1", name="bash", arguments={})
        tools = [_MockBashTool()]
        data = {"unexpected_field": "value"}

        result = parse_deferred_result(call, tools, data)

        # Pydantic ignores extra fields and fills defaults
        assert result["stdout"] == ""
        assert result["stderr"] == ""
        assert result["exit_code"] == 0

    def test_without_output_schema(self):
        """Tool has no _output_schema -> returns data dict directly."""
        call = ToolCall(id="tc-1", name="request_user_input", arguments={})
        tools = [_MockSimpleTool()]
        data = {"answer": "EUR"}

        result = parse_deferred_result(call, tools, data)

        assert result == {"answer": "EUR"}

    def test_no_tools_provided(self):
        """Tools list is None -> returns data dict directly."""
        call = ToolCall(id="tc-1", name="bash", arguments={})
        data = {"answer": "some answer"}

        result = parse_deferred_result(call, None, data)

        assert result == {"answer": "some answer"}


# ---------------------------------------------------------------------------
# Multi-part inbound (attachments)
# ---------------------------------------------------------------------------


class TestMultiPartInbound:
    @pytest.mark.asyncio
    async def test_file_part_creates_human_message_with_attachments(self):
        """A FilePart in the message creates a HumanMessage with attachments."""
        from a2a.types import FilePart, FileWithBytes, Message, Part, Role, TextPart

        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)

        ctx = FakeRequestContext()
        ctx.message = Message(
            role=Role.user,
            parts=[
                Part(root=TextPart(text="Describe this image")),
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes="aGVsbG8=",  # base64 "hello"
                            mime_type="image/png",
                            name="test.png",
                        )
                    )
                ),
            ],
            message_id="msg-mp",
        )
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        # Agent should receive a HumanMessage (not a string)
        from obelix.core.model.human_message import HumanMessage

        call_args = created[0].execute_query_stream.call_args[0][0]
        assert isinstance(call_args, HumanMessage)
        assert call_args.content == "Describe this image"
        assert len(call_args.attachments) == 1
        assert call_args.attachments[0].mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_data_part_creates_human_message_with_attachments(self):
        """A DataPart in the message creates a HumanMessage with DataContent attachment."""
        from a2a.types import DataPart, Message, Part, Role, TextPart

        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)

        ctx = FakeRequestContext()
        ctx.message = Message(
            role=Role.user,
            parts=[
                Part(root=TextPart(text="Analyze this")),
                Part(root=DataPart(data={"key": "value"})),
            ],
            message_id="msg-dp",
        )
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        from obelix.core.model.content import DataContent
        from obelix.core.model.human_message import HumanMessage

        call_args = created[0].execute_query_stream.call_args[0][0]
        assert isinstance(call_args, HumanMessage)
        assert call_args.content == "Analyze this"
        assert len(call_args.attachments) == 1
        assert isinstance(call_args.attachments[0], DataContent)
        assert call_args.attachments[0].data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_text_only_message_passes_string(self):
        """A text-only message passes a plain string to the agent (no HumanMessage)."""
        factory, created = _make_factory()
        executor = ObelixAgentExecutor(factory)

        ctx = FakeRequestContext()
        queue = FakeEventQueue()

        await executor.execute(ctx, queue)

        created[0].execute_query_stream.assert_called_once_with("Hello agent")
