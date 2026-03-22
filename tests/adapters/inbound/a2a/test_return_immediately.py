"""Test return_immediately (non-blocking) flow through the real SDK DefaultRequestHandler.

Verifies that when a client sends blocking=False in SendMessageConfiguration,
the server returns the Task immediately without waiting for the agent to finish.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from a2a.server.events import InMemoryQueueManager
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    Role,
    TaskQueryParams,
    TaskState,
    TextPart,
)

from obelix.adapters.inbound.a2a.server import ObelixAgentExecutor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slow_agent(delay: float = 0.5, response: str = "Done"):
    """Create a mock agent that takes `delay` seconds to respond."""
    from obelix.core.model.assistant_message import AssistantResponse, StreamEvent
    from obelix.core.model.system_message import SystemMessage

    agent = MagicMock()
    agent.system_message = SystemMessage(content="You are a test agent.")
    agent.conversation_history = [agent.system_message]
    agent.registered_tools = []

    async def fake_stream(query):
        await asyncio.sleep(delay)
        yield StreamEvent(
            is_final=True,
            assistant_response=AssistantResponse(
                agent_name="test_agent",
                content=response,
            ),
        )

    agent.execute_query_stream = MagicMock(side_effect=fake_stream)
    return agent


def _make_send_params(
    text: str = "Hello",
    *,
    blocking: bool | None = None,
    context_id: str | None = None,
) -> MessageSendParams:
    """Build a MessageSendParams with optional blocking configuration."""
    config = None
    if blocking is not None:
        config = MessageSendConfiguration(blocking=blocking)

    return MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(root=TextPart(text=text))],
            message_id="msg-001",
            context_id=context_id,
        ),
        configuration=config,
    )


def _build_handler(
    delay: float = 0.5,
) -> tuple[DefaultRequestHandler, InMemoryTaskStore]:
    """Build a DefaultRequestHandler wired to our executor with a slow agent."""

    def factory():
        return _make_slow_agent(delay)

    executor = ObelixAgentExecutor(factory)
    task_store = InMemoryTaskStore()
    queue_manager = InMemoryQueueManager()

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        queue_manager=queue_manager,
    )
    return handler, task_store


async def _drain_background_tasks(handler: DefaultRequestHandler) -> None:
    """Cancel and await all background tasks so the event loop can close cleanly."""
    # Cancel running agent producers
    async with handler._running_agents_lock:
        for task in handler._running_agents.values():
            task.cancel()
    # Cancel background cleanup/consume tasks
    for task in list(handler._background_tasks):
        task.cancel()
    # Give them a moment to process cancellation
    await asyncio.sleep(0.05)
    # Gather all to suppress CancelledError
    all_tasks = list(handler._background_tasks)
    async with handler._running_agents_lock:
        all_tasks.extend(handler._running_agents.values())
    for task in all_tasks:
        try:
            await asyncio.wait_for(task, timeout=0.5)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBlockingDefault:
    """Baseline: blocking mode waits for completion."""

    @pytest.mark.asyncio
    async def test_blocking_true_waits_for_completion(self):
        handler, _ = _build_handler(delay=0.1)
        params = _make_send_params(blocking=True)

        result = await handler.on_message_send(params)

        assert result.status.state == TaskState.completed

    @pytest.mark.asyncio
    async def test_no_config_defaults_to_blocking(self):
        handler, _ = _build_handler(delay=0.1)
        params = _make_send_params()

        result = await handler.on_message_send(params)

        assert result.status.state == TaskState.completed


class TestNonBlocking:
    """Non-blocking (return_immediately) flow."""

    @pytest.mark.asyncio
    async def test_returns_non_terminal_state(self):
        """blocking=False: server returns Task in non-terminal state."""
        handler, _ = _build_handler(delay=2.0)
        params = _make_send_params(blocking=False)

        result = await handler.on_message_send(params)

        assert result.status.state in (
            TaskState.submitted,
            TaskState.working,
        ), f"Expected non-terminal state, got {result.status.state}"

        await _drain_background_tasks(handler)

    @pytest.mark.asyncio
    async def test_returns_fast(self):
        """Non-blocking should return much faster than agent delay."""
        handler, _ = _build_handler(delay=2.0)
        params = _make_send_params(blocking=False)

        start = time.monotonic()
        await handler.on_message_send(params)
        elapsed = time.monotonic() - start

        assert elapsed < 1.0, f"Non-blocking took {elapsed:.2f}s, expected < 1s"

        await _drain_background_tasks(handler)

    @pytest.mark.asyncio
    async def test_task_completes_in_background(self):
        """After non-blocking return, task eventually reaches completed."""
        handler, task_store = _build_handler(delay=0.2)
        params = _make_send_params(blocking=False)

        result = await handler.on_message_send(params)
        task_id = result.id

        assert result.status.state in (TaskState.submitted, TaskState.working)

        # Poll until completed or timeout
        for _ in range(30):
            await asyncio.sleep(0.1)
            task = await task_store.get(task_id)
            if task and task.status.state == TaskState.completed:
                break

        completed = await handler.on_get_task(TaskQueryParams(id=task_id))
        assert completed.status.state == TaskState.completed

        await _drain_background_tasks(handler)

    @pytest.mark.asyncio
    async def test_completed_task_has_artifacts(self):
        """After background completion, task has artifacts."""
        handler, task_store = _build_handler(delay=0.2)
        params = _make_send_params(blocking=False)

        result = await handler.on_message_send(params)
        task_id = result.id

        # Poll until completed
        for _ in range(30):
            await asyncio.sleep(0.1)
            task = await task_store.get(task_id)
            if task and task.status.state == TaskState.completed:
                break

        task = await handler.on_get_task(TaskQueryParams(id=task_id))
        assert task.status.state == TaskState.completed
        assert task.artifacts is not None
        assert len(task.artifacts) > 0

        await _drain_background_tasks(handler)

    @pytest.mark.asyncio
    async def test_completed_task_has_history(self):
        """After background completion, task has user + agent in history."""
        handler, task_store = _build_handler(delay=0.2)
        params = _make_send_params(blocking=False)

        result = await handler.on_message_send(params)
        task_id = result.id

        # Poll until completed
        for _ in range(30):
            await asyncio.sleep(0.1)
            task = await task_store.get(task_id)
            if task and task.status.state == TaskState.completed:
                break

        task = await handler.on_get_task(TaskQueryParams(id=task_id, history_length=10))
        assert task.history is not None
        assert len(task.history) >= 1

        await _drain_background_tasks(handler)
