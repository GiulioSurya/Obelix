import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.outbound.a2a.polling import PollingWorker
from obelix.adapters.outbound.a2a.state import RemoteTaskState


def _seed(entry: ContextEntry, *, task_id: str, status: str = "working") -> None:
    entry.remote_tasks[task_id] = RemoteTaskState(
        task_id=task_id,
        agent_name="B",
        status=status,
        token=f"tok-{task_id}",
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic() - 60,  # 60s ago — stale
        last_artifact=None,
        deferred_calls=None,
    )


@pytest.fixture
def store() -> ContextStore:
    s = ContextStore(max_contexts=10)
    s.get_or_create("ctx-AAA")
    return s


@pytest.mark.asyncio
async def test_skips_terminal_tasks(store):
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="completed")

    registry = MagicMock()
    client = AsyncMock()
    registry.client_for.return_value = client
    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)

    await worker._tick_once()
    client.get_task.assert_not_called()


@pytest.mark.asyncio
async def test_skips_recently_updated_tasks(store):
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="working")
    # Override to look fresh
    entry.remote_tasks["t-1"].last_update_monotonic = time.monotonic()

    registry = MagicMock()
    client = AsyncMock()
    registry.client_for.return_value = client
    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)

    await worker._tick_once()
    client.get_task.assert_not_called()


@pytest.mark.asyncio
async def test_polls_stale_non_terminal_task(store):
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="working")

    fresh = Task(
        id="t-1",
        context_id="ctx-AAA",
        status=TaskStatus(state=TaskState.completed),
    )
    registry = MagicMock()
    client = AsyncMock()
    client.get_task.return_value = fresh
    registry.client_for.return_value = client

    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)
    await worker._tick_once()

    client.get_task.assert_called_once()
    assert entry.remote_tasks["t-1"].status == "completed"


@pytest.mark.asyncio
async def test_giveup_after_5_failures(store):
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="working")

    registry = MagicMock()
    client = AsyncMock()
    client.get_task.side_effect = RuntimeError("boom")
    registry.client_for.return_value = client

    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)

    for _ in range(5):
        # Reset last_update_monotonic so each tick is "stale enough"
        entry.remote_tasks["t-1"].last_update_monotonic = time.monotonic() - 60
        await worker._tick_once()

    state = entry.remote_tasks["t-1"]
    assert state.poll_failures == 5
    assert state.status == "failed"
    assert any("polling_giveup" in m.content for m in entry.pending_notifications)
    registry.revoke.assert_called_once()


@pytest.mark.asyncio
async def test_failure_streak_resets_on_success(store):
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="working")

    registry = MagicMock()
    client = AsyncMock()
    client.get_task.side_effect = [
        RuntimeError("boom"),
        RuntimeError("boom"),
        Task(
            id="t-1",
            context_id="ctx-AAA",
            status=TaskStatus(state=TaskState.working),  # state UNCHANGED
        ),
    ]
    registry.client_for.return_value = client

    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)
    for _ in range(3):
        entry.remote_tasks["t-1"].last_update_monotonic = time.monotonic() - 60
        await worker._tick_once()

    # The third call returns a Task with same state ("working") — handler is
    # called but its same-state early return means it does NOT touch
    # poll_failures. So the worker MUST reset poll_failures on every
    # successful HTTP response (regardless of state change).
    assert entry.remote_tasks["t-1"].poll_failures == 0
    assert entry.remote_tasks["t-1"].status == "working"


@pytest.mark.asyncio
async def test_start_stop_lifecycle(store):
    registry = MagicMock()
    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)
    await worker.start()
    await asyncio.sleep(0.03)
    await worker.stop()
    # No assertion; verify clean lifecycle (no hang, no exception).


@pytest.mark.asyncio
async def test_polls_across_multiple_contexts(store):
    """_tick_once iterates ALL contexts in the store, not just the first."""
    e1 = store.get_or_create("ctx-AAA")
    e2 = store.get_or_create("ctx-BBB")
    _seed(e1, task_id="t-A1", status="working")
    _seed(e2, task_id="t-B1", status="working")

    fresh_a = Task(
        id="t-A1",
        context_id="ctx-AAA",
        status=TaskStatus(state=TaskState.completed),
    )
    fresh_b = Task(
        id="t-B1",
        context_id="ctx-BBB",
        status=TaskStatus(state=TaskState.completed),
    )

    registry = MagicMock()
    client = AsyncMock()

    # Return the right Task based on task_id arg
    def _resolve(params):
        return fresh_a if params.id == "t-A1" else fresh_b

    client.get_task.side_effect = _resolve
    registry.client_for.return_value = client

    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)
    await worker._tick_once()

    assert client.get_task.call_count == 2
    assert e1.remote_tasks["t-A1"].status == "completed"
    assert e2.remote_tasks["t-B1"].status == "completed"


@pytest.mark.asyncio
async def test_fresh_none_is_noop(store):
    """SDK returning None (task not found) should be a no-op:
    state untouched, poll_failures NOT incremented, no notification."""
    entry = store.get_or_create("ctx-AAA")
    _seed(entry, task_id="t-1", status="working")

    registry = MagicMock()
    client = AsyncMock()
    client.get_task.return_value = None
    registry.client_for.return_value = client

    worker = PollingWorker(registry=registry, context_store=store, tick_seconds=0.01)
    await worker._tick_once()

    state = entry.remote_tasks["t-1"]
    assert state.status == "working"  # unchanged
    assert state.poll_failures == 0  # NOT incremented
    assert entry.pending_notifications == []
    registry.revoke.assert_not_called()
