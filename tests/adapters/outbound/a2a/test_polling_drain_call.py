"""Verify PollingWorker._poll_one calls maybe_spawn_drain_task after
handle_remote_update.

Iron rule: hand-written ``_RecordingExecutor`` (drainer Protocol) and
``_FakeClient`` (a2a SDK Client surface). Real ``ContextStore``,
``RemoteAgentRegistry``, ``RemoteTaskState``. No ``unittest.mock`` /
``pytest-mock`` / ``monkeypatch``.
"""

from __future__ import annotations

import inspect
import time
from datetime import UTC, datetime

import httpx
import pytest
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.outbound.a2a.polling import PollingWorker
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.state import RemoteTaskState


class _RecordingExecutor:
    """FakeExecutor compatible with the drainer ``_DrainExecutorProtocol``.

    Captures every ``spawn_drain_task`` invocation as a tuple
    ``(entry, context_id, parent_task_id)`` so tests can assert call
    count + args.
    """

    def __init__(self) -> None:
        self.spawn_calls: list[tuple[ContextEntry, str, str | None]] = []

    async def spawn_drain_task(
        self,
        *,
        entry: ContextEntry,
        context_id: str,
        parent_task_id: str | None = None,
    ) -> None:
        self.spawn_calls.append((entry, context_id, parent_task_id))


class _FakeClient:
    """Minimal Client substitute exposing only the ``get_task`` coroutine
    used by ``PollingWorker._poll_one``. Returns a Task from the fixture
    (no SDK round-trip)."""

    def __init__(self, fresh_task: Task) -> None:
        self._fresh = fresh_task

    async def get_task(self, params):  # noqa: D401 — SDK signature mirror
        return self._fresh


def _seed_state(
    entry: ContextEntry,
    *,
    task_id: str,
    token: str,
    agent_name: str = "coordinator",
    status: str = "working",
) -> None:
    """Insert a stale (>30s old) non-terminal RemoteTaskState into the
    entry so ``_tick_once`` will pick it for polling."""
    entry.remote_tasks[task_id] = RemoteTaskState(
        task_id=task_id,
        agent_name=agent_name,
        status=status,
        token=token,
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic() - 60.0,  # stale
        last_artifact=None,
        deferred_calls=None,
    )


def test_polling_worker_constructor_accepts_executor() -> None:
    """The constructor must declare an ``executor`` parameter so the call
    site in ``AgentFactory.a2a_serve`` can wire the real executor in."""
    sig = inspect.signature(PollingWorker.__init__)
    assert "executor" in sig.parameters


@pytest.mark.asyncio
async def test_polling_invokes_drainer_after_state_change() -> None:
    """End-to-end: a stale ``working`` task whose remote returns a
    completed Task triggers ``handle_remote_update`` (which appends a
    notification + revokes the token), then ``_poll_one`` awaits
    ``maybe_spawn_drain_task`` and the drainer's checks pass (queue
    non-empty, ``entry.idle.is_set()`` defaults to True), so the
    executor's ``spawn_drain_task`` is invoked exactly once."""
    store = ContextStore(max_contexts=8)
    entry = store.get_or_create("ctx-orch")
    _seed_state(entry, task_id="t-remote", token="tok-remote")

    fresh_task = Task(
        id="t-remote",
        context_id="ctx-orch",
        status=TaskStatus(state=TaskState.completed),
    )

    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    registry.register_token(
        "tok-remote", context_id="ctx-orch", agent_name="coordinator"
    )
    registry.claim_task_id("tok-remote", task_id="t-remote")
    # Inject the fake client into the registry's internal map. The public
    # API exposes only ``client_for(name)`` (lookup) and ``resolve_all``
    # (URL fetch); for an in-process test we need to register a client
    # without touching the network, so we mutate ``_clients`` directly.
    registry._clients["coordinator"] = _FakeClient(fresh_task)

    executor = _RecordingExecutor()
    worker = PollingWorker(
        registry=registry,
        context_store=store,
        executor=executor,
        tick_seconds=0.05,
    )

    await worker._tick_once()

    # handle_remote_update fired (state changed to completed + notification queued).
    state = entry.remote_tasks["t-remote"]
    assert state.status == "completed"
    assert len(entry.pending_notifications) == 1

    # Drainer was invoked exactly once with the right keyword arguments.
    assert len(executor.spawn_calls) == 1
    spawned_entry, spawned_context_id, spawned_parent_task_id = executor.spawn_calls[0]
    assert spawned_entry is entry
    assert spawned_context_id == "ctx-orch"
    # No active turn in this test, so entry.current_task_id stays None and
    # the drainer forwards None — the metadata-patch branch safely no-ops.
    assert spawned_parent_task_id is None


@pytest.mark.asyncio
async def test_polling_no_drain_when_executor_none() -> None:
    """Backward-compat: ``executor=None`` (default) means no drainer call
    and no error — empty store, no remote tasks, just exercise the
    constructor + ``_tick_once`` paths without an executor."""
    store = ContextStore(max_contexts=8)

    worker = PollingWorker(
        registry=RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient()),
        context_store=store,
        executor=None,
        tick_seconds=0.05,
    )

    # No exception, no spawn (there's no recording executor anyway).
    await worker._tick_once()


@pytest.mark.asyncio
async def test_polling_skips_drainer_when_context_busy() -> None:
    """If ``entry.idle`` is cleared (a turn is running on this context),
    the drainer's check 2 short-circuits and ``spawn_drain_task`` is NOT
    called — even though ``handle_remote_update`` queued a notification."""
    store = ContextStore(max_contexts=8)
    entry = store.get_or_create("ctx-orch")
    _seed_state(entry, task_id="t-remote", token="tok-remote")
    entry.idle.clear()  # simulate a turn already running

    fresh_task = Task(
        id="t-remote",
        context_id="ctx-orch",
        status=TaskStatus(state=TaskState.completed),
    )

    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    registry.register_token(
        "tok-remote", context_id="ctx-orch", agent_name="coordinator"
    )
    registry.claim_task_id("tok-remote", task_id="t-remote")
    registry._clients["coordinator"] = _FakeClient(fresh_task)

    executor = _RecordingExecutor()
    worker = PollingWorker(
        registry=registry,
        context_store=store,
        executor=executor,
        tick_seconds=0.05,
    )

    await worker._tick_once()

    # Notification was still queued (handler ran), but drainer did NOT spawn.
    assert len(entry.pending_notifications) == 1
    assert executor.spawn_calls == []
