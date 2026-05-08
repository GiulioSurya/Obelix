"""Verify that when the drainer spawns T3 it:
  1. saves T3 to the SDK TaskStore FIRST (so tasks/get(T3) returns 200).
  2. patches T1.metadata.spawned_task_ids to include T3.id.

Order matters: a polling client that learns T3.id but cannot yet retrieve
the Task gets a confusing -32001. Race verified empirically in spec 1
smoke testing (Bug 6).

Iron rule: no SDK mocks. We feed a real InMemoryTaskStore and observe.

Reality vs plan: the real entry point on the executor is
``spawn_drain_task`` (not ``maybe_spawn_drain_task`` — that name belongs
to the standalone drainer dispatch function in ``drainer.py``). This
test calls the real method and passes ``parent_task_id`` so the
metadata-patch branch can identify T1.
"""

from __future__ import annotations

import asyncio

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


def _build_executor(task_store):
    """Minimal executor wired with a real TaskStore.

    The agent_factory deliberately raises if invoked — this test only
    exercises the metadata-write side effects, not the agent run.
    """

    def _agent_factory():
        raise AssertionError("agent_factory should not run in this test")

    return ObelixAgentExecutor(
        agent_factory=_agent_factory,
        task_store=task_store,
    )


@pytest.mark.asyncio
async def test_drainer_appends_spawned_task_id_after_saving_child():
    store = InMemoryTaskStore()
    parent = Task(
        id="t1",
        context_id="ctx-share",
        status=TaskStatus(state=TaskState.completed),
        metadata=None,
    )
    await store.save(parent)

    executor = _build_executor(store)
    entry = ContextEntry()
    entry.context_id = "ctx-share"
    # Replicate post-completion state: agent done, parent terminal,
    # nothing yet in the agent history.
    entry.history = []
    entry.pending_notifications = []

    new_id = await executor.spawn_drain_task(
        parent_task_id="t1",
        context_id="ctx-share",
        entry=entry,
    )

    # The metadata writes (store.save(T3) + update_task_metadata(T1)) complete
    # synchronously inside spawn_drain_task BEFORE the asyncio.create_task that
    # kicks the agent run. The sleep below only yields so the background drain
    # task can start and fail fast (agent_factory raises, errors swallowed by
    # the drainer) — the assertions below do not depend on it.
    await asyncio.sleep(0.1)

    assert isinstance(new_id, str) and new_id, (
        "spawn_drain_task must return the freshly-generated child task_id"
    )

    # 1. Child task is in the store.
    child = await store.get(new_id)
    assert child is not None, (
        "T3 must be saved to TaskStore before T1.metadata is patched"
    )

    # 2. T1.metadata.spawned_task_ids contains T3.id.
    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata is not None
    assert new_id in refreshed.metadata.get("spawned_task_ids", []), (
        f"expected T3.id={new_id!r} in T1.metadata.spawned_task_ids, "
        f"got metadata={refreshed.metadata!r}"
    )


@pytest.mark.asyncio
async def test_drainer_via_drainer_module_writes_metadata_when_current_task_id_is_set():
    """Smoke test that mirrors the production call site: drainer.maybe_spawn_drain_task
    is invoked with an executor whose entry.current_task_id is set. The metadata path
    must fire end-to-end (this is the spec-2 requirement that Task 4 closes)."""
    from obelix.adapters.inbound.a2a.server.drainer import maybe_spawn_drain_task
    from obelix.core.model.human_message import HumanMessage

    store = InMemoryTaskStore()
    parent = Task(
        id="t1",
        context_id="ctx-prod",
        status=TaskStatus(state=TaskState.completed),
        metadata=None,
    )
    await store.save(parent)

    executor = _build_executor(store)
    entry = ContextEntry()
    entry.context_id = "ctx-prod"
    entry.history = []
    # maybe_spawn_drain_task short-circuits when pending_notifications is
    # empty — load one notification to force the drainer down its happy path.
    entry.pending_notifications = [HumanMessage(content="remote update")]
    entry.current_task_id = "t1"

    await maybe_spawn_drain_task(
        entry=entry,
        context_id="ctx-prod",
        executor=executor,
        parent_task_id=entry.current_task_id,
    )

    # Yield to let the background drain task start and fail fast (the
    # agent_factory raises, errors swallowed by drainer). Metadata writes
    # complete synchronously before the create_task, so this is just for
    # the background coroutine's lifecycle, not the assertions.
    await asyncio.sleep(0.05)

    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata is not None
    spawned = refreshed.metadata.get("spawned_task_ids", [])
    assert len(spawned) == 1, (
        f"production-path drainer must write exactly one spawned_task_id, got {spawned!r}"
    )
