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

    # Wait briefly for the spawned background task to register T3 + patch
    # T1.metadata. The drainer is fire-and-forget; we observe both
    # side effects landed.
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
