"""End-to-end test of update_task_metadata against a real InMemoryTaskStore.

Iron rule: no mocks. The SDK runs against itself.
"""

from __future__ import annotations

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.metadata_patch import update_task_metadata


def _seed_task(store: InMemoryTaskStore, task_id: str, metadata=None) -> Task:
    task = Task(
        id=task_id,
        context_id="ctx-test",
        status=TaskStatus(state=TaskState.completed),
        metadata=metadata,
    )
    return task


@pytest.mark.asyncio
async def test_appends_to_list_field_when_metadata_is_none():
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1")
    await store.save(task)

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = list(meta.get("spawned_task_ids", [])) + ["t3"]
        return meta

    await update_task_metadata(store, "t1", add_child)

    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata == {"spawned_task_ids": ["t3"]}


@pytest.mark.asyncio
async def test_preserves_other_metadata_fields():
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1", metadata={"existing": "value"})
    await store.save(task)

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = ["t3"]
        return meta

    await update_task_metadata(store, "t1", add_child)

    refreshed = await store.get("t1")
    assert refreshed.metadata == {"existing": "value", "spawned_task_ids": ["t3"]}


@pytest.mark.asyncio
async def test_noop_when_task_evicted():
    """If the task is gone from the store, the helper must NOT raise."""
    store = InMemoryTaskStore()

    async def add_child(meta: dict) -> dict:
        meta["spawned_task_ids"] = ["t3"]
        return meta

    # Should silently no-op — task never existed.
    await update_task_metadata(store, "missing", add_child)


@pytest.mark.asyncio
async def test_concurrent_appends_no_loss_under_serial_calls():
    """Two sequential calls (the realistic case for spec 2's drainer)
    must not lose data."""
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1")
    await store.save(task)

    async def add(child_id: str):
        async def patch(meta: dict) -> dict:
            meta["spawned_task_ids"] = list(meta.get("spawned_task_ids", [])) + [
                child_id
            ]
            return meta

        await update_task_metadata(store, "t1", patch)

    await add("t3")
    await add("t4")

    refreshed = await store.get("t1")
    assert refreshed.metadata["spawned_task_ids"] == ["t3", "t4"]


@pytest.mark.asyncio
async def test_patch_fn_returning_empty_dict_clears_metadata():
    """Docstring contract: returning ``{}`` clears the metadata."""
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1", metadata={"existing": "value"})
    await store.save(task)

    async def clear(_meta: dict) -> dict:
        return {}

    await update_task_metadata(store, "t1", clear)

    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata == {}


@pytest.mark.asyncio
async def test_patch_fn_returning_none_clears_metadata():
    """Docstring contract: returning ``None`` is treated as ``{}``."""
    store = InMemoryTaskStore()
    task = _seed_task(store, "t1", metadata={"existing": "value"})
    await store.save(task)

    async def clear(_meta: dict):
        return None

    await update_task_metadata(store, "t1", clear)

    refreshed = await store.get("t1")
    assert refreshed is not None
    assert refreshed.metadata == {}
