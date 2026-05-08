"""In-place mutation of Task.metadata via the SDK's TaskStore.

Verified pattern from pre-impl research (spike a2a-metadata-mutability):
- TaskStatusUpdateEvent post-terminal is dropped silently (queue closed).
- TaskStore.save() is the only working path.
- DefaultRequestHandler.on_get_task always re-reads from the store, so
  the next polling client sees the patched metadata immediately.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from a2a.server.tasks.task_store import TaskStore

PatchFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


async def update_task_metadata(
    store: TaskStore,
    task_id: str,
    patch_fn: PatchFn,
) -> None:
    """Read the Task, run patch_fn over a mutable copy of its metadata, save.

    No-ops silently if the task is not in the store (evicted, never existed).

    The patch_fn receives a dict that is safe to mutate in-place; whatever
    dict it returns becomes the new ``Task.metadata``. Returning ``{}``
    clears the metadata; returning ``None`` is treated as ``{}``.
    """
    task = await store.get(task_id)
    if task is None:
        return
    current = dict(task.metadata) if task.metadata else {}
    new_meta = await patch_fn(current)
    task.metadata = new_meta or {}
    await store.save(task)
