"""Auto-spawn drainer for pending_notifications.

When a remote A2A task completes and `handle_remote_update` accodes a
HumanMessage in `entry.pending_notifications`, this function is called by
the two async call sites (webhook.py outbound, polling.py) to decide if a
new A2A turn should be started spontaneously.

The drainer is event-driven (not a loop). It performs two cheap checks:
  1. Is there at least one pending notification? (else nothing to drain)
  2. Is the context idle? (else: a turn is already in progress; the existing
     drain logic in executor.py:323-330 will pick up the new notification
     when that turn starts)

If both checks pass, it asks the executor to spawn a fresh A2A task on the
same context. The spawned task runs the same agent pipeline and processes
the drained notifications as the first message of its history.

This module is the implementation of spec 1 (A+B):
docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry

logger = get_logger(__name__)


class _DrainExecutorProtocol(Protocol):
    """Subset of the real executor that the drainer needs.

    Implementations: ObelixAgentExecutor (production), FakeExecutor (tests).
    """

    async def spawn_drain_task(
        self,
        *,
        entry: ContextEntry,
        context_id: str,
        parent_task_id: str | None = None,
    ) -> str: ...


async def maybe_spawn_drain_task(
    *,
    entry: ContextEntry,
    context_id: str,
    executor: _DrainExecutorProtocol,
    parent_task_id: str | None = None,
) -> None:
    """If notifications are pending and no turn is active, spawn a drain task.

    Idempotent w.r.t. its own execution: this function is stateless. Repeated
    invocations with the same inputs return the same decision. Duplicate
    spawning is prevented by the spawned task itself, which calls
    ``entry.idle.clear()`` early — making subsequent invocations short-circuit
    at check 2 below.

    ``parent_task_id`` is forwarded to ``executor.spawn_drain_task`` so the
    drainer can patch ``T_parent.metadata.spawned_task_ids`` for polling
    clients (spec 2 §2). Callers should pass ``entry.current_task_id`` —
    when the parent task already finished and the executor cleared the slot,
    ``None`` here makes the metadata-patch branch safely no-op.
    """
    # Check 1: anything to drain?
    if not entry.pending_notifications:
        return

    # Check 2: a turn already in progress?
    # entry.idle is asyncio.Event; set = idle (ready), clear = busy (turn running).
    if not entry.idle.is_set():
        return

    logger.debug(
        f"[A2A drain] spawning drain task | context_id={context_id} "
        f"pending={len(entry.pending_notifications)} "
        f"parent_task_id={parent_task_id}"
    )
    await executor.spawn_drain_task(
        entry=entry,
        context_id=context_id,
        parent_task_id=parent_task_id,
    )
