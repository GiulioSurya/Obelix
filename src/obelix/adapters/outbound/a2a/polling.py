"""Single global polling worker that scans every ContextEntry for
non-terminal remote tasks whose last_update is stale (>30s) and falls
back to client.get_task() if the webhook never arrived.

Lifecycle: created in AgentFactory.a2a_serve when remote_agents is
non-empty. Started via FastAPI startup event, stopped via shutdown event.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from a2a.types import TaskQueryParams

from obelix.adapters.outbound.a2a.handler import handle_remote_update
from obelix.adapters.outbound.a2a.notification import (
    build_remote_task_update_message,
)
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.tasks.task_store import TaskStore

    from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
    from obelix.adapters.inbound.a2a.server.drainer import _DrainExecutorProtocol
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
    from obelix.adapters.outbound.a2a.state import RemoteTaskState

logger = get_logger(__name__)


_STALE_AFTER_SECONDS = 30.0
_MAX_FAILURES = 5


class PollingWorker:
    """Single global polling worker. Iterates all contexts every tick."""

    def __init__(
        self,
        *,
        registry: RemoteAgentRegistry,
        context_store: ContextStore,
        executor: _DrainExecutorProtocol | None = None,
        tick_seconds: float = 5.0,
        task_store: TaskStore | None = None,
    ) -> None:
        self._registry = registry
        self._store = context_store
        # When set, ``_poll_one`` awaits ``maybe_spawn_drain_task`` after
        # ``handle_remote_update`` so that pending notifications produced by
        # terminal/input_required state changes trigger a fresh A2A turn on
        # the same context (spec 1, drainer component). The drainer is
        # idempotent and short-circuits when the queue is empty or a turn is
        # already running, so passing the executor unconditionally is safe.
        self._executor = executor
        self._tick = tick_seconds
        # Optional SDK TaskStore — when wired, every observed peer-state
        # change is mirrored onto T_parent.metadata.dispatched_peers so the
        # CLI status bar (which polls T1.metadata) stays in sync with
        # entry.remote_tasks. Defaults to None for backwards compatibility
        # with existing tests that exercise polling in isolation.
        self._task_store = task_store
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Start the background polling loop."""
        self._stop.clear()
        self._task = asyncio.create_task(self._loop(), name="a2a-polling-worker")
        logger.info("[A2A] polling worker started")

    async def stop(self) -> None:
        """Signal stop and await the loop to exit."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[A2A] polling worker stopped")

    async def _loop(self) -> None:
        # belt-and-suspenders: cancel() is the primary stop signal (loop
        # exits on CancelledError at the sleep). _stop guards against the
        # rare path where stop() is called before _task is set.
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self._tick)
                await self._tick_once()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"[A2A polling] tick failed | error={e}")

    async def _tick_once(self) -> None:
        """One pass: scan every ContextEntry's remote_tasks for stale
        non-terminal tasks and poll them. Public for tests."""
        now = time.monotonic()
        # Snapshot iteration to avoid mutating the dict while looping.
        for ctx_entry in self._store.iter_entries():
            for state in list(ctx_entry.remote_tasks.values()):
                if state.is_terminal:
                    continue
                if now - state.last_update_monotonic < _STALE_AFTER_SECONDS:
                    continue
                await self._poll_one(ctx_entry, state)

    async def _poll_one(self, ctx_entry: ContextEntry, state: RemoteTaskState) -> None:
        """Poll a single task. On HTTP success, feed through handle_remote_update
        (which resets poll_failures only on state change). On HTTP success with
        unchanged state, reset poll_failures explicitly. On HTTP failure,
        increment poll_failures; on 5 consecutive failures, mark failed
        locally + emit notification + revoke token."""
        try:
            client = self._registry.client_for(state.agent_name)
            fresh = await client.get_task(TaskQueryParams(id=state.task_id))
        except Exception as e:
            state.poll_failures += 1
            logger.debug(
                f"[A2A polling] get_task failed | task_id={state.task_id} "
                f"failures={state.poll_failures} error={e}"
            )
            if state.poll_failures >= _MAX_FAILURES:
                state.status = "failed"
                # Pair last_update with last_update_monotonic — same contract
                # as handler.py, dispatch.py, task_ops.py, and executor.py:
                # both must be written together so task_list/task_get don't
                # surface a stale wall-clock timestamp to the LLM.
                state.last_update = datetime.now(UTC)
                state.last_update_monotonic = time.monotonic()
                ctx_entry.pending_notifications.append(
                    build_remote_task_update_message(
                        task_id=state.task_id,
                        agent_name=state.agent_name,
                        status="failed",
                        error_text="polling_giveup",
                    )
                )
                self._registry.revoke(state.token)
                # Mirror to T_parent.metadata so the CLI status bar shows
                # the failed peer immediately rather than freezing on
                # "working" until the next manual refresh.
                if (
                    self._task_store is not None
                    and ctx_entry.current_task_id is not None
                ):
                    from obelix.adapters.inbound.a2a.server.metadata_patch import (
                        update_dispatched_peer_state,
                    )

                    await update_dispatched_peer_state(
                        self._task_store,
                        ctx_entry.current_task_id,
                        state.task_id,
                        "failed",
                    )
                logger.warning(
                    f"[A2A polling] giveup | task_id={state.task_id} "
                    f"after {_MAX_FAILURES} consecutive failures"
                )
            return

        # SDK returns None when the task_id is not found on the remote
        # (HTTP 404-equivalent). Treat as "no update" and try again next
        # tick; do NOT increment poll_failures.
        if fresh is None:
            return

        # HTTP success: reset poll_failures BEFORE delegating, since
        # handle_remote_update only resets on state CHANGE (its early-return
        # for same-state preserves the counter).
        state.poll_failures = 0
        # Delegate to the same handler the webhook uses. Idempotency,
        # notification building, and termination handling all live there.
        handle_remote_update(
            entry=ctx_entry,
            task_id=state.task_id,
            fresh=fresh,
            registry=self._registry,
        )

        # Mirror the (possibly new) peer state onto T_parent.metadata.
        # ``handle_remote_update`` is idempotent — same-state polls early-return
        # before mutating, but reading state.status after the call still gives
        # the canonical post-update value. The CLI status bar polls this
        # metadata to render one segment per active peer.
        if self._task_store is not None and ctx_entry.current_task_id is not None:
            from obelix.adapters.inbound.a2a.server.metadata_patch import (
                update_dispatched_peer_state,
            )

            mirrored_state = (
                ctx_entry.remote_tasks[state.task_id].status
                if state.task_id in ctx_entry.remote_tasks
                else None
            )
            if mirrored_state is not None:
                await update_dispatched_peer_state(
                    self._task_store,
                    ctx_entry.current_task_id,
                    state.task_id,
                    mirrored_state,
                )

        # Drainer: kick a fresh A2A turn if the update produced a pending
        # notification AND the context is idle. ``maybe_spawn_drain_task``
        # itself enforces both checks, so calling it unconditionally here
        # (when an executor is wired) is safe and idempotent. The
        # context_id is recovered from ``ctx_entry.context_id`` (populated
        # by ``ContextStore.get_or_create``) — the iter_entries snapshot
        # itself doesn't carry it.
        if self._executor is not None and ctx_entry.context_id is not None:
            from obelix.adapters.inbound.a2a.server.drainer import (
                maybe_spawn_drain_task,
            )

            await maybe_spawn_drain_task(
                entry=ctx_entry,
                context_id=ctx_entry.context_id,
                executor=self._executor,
                parent_task_id=ctx_entry.current_task_id,
            )
