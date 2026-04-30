"""Single global polling worker that scans every ContextEntry for
non-terminal remote tasks whose last_update is stale (>30s) and falls
back to client.get_task() if the webhook never arrived.

Lifecycle: created in AgentFactory.a2a_serve when remote_agents is
non-empty. Started via FastAPI startup event, stopped via shutdown event.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from a2a.types import TaskQueryParams

from obelix.adapters.outbound.a2a.handler import handle_remote_update
from obelix.adapters.outbound.a2a.notification import (
    build_remote_task_update_message,
)
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
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
        tick_seconds: float = 5.0,
    ) -> None:
        self._registry = registry
        self._store = context_store
        self._tick = tick_seconds
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
