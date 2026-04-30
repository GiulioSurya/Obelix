"""task_list / task_get / task_stop — LLM tools to inspect and manage
the parent agent's view of remote tasks."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import Field

from obelix.adapters.outbound.a2a.tools._base import _ContextAware
from obelix.core.tool.tool_decorator import tool
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry

logger = get_logger(__name__)


@tool(
    name="task_list",
    description=(
        "List remote tasks dispatched from this conversation. Returns all "
        "states (working, completed, failed, etc.) sorted by most recent "
        "update first. Use this to check progress or see history. "
        "Bounded by `limit` (default 50, max 500) to avoid context blowup."
    ),
    is_deferred=False,
    read_only=True,
)
class TaskListTool(_ContextAware):
    limit: int = Field(default=50, ge=1, le=500)

    async def execute(self) -> dict:
        ctx_entry = self._require_context("TaskListTool")

        all_states = sorted(
            ctx_entry.remote_tasks.values(),
            key=lambda s: s.last_update_monotonic,
            reverse=True,
        )
        shown = all_states[: self.limit]
        return {
            "tasks": [
                {
                    "task_id": s.task_id,
                    "agent": s.agent_name,
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                    "last_update": s.last_update.isoformat(),
                }
                for s in shown
            ],
            "shown": len(shown),
            "total": len(all_states),
        }


@tool(
    name="task_get",
    description="Get full details of one remote task by its task_id.",
    is_deferred=False,
    read_only=True,
)
class TaskGetTool(_ContextAware):
    task_id: str = Field(...)

    async def execute(self) -> dict:
        ctx_entry = self._require_context("TaskGetTool")
        s = ctx_entry.remote_tasks.get(self.task_id)
        if s is None:
            raise ValueError(f"task {self.task_id!r} not found")
        return {
            "task_id": s.task_id,
            "agent": s.agent_name,
            "status": s.status,
            "created_at": s.created_at.isoformat(),
            "last_update": s.last_update.isoformat(),
            "last_artifact": s.last_artifact,
            "deferred_calls": s.deferred_calls,
        }


@tool(
    name="task_stop",
    description=(
        "Stop tracking a remote task locally. Flips its local status to "
        "'killed' and silences future webhook updates for that task. Does "
        "NOT send a cancel to the remote agent — they keep working. "
        "Idempotent on already-terminal tasks (returns {noop: true})."
    ),
    is_deferred=False,
)
class TaskStopTool(_ContextAware):
    task_id: str = Field(...)

    def __init__(self, registry: RemoteAgentRegistry) -> None:
        self._registry = registry

    async def execute(self) -> dict:
        ctx_entry = self._require_context("TaskStopTool")
        s = ctx_entry.remote_tasks.get(self.task_id)
        if s is None:
            raise ValueError(f"task {self.task_id!r} not found")
        if s.is_terminal:
            return {"task_id": s.task_id, "status": s.status, "noop": True}

        self._registry.revoke(s.token)
        s.status = "killed"
        # Update both timestamps together — same contract as handler.py
        # and dispatch.py: last_update and last_update_monotonic must be
        # written as a pair on every state change.
        s.last_update = datetime.now(UTC)
        s.last_update_monotonic = time.monotonic()
        logger.info(
            f"[A2A task_stop] killed locally | task_id={s.task_id} agent={s.agent_name}"
        )
        return {"task_id": s.task_id, "status": "killed"}
