"""In-memory state for outbound A2A dispatches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(eq=False, slots=True)
class RemoteTaskState:
    """State of one outbound dispatch tracked by an agent context.

    Lives inside ContextEntry.remote_tasks. Updated by webhook handler and
    polling worker. status is the most recent A2A state observed.
    """

    task_id: str
    agent_name: str
    status: str  # submitted | working | input_required | completed | failed | canceled | rejected | killed
    token: str
    created_at: datetime
    last_update: datetime
    last_update_monotonic: float
    last_artifact: dict | None
    deferred_calls: list[dict] | None
    poll_failures: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "failed", "canceled", "rejected", "killed")


@dataclass(slots=True)
class TokenRoute:
    """Routing entry: webhook arrives with token → resolve to context+task.

    No asyncio.Event. The race "webhook before send_message returns" is
    handled by the webhook handler using body.id as fallback when task_id
    is still None.
    """

    context_id: str
    agent_name: str
    task_id: str | None
    registered_at: datetime
