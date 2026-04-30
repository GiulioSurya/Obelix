"""Common code path used by webhook handler and polling worker.

Applies a fresh A2A Task state to the local RemoteTaskState, emits a
<remote_task_update> notification on terminal/input_required states,
and revokes the token on terminal states. Idempotent: re-running with
the same state is a no-op.

Tracer events are NOT emitted here — webhook handler (T7) attaches them
across the HTTP boundary using entry.trace_session before/after this call.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from a2a.types import DataPart, Task, TaskState, TextPart

from obelix.adapters.outbound.a2a.notification import (
    build_remote_task_update_message,
)
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry
    from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry

logger = get_logger(__name__)


_TERMINAL = ("completed", "failed", "canceled", "rejected")


def _state_str(s: TaskState) -> str:
    """Map SDK TaskState enum to our string representation.

    The SDK uses hyphenated values (e.g. ``input-required``) but the rest
    of Obelix (RemoteTaskState.status, notification builder, registry)
    uses the underscore form. Normalize here so callers never have to
    care which side of the boundary they're on.
    """
    raw = s.value if hasattr(s, "value") else str(s)
    return raw.replace("-", "_")


def _extract_artifact_text(task: Task) -> str:
    pieces: list[str] = []
    for art in task.artifacts or []:
        for p in art.parts or []:
            root = p.root
            if isinstance(root, TextPart) and root.text:
                pieces.append(root.text)
    return "".join(pieces)


def _extract_status_message_text(task: Task) -> str:
    """Pull text from status.message.parts (used for failed/rejected reasons)."""
    msg = getattr(task.status, "message", None)
    if msg is None:
        return ""
    pieces: list[str] = []
    for p in msg.parts or []:
        root = p.root
        if isinstance(root, TextPart) and root.text:
            pieces.append(root.text)
    return "".join(pieces)


def _extract_deferred_calls(task: Task) -> list[dict] | None:
    """For input_required, deferred_tool_calls live in status.message.parts[0]
    as a DataPart with shape {deferred_tool_calls: [...]}."""
    msg = getattr(task.status, "message", None)
    if msg is None:
        return None
    for p in msg.parts or []:
        root = p.root
        if isinstance(root, DataPart):
            data = root.data
            if isinstance(data, dict) and "deferred_tool_calls" in data:
                return data["deferred_tool_calls"]
    return None


def handle_remote_update(
    *,
    entry: ContextEntry,
    task_id: str,
    fresh: Task,
    registry: RemoteAgentRegistry,
) -> None:
    """Apply ``fresh`` (a Task observed via webhook or polling) to the
    local ``RemoteTaskState`` and side-effect notifications/token-revoke.

    No-op cases (idempotent): unknown task_id, status unchanged, locally
    killed (status was set by task_stop or context cancel — late updates
    must not resurrect).
    """
    state = entry.remote_tasks.get(task_id)
    if state is None:
        logger.debug(f"[A2A] webhook for unknown task | task_id={task_id}")
        return

    if state.status == "killed":
        logger.debug(
            f"[A2A] dropping late update for locally killed task | task_id={task_id}"
        )
        return

    new_status = _state_str(fresh.status.state)
    if state.status == new_status:
        # Idempotent: same state, no notification.
        return

    # Apply update.
    state.status = new_status
    state.last_update = datetime.now(UTC)
    state.last_update_monotonic = time.monotonic()
    state.poll_failures = 0  # any successful update resets the streak

    if new_status in _TERMINAL:
        # Build notification with result OR error.
        if new_status == "completed":
            text = _extract_artifact_text(fresh)
            state.last_artifact = (
                {"text": text} if text else None
            )  # store something readable for task_get
            note = build_remote_task_update_message(
                task_id=task_id,
                agent_name=state.agent_name,
                status="completed",
                result_text=text,
            )
        else:
            err = _extract_status_message_text(fresh) or new_status
            note = build_remote_task_update_message(
                task_id=task_id,
                agent_name=state.agent_name,
                status=new_status,
                error_text=err,
            )
        entry.pending_notifications.append(note)
        registry.revoke(state.token)
        return

    if new_status == "input_required":
        deferred = _extract_deferred_calls(fresh) or []
        state.deferred_calls = deferred
        note = build_remote_task_update_message(
            task_id=task_id,
            agent_name=state.agent_name,
            status="input_required",
            deferred_calls=deferred,
        )
        entry.pending_notifications.append(note)
        return

    # Intermediate states (working/submitted): state already updated above,
    # no notification emitted.
