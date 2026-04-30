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
# SDK intermediate states that we deliberately don't notify on.
# "killed" is a local sentinel set by task_stop / context cancel and
# never arrives from the SDK, so it's intentionally absent here.
_KNOWN_INTERMEDIATE = ("working", "submitted")


def _state_str(s: TaskState) -> str:
    """Map SDK TaskState enum to Obelix's string representation.

    The A2A SDK uses hyphenated enum values (e.g. "input-required",
    "auth-required") while Obelix uses the underscore form everywhere
    else (RemoteTaskState.status, _VALID_STATUSES, etc.). Normalizing
    here keeps the boundary contained.

    Note: SDK states `auth_required` and `unknown` normalize but fall
    through as intermediate states — no notification is emitted for
    them. If a future task needs explicit handling, extend the dispatch
    branches in handle_remote_update.
    """
    return s.value.replace("-", "_")


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

    **No-op cases** (idempotent): unknown task_id, status unchanged,
    locally killed (status was set by task_stop or context cancel — late
    updates must not resurrect).

    **Active cases**:
    - Terminal states (completed/failed/canceled/rejected): build a
      <remote_task_update> notification (with <result> for completed,
      <error> otherwise), append to entry.pending_notifications, revoke
      the token on the registry.
    - input_required: extract deferred_tool_calls from status.message,
      set state.deferred_calls, append a notification, do NOT revoke
      the token (the task may receive a follow-up via respond_to_remote).
    - Intermediate states (working/submitted/auth_required/unknown):
      state is updated silently, no notification, no revoke.

    **poll_failures contract**: This function resets ``state.poll_failures
    = 0`` on every state change. Same-state polls (e.g., a polling worker
    observing repeated "working") are early-returned BEFORE this reset, so
    ``poll_failures`` is NOT touched. Callers that count consecutive HTTP
    errors (the polling worker, T8) must reset ``poll_failures`` themselves
    on every successful HTTP response, regardless of whether the state
    changed. This function only sees the parsed Task; it cannot tell whether
    the HTTP call that produced it succeeded.

    **Thread safety**: Not thread-safe. Callers must serialize access to
    ``entry.remote_tasks`` and ``entry.pending_notifications`` (typically
    the executor's per-context idle gate already enforces this).
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

    if (
        new_status not in _TERMINAL
        and new_status != "input_required"
        and new_status not in _KNOWN_INTERMEDIATE
    ):
        logger.warning(
            f"[A2A] unknown remote state | task_id={task_id} "
            f"agent={state.agent_name} status={new_status} — "
            f"state updated but no notification emitted; if this state "
            f"requires LLM action, extend handle_remote_update."
        )
        # Fall through to the end (no notification, no revoke).

    if new_status in _TERMINAL:
        # Build notification with result OR error.
        if new_status == "completed":
            text = _extract_artifact_text(fresh)
            state.last_artifact = {"text": text} if text else None
            # text or None: avoid emitting an empty <result></result>
            # tag when the remote completes without text artifacts.
            note = build_remote_task_update_message(
                task_id=task_id,
                agent_name=state.agent_name,
                status="completed",
                result_text=text or None,
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
