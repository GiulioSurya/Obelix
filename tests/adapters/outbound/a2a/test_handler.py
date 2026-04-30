import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from a2a.types import (
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.outbound.a2a.handler import handle_remote_update
from obelix.adapters.outbound.a2a.state import RemoteTaskState


def _state(status: str = "submitted") -> RemoteTaskState:
    return RemoteTaskState(
        task_id="t-1",
        agent_name="B",
        status=status,
        token="tok",
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic(),
        last_artifact=None,
        deferred_calls=None,
    )


def _build_task(state: TaskState, artifact_text: str | None = None) -> Task:
    artifacts = []
    if artifact_text is not None:
        artifacts = [
            Artifact(
                artifact_id="a-1",
                parts=[Part(root=TextPart(text=artifact_text))],
            )
        ]
    return Task(
        id="t-1",
        context_id="ctx-AAA",
        status=TaskStatus(state=state),
        artifacts=artifacts,
    )


@pytest.fixture
def entry() -> ContextEntry:
    e = ContextEntry()
    e.remote_tasks["t-1"] = _state()
    return e


@pytest.fixture
def registry() -> MagicMock:
    return MagicMock()


def test_completed_emits_notification_and_revokes_token(entry, registry):
    fresh = _build_task(TaskState.completed, artifact_text="Done.")
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)

    assert entry.remote_tasks["t-1"].status == "completed"
    assert len(entry.pending_notifications) == 1
    msg = entry.pending_notifications[0]
    assert "<status>completed</status>" in msg.content
    assert "<result>Done.</result>" in msg.content
    registry.revoke.assert_called_once_with("tok")


def test_working_updates_state_no_notification(entry, registry):
    fresh = _build_task(TaskState.working)
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)

    assert entry.remote_tasks["t-1"].status == "working"
    assert entry.pending_notifications == []
    registry.revoke.assert_not_called()


def test_idempotent_same_state_no_op(entry, registry):
    entry.remote_tasks["t-1"].status = "working"
    fresh = _build_task(TaskState.working)
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)
    assert entry.pending_notifications == []
    registry.revoke.assert_not_called()


def test_input_required_emits_notification_with_deferred(entry, registry):
    from a2a.types import DataPart, Message, Role

    deferred_payload = {
        "deferred_tool_calls": [{"tool_name": "bash", "arguments": {"command": "ls"}}]
    }
    fresh = Task(
        id="t-1",
        context_id="ctx-AAA",
        status=TaskStatus(
            state=TaskState.input_required,
            message=Message(
                role=Role.agent,
                parts=[Part(root=DataPart(data=deferred_payload))],
                message_id="m-1",
            ),
        ),
    )
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)

    assert entry.remote_tasks["t-1"].status == "input_required"
    assert entry.remote_tasks["t-1"].deferred_calls is not None
    assert len(entry.pending_notifications) == 1
    assert "<status>input_required</status>" in entry.pending_notifications[0].content
    # Token NOT revoked on input_required
    registry.revoke.assert_not_called()


def test_failed_emits_notification_with_error(entry, registry):
    from a2a.types import Message, Role

    fresh = Task(
        id="t-1",
        context_id="ctx-AAA",
        status=TaskStatus(
            state=TaskState.failed,
            message=Message(
                role=Role.agent,
                parts=[Part(root=TextPart(text="Database down"))],
                message_id="m-x",
            ),
        ),
    )
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)

    assert entry.remote_tasks["t-1"].status == "failed"
    assert "<status>failed</status>" in entry.pending_notifications[0].content
    assert "Database down" in entry.pending_notifications[0].content
    registry.revoke.assert_called_once_with("tok")


def test_unknown_task_id_no_op(entry, registry):
    fresh = _build_task(TaskState.completed, artifact_text="x")
    handle_remote_update(
        entry=entry, task_id="t-XYZ-unknown", fresh=fresh, registry=registry
    )
    assert entry.pending_notifications == []
    registry.revoke.assert_not_called()


def test_killed_state_locally_drops_late_update(entry, registry):
    """task_stop or context cancel set status='killed'. A late webhook
    must NOT resurrect or re-notify."""
    entry.remote_tasks["t-1"].status = "killed"
    fresh = _build_task(TaskState.completed, artifact_text="late")
    handle_remote_update(entry=entry, task_id="t-1", fresh=fresh, registry=registry)
    assert entry.remote_tasks["t-1"].status == "killed"  # unchanged
    assert entry.pending_notifications == []
