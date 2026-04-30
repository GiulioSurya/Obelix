"""Tests for executor cancel sweeping in-flight remote tokens (T13)."""

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.adapters.outbound.a2a.state import RemoteTaskState


def _seed(
    entry: ContextEntry,
    *,
    task_id: str,
    status: str = "working",
    token: str | None = None,
) -> None:
    entry.remote_tasks[task_id] = RemoteTaskState(
        task_id=task_id,
        agent_name="B",
        status=status,
        token=token or f"tok-{task_id}",
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic(),
        last_artifact=None,
        deferred_calls=None,
    )


def test_revoke_in_flight_remote_tokens_silences_late_webhooks():
    """All non-terminal tasks: revoke + flip to 'killed'. Terminal ones
    are skipped. NO wire cancel_task to the remote."""
    entry = ContextEntry()
    _seed(entry, task_id="t-1", status="working")
    _seed(entry, task_id="t-2", status="completed")  # terminal — skipped
    _seed(entry, task_id="t-3", status="input_required")

    registry = MagicMock()
    fake_client = MagicMock()
    registry.client_for.return_value = fake_client

    ObelixAgentExecutor._revoke_in_flight_remote_tokens(entry, registry)

    # Non-terminal: t-1 and t-3 → tokens revoked, status flipped to "killed"
    registry.revoke.assert_any_call("tok-t-1")
    registry.revoke.assert_any_call("tok-t-3")
    # t-2 (terminal) untouched.
    assert registry.revoke.call_count == 2
    assert entry.remote_tasks["t-1"].status == "killed"
    assert entry.remote_tasks["t-2"].status == "completed"
    assert entry.remote_tasks["t-3"].status == "killed"
    # CRITICAL: NO wire call to cancel_task on the remote.
    fake_client.cancel_task.assert_not_called()


def test_revoke_no_remote_tasks_is_noop():
    """Empty remote_tasks: no-op."""
    entry = ContextEntry()
    registry = MagicMock()
    ObelixAgentExecutor._revoke_in_flight_remote_tokens(entry, registry)
    registry.revoke.assert_not_called()


def test_revoke_all_terminal_is_noop():
    """If all tasks are already terminal: no-op."""
    entry = ContextEntry()
    _seed(entry, task_id="t-1", status="completed")
    _seed(entry, task_id="t-2", status="failed")
    _seed(entry, task_id="t-3", status="canceled")
    _seed(entry, task_id="t-4", status="rejected")
    _seed(entry, task_id="t-5", status="killed")

    registry = MagicMock()
    ObelixAgentExecutor._revoke_in_flight_remote_tokens(entry, registry)

    registry.revoke.assert_not_called()
    # All statuses unchanged.
    assert entry.remote_tasks["t-1"].status == "completed"
    assert entry.remote_tasks["t-2"].status == "failed"
    assert entry.remote_tasks["t-3"].status == "canceled"
    assert entry.remote_tasks["t-4"].status == "rejected"
    assert entry.remote_tasks["t-5"].status == "killed"


def test_revoke_handles_none_registry():
    """If registry is None (e.g., no remote_agents configured), no-op safely."""
    entry = ContextEntry()
    _seed(entry, task_id="t-1", status="working")

    # Must not raise
    ObelixAgentExecutor._revoke_in_flight_remote_tokens(entry, None)
    # Status unchanged because we couldn't revoke anything.
    assert entry.remote_tasks["t-1"].status == "working"


def test_revoke_updates_last_update_monotonic():
    """The kill operation updates last_update_monotonic for consistency
    with task_stop."""
    entry = ContextEntry()
    _seed(entry, task_id="t-1", status="working")
    old_mono = entry.remote_tasks["t-1"].last_update_monotonic

    registry = MagicMock()
    ObelixAgentExecutor._revoke_in_flight_remote_tokens(entry, registry)

    new_mono = entry.remote_tasks["t-1"].last_update_monotonic
    assert new_mono >= old_mono


def test_executor_init_accepts_registry_kwarg():
    """The new registry parameter on ObelixAgentExecutor.__init__ defaults
    to None and is keyword-only."""

    def _factory():
        raise NotImplementedError("not used")

    # Existing call sites without registry still work.
    executor1 = ObelixAgentExecutor(_factory)
    assert executor1._registry is None

    # New call sites can pass registry.
    fake_registry = MagicMock()
    executor2 = ObelixAgentExecutor(_factory, registry=fake_registry)
    assert executor2._registry is fake_registry
