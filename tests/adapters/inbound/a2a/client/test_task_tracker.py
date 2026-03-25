"""Tests for TaskTracker.force_terminal() method.

Covers:
- force_terminal sets state on existing task
- force_terminal on unknown task is a no-op
- is_terminal becomes True after force_terminal with terminal states
"""

from __future__ import annotations

import pytest

from obelix.adapters.inbound.a2a.client.webhook_server import TaskInfo, TaskTracker


class TestForceTerminal:
    @pytest.mark.asyncio
    async def test_force_terminal_sets_state(self):
        """force_terminal() updates the task state."""
        tracker = TaskTracker()
        await tracker.register("task-1", "agent-a")

        tracker.force_terminal("task-1", "canceled")

        info = tracker.get("task-1")
        assert info is not None
        assert info.state == "canceled"

    @pytest.mark.asyncio
    async def test_force_terminal_makes_task_terminal(self):
        """After force_terminal with a terminal state, is_terminal is True."""
        tracker = TaskTracker()
        await tracker.register("task-2", "agent-b")

        tracker.force_terminal("task-2", "canceled")

        info = tracker.get("task-2")
        assert info.is_terminal is True

    @pytest.mark.asyncio
    async def test_force_terminal_unknown_task_is_noop(self):
        """force_terminal on non-existent task does not raise."""
        tracker = TaskTracker()
        # Should not raise
        tracker.force_terminal("nonexistent", "canceled")

    @pytest.mark.asyncio
    async def test_force_terminal_completed_state(self):
        """force_terminal with 'completed' sets terminal state."""
        tracker = TaskTracker()
        await tracker.register("task-3", "agent-c")

        tracker.force_terminal("task-3", "completed")

        info = tracker.get("task-3")
        assert info.state == "completed"
        assert info.is_terminal is True

    @pytest.mark.asyncio
    async def test_force_terminal_failed_state(self):
        """force_terminal with 'failed' sets terminal state."""
        tracker = TaskTracker()
        await tracker.register("task-4", "agent-d")

        tracker.force_terminal("task-4", "failed")

        info = tracker.get("task-4")
        assert info.state == "failed"
        assert info.is_terminal is True

    @pytest.mark.asyncio
    async def test_force_terminal_updates_timestamp(self):
        """force_terminal updates the timestamp."""
        tracker = TaskTracker()
        await tracker.register("task-5", "agent-e")

        info_before = tracker.get("task-5")
        ts_before = info_before.timestamp

        # Small delay to ensure timestamp changes
        import time

        time.sleep(0.01)

        tracker.force_terminal("task-5", "canceled")

        info_after = tracker.get("task-5")
        assert info_after.timestamp >= ts_before

    @pytest.mark.asyncio
    async def test_force_terminal_removes_from_active(self):
        """After force_terminal, task no longer appears in get_active()."""
        tracker = TaskTracker()
        await tracker.register("task-6", "agent-f")

        # Before: task is active (state='submitted')
        active_before = tracker.get_active()
        assert any(t.task_id == "task-6" for t in active_before)

        tracker.force_terminal("task-6", "canceled")

        # After: task is no longer active
        active_after = tracker.get_active()
        assert not any(t.task_id == "task-6" for t in active_after)


class TestTaskInfoIsTerminal:
    @pytest.mark.parametrize(
        "state,expected",
        [
            ("completed", True),
            ("failed", True),
            ("canceled", True),
            ("rejected", True),
            ("working", False),
            ("submitted", False),
            ("input-required", False),
        ],
        ids=[
            "completed",
            "failed",
            "canceled",
            "rejected",
            "working",
            "submitted",
            "input-required",
        ],
    )
    def test_is_terminal_for_various_states(self, state: str, expected: bool):
        info = TaskInfo(task_id="t", agent_name="a", state=state)
        assert info.is_terminal is expected
