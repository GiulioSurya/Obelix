import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.adapters.outbound.a2a.tools.task_ops import (
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from obelix.core.model.tool_message import ToolCall, ToolStatus


def _state(task_id: str, status: str, *, age_minutes: int = 0) -> RemoteTaskState:
    return RemoteTaskState(
        task_id=task_id,
        agent_name="B",
        status=status,
        token=f"tok-{task_id}",
        created_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
        last_update=datetime.now(UTC) - timedelta(minutes=age_minutes),
        last_update_monotonic=time.monotonic() - age_minutes * 60,
        last_artifact={"text": "hi"} if status == "completed" else None,
        deferred_calls=None,
    )


@pytest.fixture
def entry():
    e = ContextEntry()
    e.remote_tasks["t-1"] = _state("t-1", "completed", age_minutes=10)
    e.remote_tasks["t-2"] = _state("t-2", "working", age_minutes=5)
    e.remote_tasks["t-3"] = _state("t-3", "failed", age_minutes=1)
    return e


@pytest.fixture
def registry():
    import httpx

    return RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())


def _call(name: str, args: dict) -> ToolCall:
    return ToolCall(id="c-1", name=name, arguments=args)


# ── task_list ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_list_returns_all_states(entry):
    tool = TaskListTool()
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_list", {}))
    assert result.status == ToolStatus.SUCCESS
    statuses = {t["status"] for t in result.result["tasks"]}
    assert statuses == {"completed", "working", "failed"}


@pytest.mark.asyncio
async def test_task_list_sorted_recent_first(entry):
    tool = TaskListTool()
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_list", {}))
    ids = [t["task_id"] for t in result.result["tasks"]]
    # t-3 is most recent (1 min ago), t-1 is oldest (10 min ago)
    assert ids == ["t-3", "t-2", "t-1"]


@pytest.mark.asyncio
async def test_task_list_respects_limit(entry):
    tool = TaskListTool()
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_list", {"limit": 2}))
    assert result.result["shown"] == 2
    assert result.result["total"] == 3
    assert len(result.result["tasks"]) == 2


@pytest.mark.asyncio
async def test_task_list_empty(entry):
    tool = TaskListTool()
    tool.set_context_entry(ContextEntry())
    result = await tool.execute(_call("task_list", {}))
    assert result.result["tasks"] == []
    assert result.result["shown"] == 0
    assert result.result["total"] == 0


# ── task_get ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_get_unknown_returns_error(entry):
    tool = TaskGetTool()
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_get", {"task_id": "nope"}))
    assert result.status == ToolStatus.ERROR


@pytest.mark.asyncio
async def test_task_get_returns_full_state(entry):
    tool = TaskGetTool()
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_get", {"task_id": "t-1"}))
    assert result.status == ToolStatus.SUCCESS
    payload = result.result
    assert payload["task_id"] == "t-1"
    assert payload["status"] == "completed"
    assert payload["last_artifact"] == {"text": "hi"}


# ── task_stop ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task_stop_flips_state_revokes_token_no_wire_call(entry, registry):
    registry.register_token("tok-t-2", context_id="ctx", agent_name="B")
    fake_client = MagicMock()
    registry._clients["B"] = fake_client  # to verify cancel_task NOT called

    tool = TaskStopTool(registry=registry)
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_stop", {"task_id": "t-2"}))

    assert result.status == ToolStatus.SUCCESS
    assert entry.remote_tasks["t-2"].status == "killed"
    assert registry.lookup("tok-t-2") is None
    fake_client.cancel_task.assert_not_called()


@pytest.mark.asyncio
async def test_task_stop_unknown_returns_error(entry, registry):
    tool = TaskStopTool(registry=registry)
    tool.set_context_entry(entry)
    result = await tool.execute(_call("task_stop", {"task_id": "nope"}))
    assert result.status == ToolStatus.ERROR


@pytest.mark.asyncio
async def test_task_stop_idempotent_on_already_terminal(entry, registry):
    tool = TaskStopTool(registry=registry)
    tool.set_context_entry(entry)
    # t-1 is already 'completed' — stop should be a no-op success.
    result = await tool.execute(_call("task_stop", {"task_id": "t-1"}))
    assert result.status == ToolStatus.SUCCESS
    assert entry.remote_tasks["t-1"].status == "completed"  # unchanged
    # Result indicates no-op
    assert result.result.get("noop") is True


@pytest.mark.asyncio
async def test_task_list_set_context_entry_required():
    tool = TaskListTool()
    # No set_context_entry called.
    result = await tool.execute(_call("task_list", {}))
    assert result.status == ToolStatus.ERROR
    assert "context" in (result.error or "").lower()
