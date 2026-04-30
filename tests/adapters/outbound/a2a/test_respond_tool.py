import time
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.adapters.outbound.a2a.tools.respond import RespondToRemoteTool
from obelix.core.model.tool_message import ToolCall, ToolStatus


def _seed_input_required(entry: ContextEntry, *, task_id: str = "t-1") -> None:
    entry.remote_tasks[task_id] = RemoteTaskState(
        task_id=task_id,
        agent_name="B",
        status="input_required",
        token="tok",
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic(),
        last_artifact=None,
        deferred_calls=[
            {"tool_name": "bash", "id": "c-1", "arguments": {"command": "ls"}}
        ],
    )


@pytest.fixture
def registry():
    import httpx

    reg = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    fake_card = MagicMock()
    fake_card.name = "B"
    reg._cards["B"] = fake_card

    fake_client = MagicMock()
    sent = []

    async def _send(message, **kwargs):
        sent.append(message)
        # Generator that does nothing — the client side of a continuation
        # send_message returns void (no new task).
        return
        yield  # pragma: no cover

    fake_client.send_message = _send
    fake_client._sent = sent
    reg._clients["B"] = fake_client
    return reg


@pytest.fixture
def entry():
    e = ContextEntry()
    _seed_input_required(e)
    return e


def _call(args: dict) -> ToolCall:
    return ToolCall(id="c-1", name="respond_to_remote", arguments=args)


@pytest.mark.asyncio
async def test_happy_path_transitions_to_submitted(registry, entry):
    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")

    result = await tool.execute(
        _call({"task_id": "t-1", "data": {"answer": "approve"}})
    )
    assert result.status == ToolStatus.SUCCESS
    assert result.result["task_id"] == "t-1"
    assert result.result["status"] == "submitted"
    assert entry.remote_tasks["t-1"].status == "submitted"
    assert entry.remote_tasks["t-1"].deferred_calls is None


@pytest.mark.asyncio
async def test_unknown_task_returns_error(registry, entry):
    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")

    result = await tool.execute(_call({"task_id": "unknown", "data": {}}))
    assert result.status == ToolStatus.ERROR
    err = (result.error or "").lower()
    assert "not found" in err or "unknown" in err


@pytest.mark.asyncio
async def test_double_respond_blocked_by_idempotency(registry, entry):
    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")
    # First respond — ok
    first = await tool.execute(_call({"task_id": "t-1", "data": {"answer": "ok"}}))
    assert first.status == ToolStatus.SUCCESS
    # Second respond — must error (status now "submitted", not input_required)
    result = await tool.execute(_call({"task_id": "t-1", "data": {"answer": "again"}}))
    assert result.status == ToolStatus.ERROR
    assert "input_required" in (result.error or "")


@pytest.mark.asyncio
async def test_terminal_task_rejected(registry, entry):
    entry.remote_tasks["t-1"].status = "completed"
    entry.remote_tasks["t-1"].deferred_calls = None
    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")
    result = await tool.execute(_call({"task_id": "t-1", "data": {}}))
    assert result.status == ToolStatus.ERROR


@pytest.mark.asyncio
async def test_new_input_required_cycle_allows_respond(registry, entry):
    """After completing one cycle, B emits a NEW input_required → respond again ok."""
    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")

    first = await tool.execute(_call({"task_id": "t-1", "data": {"answer": "ok"}}))
    assert first.status == ToolStatus.SUCCESS

    # Simulate B re-entering input_required for a new deferred tool.
    entry.remote_tasks["t-1"].status = "input_required"
    entry.remote_tasks["t-1"].deferred_calls = [
        {"tool_name": "bash", "id": "c-2", "arguments": {"command": "pwd"}}
    ]
    result = await tool.execute(_call({"task_id": "t-1", "data": {"answer": "again"}}))
    assert result.status == ToolStatus.SUCCESS
    assert entry.remote_tasks["t-1"].status == "submitted"
    assert entry.remote_tasks["t-1"].deferred_calls is None


@pytest.mark.asyncio
async def test_set_context_entry_required(registry):
    tool = RespondToRemoteTool(registry=registry)
    tool.set_webhook_url("http://a:8000/webhook")
    # Did NOT call set_context_entry.
    result = await tool.execute(_call({"task_id": "t-1", "data": {}}))
    assert result.status == ToolStatus.ERROR
    assert "context" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_response_uses_existing_token(registry, entry):
    """The follow-up message must carry the SAME token as the original
    dispatch (so the remote's resume notifications still route)."""
    captured = {}

    async def _capture(message, *, configuration=None, **kwargs):
        captured["configuration"] = configuration
        captured["message"] = message
        return
        yield  # pragma: no cover

    fake_client = MagicMock()
    fake_client.send_message = _capture
    registry._clients["B"] = fake_client

    tool = RespondToRemoteTool(registry=registry)
    tool.set_context_entry(entry)
    tool.set_webhook_url("http://a:8000/webhook")
    await tool.execute(_call({"task_id": "t-1", "data": {"answer": "ok"}}))

    cfg = captured["configuration"]
    assert cfg is not None
    assert cfg.push_notification_config.token == "tok"  # original token, not new one
    msg = captured["message"]
    assert msg.task_id == "t-1"  # continuation tied to the same task
    # The DataPart payload matches what we passed.
    parts = msg.parts
    assert len(parts) == 1
    data_part = parts[0].root
    assert data_part.data == {"answer": "ok"}
