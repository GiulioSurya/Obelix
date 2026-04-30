from unittest.mock import MagicMock

import pytest
from a2a.types import Task, TaskState, TaskStatus

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.tools.dispatch import DispatchAgentTool
from obelix.core.model.tool_message import ToolCall, ToolStatus


@pytest.fixture
def registry_with_b():
    import httpx

    reg = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    fake_card = MagicMock()
    fake_card.name = "B"
    fake_card.description = "inventory"
    skill = MagicMock()
    skill.name = "lookup"
    fake_card.skills = [skill]
    reg._cards["B"] = fake_card

    fake_client = MagicMock()

    async def _send_message(message, **kwargs):
        # Yield a single tuple (Task, None) like a non-streaming SDK call.
        task = Task(
            id="t-001",
            context_id="ctx-remote",
            status=TaskStatus(state=TaskState.submitted),
        )
        yield (task, None)

    fake_client.send_message = _send_message
    reg._clients["B"] = fake_client
    return reg


@pytest.fixture
def entry() -> ContextEntry:
    e = ContextEntry()
    return e


def _make_call(args: dict) -> ToolCall:
    return ToolCall(id="c-1", name="dispatch_agent", arguments=args)


@pytest.mark.asyncio
async def test_dispatch_happy_path(registry_with_b, entry):
    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")

    result = await tool.execute(_make_call({"agent_name": "B", "query": "do X"}))

    assert result.status == ToolStatus.SUCCESS
    assert result.result["status"] == "submitted"
    assert result.result["task_id"] == "t-001"
    assert result.result["agent"] == "B"

    # State recorded in context.
    assert "t-001" in entry.remote_tasks
    state = entry.remote_tasks["t-001"]
    assert state.agent_name == "B"
    assert state.status == "submitted"
    # Token registered before send_message; lookup must work.
    route = registry_with_b.lookup(state.token)
    assert route is not None
    assert route.context_id == "ctx-MARIO"
    assert route.task_id == "t-001"


@pytest.mark.asyncio
async def test_dispatch_unknown_agent_returns_error(registry_with_b, entry):
    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")

    result = await tool.execute(
        _make_call({"agent_name": "DoesNotExist", "query": "do X"})
    )
    assert result.status == ToolStatus.ERROR
    assert "DoesNotExist" in (result.error or "")
    # No state added.
    assert entry.remote_tasks == {}


@pytest.mark.asyncio
async def test_dispatch_send_message_failure_cleans_token(registry_with_b, entry):
    fake_client = MagicMock()

    async def _boom(message, **kwargs):
        raise RuntimeError("network down")
        yield  # pragma: no cover

    fake_client.send_message = _boom
    registry_with_b._clients["B"] = fake_client

    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")
    result = await tool.execute(_make_call({"agent_name": "B", "query": "do X"}))

    assert result.status == ToolStatus.ERROR
    assert "network down" in (result.error or "")
    # No leftover token in registry.
    assert len(registry_with_b._token_map) == 0
    assert entry.remote_tasks == {}


@pytest.mark.asyncio
async def test_dispatch_never_returns_none_result(registry_with_b, entry):
    """Critical: the tool must NEVER return a None result, since that would
    falsely trigger BaseAgent's deferred-tool detection."""
    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")
    result = await tool.execute(_make_call({"agent_name": "B", "query": "x"}))
    assert result.result is not None


def test_system_prompt_fragment_lists_remotes(registry_with_b):
    tool = DispatchAgentTool(registry=registry_with_b)
    fragment = tool.system_prompt_fragment()
    assert "B" in fragment
    assert "inventory" in fragment
    assert "<remote_task_update>" in fragment


def test_system_prompt_fragment_returns_none_when_no_remotes():
    import httpx

    reg = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    tool = DispatchAgentTool(registry=reg)
    assert tool.system_prompt_fragment() is None


@pytest.mark.asyncio
async def test_set_context_entry_required_for_execute(registry_with_b):
    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_webhook_url("http://a:8000/webhook")
    # No set_context_entry called.
    result = await tool.execute(_make_call({"agent_name": "B", "query": "x"}))
    assert result.status == ToolStatus.ERROR
    assert "context" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_set_webhook_url_required_for_execute(registry_with_b, entry):
    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    # No set_webhook_url called.
    result = await tool.execute(_make_call({"agent_name": "B", "query": "x"}))
    assert result.status == ToolStatus.ERROR
    assert "webhook" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_dispatch_passes_token_in_push_config(registry_with_b, entry):
    """Verify the PushNotificationConfig.token sent to the remote matches
    what we register in the local token map."""
    captured_config = {}

    async def _capture_send(message, *, configuration=None, **kwargs):
        captured_config["configuration"] = configuration
        task = Task(
            id="t-001",
            context_id="ctx-remote",
            status=TaskStatus(state=TaskState.submitted),
        )
        yield (task, None)

    fake_client = MagicMock()
    fake_client.send_message = _capture_send
    registry_with_b._clients["B"] = fake_client

    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")
    await tool.execute(_make_call({"agent_name": "B", "query": "x"}))

    cfg = captured_config["configuration"]
    assert cfg is not None
    assert cfg.push_notification_config is not None
    assert cfg.push_notification_config.url == "http://a:8000/webhook"
    sent_token = cfg.push_notification_config.token
    # The same token that was sent to the remote must be in the local
    # token map (so the inbound webhook with that token routes correctly).
    assert registry_with_b.lookup(sent_token) is not None
    assert entry.remote_tasks["t-001"].token == sent_token


@pytest.mark.asyncio
async def test_dispatch_direct_message_reply(registry_with_b, entry):
    """Some simple agents return a Message directly instead of a Task tuple.
    Treat this as immediate completion: revoke token, extract text content,
    no state inserted."""
    from a2a.types import Message as A2AMessage
    from a2a.types import Role

    async def _direct_reply(message, **kwargs):
        from a2a.types import Part, TextPart

        yield A2AMessage(
            message_id="m-1",
            role=Role.agent,
            parts=[Part(root=TextPart(text="immediate answer"))],
        )

    fake_client = MagicMock()
    fake_client.send_message = _direct_reply
    registry_with_b._clients["B"] = fake_client

    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")
    result = await tool.execute(_make_call({"agent_name": "B", "query": "x"}))

    assert result.status == ToolStatus.SUCCESS
    assert result.result["status"] == "completed"
    assert result.result["agent"] == "B"
    assert result.result["result"] == "immediate answer"
    # Token revoked, no state in remote_tasks
    assert len(registry_with_b._token_map) == 0
    assert entry.remote_tasks == {}


@pytest.mark.asyncio
async def test_dispatch_cancelled_error_revokes_token(registry_with_b, entry):
    """If the asyncio task is cancelled mid-dispatch, the token must still
    be revoked. We verify this via direct internal-state inspection after
    the cancellation propagates through (or is converted by the wrapper)."""
    import asyncio

    async def _hang(message, **kwargs):
        raise asyncio.CancelledError()
        yield  # pragma: no cover

    fake_client = MagicMock()
    fake_client.send_message = _hang
    registry_with_b._clients["B"] = fake_client

    tool = DispatchAgentTool(registry=registry_with_b)
    tool.set_context_entry(entry, context_id="ctx-MARIO")
    tool.set_webhook_url("http://a:8000/webhook")

    # CancelledError is caught by dispatch.py's own `except BaseException`
    # block first — that's where the token gets revoked, before re-raising.
    # The decorator's wrapped_execute uses `except Exception` and so does NOT
    # catch CancelledError (BaseException-class), letting it propagate up to
    # us. Catch it here to verify the token cleanup happened upstream.
    try:
        await tool.execute(_make_call({"agent_name": "B", "query": "x"}))
    except asyncio.CancelledError:
        pass

    # The critical assertion: token revoked despite BaseException-class cancel.
    assert len(registry_with_b._token_map) == 0
    assert entry.remote_tasks == {}
