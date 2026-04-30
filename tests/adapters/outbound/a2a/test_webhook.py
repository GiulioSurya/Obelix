import time
from datetime import UTC, datetime

import pytest
from starlette.requests import Request

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.adapters.outbound.a2a.webhook import make_webhook_handler


def _seed_state(entry: ContextEntry, task_id: str = "t-1", token: str = "tok") -> None:
    entry.remote_tasks[task_id] = RemoteTaskState(
        task_id=task_id,
        agent_name="B",
        status="submitted",
        token=token,
        created_at=datetime.now(UTC),
        last_update=datetime.now(UTC),
        last_update_monotonic=time.monotonic(),
        last_artifact=None,
        deferred_calls=None,
    )


def _make_request(headers: dict[str, str], body: dict) -> Request:
    """Build a minimal Starlette Request with headers + JSON body."""
    import json

    body_bytes = json.dumps(body).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/webhook",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }
    return Request(scope, receive)


def _task_payload(task_id: str = "t-1", state: str = "completed") -> dict:
    """JSON-shape payload as a2a sender would POST."""
    return {
        "id": task_id,
        "context_id": "ctx-AAA",
        "status": {"state": state},
        "artifacts": [
            {
                "artifact_id": "a-1",
                "parts": [{"kind": "text", "text": "Done."}],
            },
        ],
    }


@pytest.fixture
def store_with_ctx() -> ContextStore:
    s = ContextStore(max_contexts=10)
    s.get_or_create("ctx-AAA")
    return s


@pytest.fixture
def registry() -> RemoteAgentRegistry:
    import httpx

    return RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())


@pytest.mark.asyncio
async def test_valid_token_routes_and_updates(store_with_ctx, registry):
    entry = store_with_ctx.get_or_create("ctx-AAA")
    _seed_state(entry, task_id="t-1", token="tok-A")
    registry.register_token("tok-A", context_id="ctx-AAA", agent_name="B")
    registry.claim_task_id("tok-A", task_id="t-1")

    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    req = _make_request(
        headers={
            "X-A2A-Notification-Token": "tok-A",
            "content-type": "application/json",
        },
        body=_task_payload(),
    )
    resp = await handler(req)
    assert resp.status_code == 200
    assert entry.remote_tasks["t-1"].status == "completed"
    assert len(entry.pending_notifications) == 1
    # Token revoked on terminal state (security invariant).
    assert registry.lookup("tok-A") is None


@pytest.mark.asyncio
async def test_unknown_token_returns_401(store_with_ctx, registry):
    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    req = _make_request(
        headers={"X-A2A-Notification-Token": "totally-bogus"},
        body=_task_payload(),
    )
    resp = await handler(req)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_missing_token_header_returns_401(store_with_ctx, registry):
    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    req = _make_request(headers={}, body=_task_payload())
    resp = await handler(req)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_race_task_id_none_uses_body_id(store_with_ctx, registry):
    entry = store_with_ctx.get_or_create("ctx-AAA")
    _seed_state(entry, task_id="t-RACE", token="tok-R")
    registry.register_token("tok-R", context_id="ctx-AAA", agent_name="B")
    # Note: claim_task_id NOT called yet — task_id stays None on the route.

    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-R"},
        body=_task_payload(task_id="t-RACE"),
    )
    resp = await handler(req)
    assert resp.status_code == 200
    assert entry.remote_tasks["t-RACE"].status == "completed"
    # Route should now have task_id claimed
    assert registry.lookup("tok-R") is None  # revoked on terminal


@pytest.mark.asyncio
async def test_evicted_context_returns_200_no_crash(registry):
    store = ContextStore(max_contexts=10)
    # Register token pointing to a context that was never created.
    registry.register_token("tok-E", context_id="ctx-GHOST", agent_name="B")
    registry.claim_task_id("tok-E", task_id="t-1")

    handler = make_webhook_handler(registry, store, tracer=None)
    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-E"},
        body=_task_payload(),
    )
    resp = await handler(req)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_idempotent_retransmit(store_with_ctx, registry):
    entry = store_with_ctx.get_or_create("ctx-AAA")
    _seed_state(entry, task_id="t-1", token="tok-I")
    entry.remote_tasks["t-1"].status = "completed"  # already terminal
    registry.register_token("tok-I", context_id="ctx-AAA", agent_name="B")
    registry.claim_task_id("tok-I", task_id="t-1")

    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-I"},
        body=_task_payload(),
    )
    resp = await handler(req)
    assert resp.status_code == 200
    # No duplicate notification accodata.
    assert entry.pending_notifications == []


@pytest.mark.asyncio
async def test_malformed_json_returns_400(store_with_ctx, registry):
    """Invalid JSON body → 400 (not 401, since the token is valid)."""
    entry = store_with_ctx.get_or_create("ctx-AAA")
    _seed_state(entry, task_id="t-1", token="tok-M")
    registry.register_token("tok-M", context_id="ctx-AAA", agent_name="B")

    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)

    # Build a request with invalid JSON
    body_bytes = b"not valid json {{{"
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/webhook",
        "headers": [
            (b"x-a2a-notification-token", b"tok-M"),
            (b"content-type", b"application/json"),
        ],
    }
    req = Request(scope, receive)
    resp = await handler(req)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_invalid_task_body_returns_400(store_with_ctx, registry):
    """Valid JSON that doesn't parse as a Task → 400 (not 401, since token is valid)."""
    entry = store_with_ctx.get_or_create("ctx-AAA")
    _seed_state(entry, task_id="t-1", token="tok-V")
    registry.register_token("tok-V", context_id="ctx-AAA", agent_name="B")

    handler = make_webhook_handler(registry, store_with_ctx, tracer=None)
    # Valid JSON, but missing required Task fields (e.g. no id, no status).
    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-V"},
        body={"this": "is", "not": "a task"},
    )
    resp = await handler(req)
    assert resp.status_code == 400
