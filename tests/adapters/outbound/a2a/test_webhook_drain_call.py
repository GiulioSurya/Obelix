"""Verify that the outbound webhook handler invokes maybe_spawn_drain_task
after handle_remote_update.

Iron rule: hand-written FakeExecutor implementing the drainer Protocol;
real Starlette ``Request`` exercising the real ``make_webhook_handler``
closure; real ``ContextStore`` and real ``RemoteAgentRegistry``. No mocks.
"""

from __future__ import annotations

import inspect
import json
import time
from datetime import UTC, datetime

import httpx
import pytest
from starlette.requests import Request

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry
from obelix.adapters.outbound.a2a.state import RemoteTaskState
from obelix.adapters.outbound.a2a.webhook import make_webhook_handler


class _RecordingExecutor:
    """FakeExecutor that captures ``spawn_drain_task`` invocations.

    Implements only the surface the drainer's ``_DrainExecutorProtocol``
    requires (a single async ``spawn_drain_task`` method).
    """

    def __init__(self) -> None:
        self.spawn_calls: list[tuple[ContextEntry, str]] = []

    async def spawn_drain_task(self, *, entry: ContextEntry, context_id: str) -> None:
        self.spawn_calls.append((entry, context_id))


def _make_request(headers: dict[str, str], body: dict) -> Request:
    """Build a minimal Starlette Request with headers + JSON body."""
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


def _seed_state(entry: ContextEntry, *, task_id: str, token: str) -> None:
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


def _task_payload(task_id: str = "t-drain", state: str = "completed") -> dict:
    return {
        "id": task_id,
        "context_id": "ctx-DRAIN",
        "status": {"state": state},
        "artifacts": [
            {
                "artifact_id": "a-1",
                "parts": [{"kind": "text", "text": "Done."}],
            },
        ],
    }


def test_make_webhook_handler_signature_accepts_executor() -> None:
    """The factory must declare an ``executor`` parameter so callers can wire
    it from ``AgentFactory.a2a_serve``."""
    sig = inspect.signature(make_webhook_handler)
    assert "executor" in sig.parameters


@pytest.mark.asyncio
async def test_webhook_handler_invokes_drainer_after_terminal_update() -> None:
    """End-to-end: terminal Task POST → handle_remote_update appends a
    HumanMessage to ``entry.pending_notifications`` → the webhook handler
    awaits ``maybe_spawn_drain_task`` → drainer's checks pass (queue
    non-empty, ``entry.idle.is_set()`` defaults to True) → executor's
    ``spawn_drain_task`` is invoked exactly once.
    """
    store = ContextStore(max_contexts=8)
    entry = store.get_or_create("ctx-DRAIN")
    _seed_state(entry, task_id="t-drain", token="tok-drain")

    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    registry.register_token("tok-drain", context_id="ctx-DRAIN", agent_name="B")
    registry.claim_task_id("tok-drain", task_id="t-drain")

    executor = _RecordingExecutor()

    handler = make_webhook_handler(registry, store, executor=executor, tracer=None)

    req = _make_request(
        headers={
            "X-A2A-Notification-Token": "tok-drain",
            "content-type": "application/json",
        },
        body=_task_payload(),
    )

    resp = await handler(req)
    assert resp.status_code == 200

    # Sanity: handle_remote_update did its job — a notification is queued.
    assert len(entry.pending_notifications) == 1

    # Drainer was invoked exactly once with the right keyword arguments.
    assert len(executor.spawn_calls) == 1
    spawned_entry, spawned_context_id = executor.spawn_calls[0]
    assert spawned_entry is entry
    assert spawned_context_id == "ctx-DRAIN"


@pytest.mark.asyncio
async def test_webhook_handler_skips_drainer_when_executor_none() -> None:
    """Backward-compat: ``executor=None`` (default) means no drainer call
    and no error — caller might be running the legacy non-drainer path."""
    store = ContextStore(max_contexts=8)
    entry = store.get_or_create("ctx-DRAIN")
    _seed_state(entry, task_id="t-drain", token="tok-drain")

    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    registry.register_token("tok-drain", context_id="ctx-DRAIN", agent_name="B")
    registry.claim_task_id("tok-drain", task_id="t-drain")

    handler = make_webhook_handler(registry, store, tracer=None)

    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-drain"},
        body=_task_payload(),
    )

    resp = await handler(req)
    assert resp.status_code == 200
    assert len(entry.pending_notifications) == 1


@pytest.mark.asyncio
async def test_webhook_handler_skips_drainer_when_context_busy() -> None:
    """If ``entry.idle`` is cleared (a turn is running), the drainer's
    check 2 short-circuits and ``spawn_drain_task`` is NOT called.

    This test verifies the integration still funnels through the drainer
    function (so its idempotency logic is honored) — not that the webhook
    duplicates the check itself.
    """
    store = ContextStore(max_contexts=8)
    entry = store.get_or_create("ctx-DRAIN")
    _seed_state(entry, task_id="t-drain", token="tok-drain")
    entry.idle.clear()  # simulate "turn in progress"

    registry = RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())
    registry.register_token("tok-drain", context_id="ctx-DRAIN", agent_name="B")
    registry.claim_task_id("tok-drain", task_id="t-drain")

    executor = _RecordingExecutor()
    handler = make_webhook_handler(registry, store, executor=executor, tracer=None)

    req = _make_request(
        headers={"X-A2A-Notification-Token": "tok-drain"},
        body=_task_payload(),
    )
    resp = await handler(req)
    assert resp.status_code == 200
    # Notification was still queued, but drainer did NOT spawn.
    assert len(entry.pending_notifications) == 1
    assert executor.spawn_calls == []
