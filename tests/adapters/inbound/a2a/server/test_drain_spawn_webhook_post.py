"""Integration test: when a drain-spawn task changes state, the executor
POSTs the Task JSON (camelCase) to entry.client_webhook_url with the
X-A2A-Notification-Token header.

NO MOCKS for httpx: uses a real ASGI test webhook (Starlette + httpx ASGITransport).
TEMP-PATCH-SPEC-1.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

pytestmark = pytest.mark.skip(reason="adapted in spec2 Task 16")


@pytest.mark.asyncio
async def test_post_to_webhook_sends_camelcase_payload_with_token():
    """Round-trip a real Task through model_dump and a real Starlette webhook
    via httpx ASGI transport. Verifies camelCase contract + token header."""
    received: list[dict] = []
    received_headers: list[dict] = []

    async def webhook(request: Request) -> JSONResponse:
        received.append(await request.json())
        received_headers.append(dict(request.headers))
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        from a2a.types import Task, TaskState, TaskStatus

        task = Task(
            id="t-1",
            context_id="ctx-1",
            status=TaskStatus(
                state=TaskState.completed,
                timestamp="2026-05-07T12:00:00+00:00",
            ),
        )
        payload = task.model_dump(mode="json", exclude_none=True)

        # Verify camelCase invariant
        assert "contextId" in payload, f"expected camelCase, got {list(payload)}"
        assert "id" in payload

        response = await client.post(
            "/webhook",
            json=payload,
            headers={"X-A2A-Notification-Token": "test-token-xyz"},
            timeout=5.0,
        )
        assert response.status_code == 200

    assert len(received) == 1
    assert received[0]["contextId"] == "ctx-1"
    assert received[0]["id"] == "t-1"
    assert received[0]["status"]["state"] == "completed"
    assert received_headers[0]["x-a2a-notification-token"] == "test-token-xyz"


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_posts_on_status_update_event():
    """_DrainSpawnEventQueue receives TaskStatusUpdateEvent → POSTs reconstructed Task.

    No httpx mock. Uses real ASGITransport + Starlette webhook.
    """
    from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

    from obelix.adapters.inbound.a2a.server.executor import _DrainSpawnEventQueue

    received: list[dict] = []

    async def webhook(request: Request) -> JSONResponse:
        received.append(await request.json())
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        queue = _DrainSpawnEventQueue(
            httpx_client=client,
            webhook_url="http://testserver/webhook",
            webhook_token="abc",
            task_id="t-1",
            context_id="ctx-1",
        )

        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-1",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp="2026-05-07T12:00:00+00:00",
                ),
            )
        )

    assert len(received) == 1
    assert received[0]["contextId"] == "ctx-1"
    assert received[0]["id"] == "t-1"
    assert received[0]["status"]["state"] == "working"


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_no_dedup_each_event_posts():
    """Each TaskStatusUpdateEvent must produce a POST.

    The previous implementation deduped by ``status.state`` and dropped
    consecutive ``working`` updates — that broke the realistic flow where
    the agent's reply arrives via a second ``working`` update (with
    ``status.message`` populated) BEFORE the ``completed`` update. Dedup
    by state alone silently lost the agent's message body.

    The new contract: post every ``TaskStatusUpdateEvent`` as-is. Volume
    is low (typical drain-spawn turn produces 3-4 status updates).
    """
    from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

    from obelix.adapters.inbound.a2a.server.executor import _DrainSpawnEventQueue

    received: list[dict] = []

    async def webhook(request: Request) -> JSONResponse:
        received.append(await request.json())
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        queue = _DrainSpawnEventQueue(
            httpx_client=client,
            webhook_url="http://testserver/webhook",
            webhook_token="abc",
            task_id="t-1",
            context_id="ctx-1",
        )
        evt = TaskStatusUpdateEvent(
            task_id="t-1",
            context_id="ctx-1",
            final=False,
            status=TaskStatus(
                state=TaskState.working,
                timestamp="2026-05-07T12:00:00+00:00",
            ),
        )
        await queue.enqueue_event(evt)
        await queue.enqueue_event(evt)
        await queue.enqueue_event(evt)

    assert len(received) == 3, f"Expected 3 POSTs (no dedup), got {len(received)}"


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_posts_on_state_change():
    """Different states result in MULTIPLE POSTs (one per state change)."""
    from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

    from obelix.adapters.inbound.a2a.server.executor import _DrainSpawnEventQueue

    received: list[dict] = []

    async def webhook(request: Request) -> JSONResponse:
        received.append(await request.json())
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        queue = _DrainSpawnEventQueue(
            httpx_client=client,
            webhook_url="http://testserver/webhook",
            webhook_token="abc",
            task_id="t-1",
            context_id="ctx-1",
        )
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-1",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(state=TaskState.working, timestamp="t1"),
            )
        )
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-1",
                context_id="ctx-1",
                final=True,
                status=TaskStatus(state=TaskState.completed, timestamp="t2"),
            )
        )

    assert len(received) == 2
    assert received[0]["status"]["state"] == "working"
    assert received[1]["status"]["state"] == "completed"


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_swallows_exceptions():
    """Best-effort: if the webhook returns 5xx or is unreachable, the queue
    must NOT raise — just log a warning.
    """
    from a2a.types import TaskState, TaskStatus, TaskStatusUpdateEvent

    from obelix.adapters.inbound.a2a.server.executor import _DrainSpawnEventQueue

    # ASGI app that always returns 500
    async def webhook(request: Request) -> JSONResponse:
        return JSONResponse({"err": "boom"}, status_code=500)

    app = Starlette(routes=[Route("/webhook", webhook, methods=["POST"])])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        queue = _DrainSpawnEventQueue(
            httpx_client=client,
            webhook_url="http://testserver/webhook",
            webhook_token="abc",
            task_id="t-1",
            context_id="ctx-1",
        )
        # Must not raise even on 500
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-1",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(state=TaskState.working, timestamp="t1"),
            )
        )
