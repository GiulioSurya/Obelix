"""Bug regression: ``_DrainSpawnEventQueue`` silently drops artifacts.

Symptom (observed in smoke test 2026-05-08): after a drain-spawn task
completes, the CLI shows ``"completed (no content)"`` even though the
agent produced an 800+ character response. The orchestrator log shows:

    [A2A drain] _post POSTING state=completed http_status=200

with no artifacts in the payload. The 11+ ``TaskArtifactUpdateEvent``
that carry the chunked agent response are silently skipped because the
queue assumes "the completed status update carries artifacts" — but the
queue itself reconstructs a *minimal* Task (id/context_id/status only),
so artifacts never make it to the wire.

Plus a second bug: the ``status=working`` event that carries the agent's
final ``status.message`` is dropped by the dedup filter (same state as
the previous ``working``), losing the message body too.

Iron rule: NO mock for httpx. Real Starlette + ASGITransport.
"""

from __future__ import annotations

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_includes_artifacts_in_completed_post():
    """When the executor enqueues TaskArtifactUpdateEvent followed by a
    completed TaskStatusUpdateEvent, the resulting POST payload to the
    CLI webhook MUST include the artifacts (with the agent's text reply).
    """
    from a2a.types import (
        Artifact,
        Part,
        TaskArtifactUpdateEvent,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
        TextPart,
    )

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
            task_id="t-drain",
            context_id="ctx-1",
        )

        # 1. working initial
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-drain",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp="2026-05-08T10:00:00+00:00",
                ),
            )
        )

        # 2. ArtifactUpdate carrying the agent's text response
        artifact = Artifact(
            artifact_id="art-1",
            name="reply",
            parts=[Part(root=TextPart(text="MARKER-AGENT-RESPONSE"))],
        )
        await queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id="t-drain",
                context_id="ctx-1",
                artifact=artifact,
                append=False,
                last_chunk=True,
            )
        )

        # 3. completed status update
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-drain",
                context_id="ctx-1",
                final=True,
                status=TaskStatus(
                    state=TaskState.completed,
                    timestamp="2026-05-08T10:00:01+00:00",
                ),
            )
        )

    # The completed POST MUST carry the artifacts.
    assert len(received) >= 2, (
        f"Expected at least working + completed POSTs, got {len(received)}"
    )
    completed = next((p for p in received if p["status"]["state"] == "completed"), None)
    assert completed is not None, (
        f"No completed POST received. Posts: {[p['status']['state'] for p in received]}"
    )

    artifacts = completed.get("artifacts", [])
    assert artifacts, (
        "Bug: completed POST has NO artifacts. The ArtifactUpdate events "
        "were dropped and not re-included on the completed Task. "
        f"Payload: {completed}"
    )

    flat_text = "".join(
        part["text"]
        for art in artifacts
        for part in art.get("parts", [])
        if "text" in part
    )
    assert "MARKER-AGENT-RESPONSE" in flat_text, (
        f"Bug: artifact text lost between ArtifactUpdate and completed POST. "
        f"Posted artifacts: {artifacts}"
    )


@pytest.mark.asyncio
async def test_drain_spawn_event_queue_does_not_dedup_status_with_message():
    """A TaskStatusUpdateEvent that carries a non-None ``status.message``
    must be POSTed even if its ``status.state`` matches the previous
    state. The current dedup-by-state-only loses the agent's reply
    message when it arrives on a working->working transition.
    """
    from a2a.types import (
        Message,
        Part,
        Role,
        TaskState,
        TaskStatus,
        TaskStatusUpdateEvent,
        TextPart,
    )

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
            task_id="t-drain",
            context_id="ctx-1",
        )

        # 1. working (no message)
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-drain",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp="2026-05-08T10:00:00+00:00",
                ),
            )
        )

        # 2. working WITH agent message
        agent_msg = Message(
            message_id="m-1",
            role=Role.agent,
            parts=[Part(root=TextPart(text="MARKER-AGENT-MESSAGE"))],
        )
        await queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id="t-drain",
                context_id="ctx-1",
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    timestamp="2026-05-08T10:00:01+00:00",
                    message=agent_msg,
                ),
            )
        )

    # Two distinct POSTs (the dedup must NOT collapse the second one because
    # it carries new content via ``status.message``).
    assert len(received) == 2, (
        f"Expected 2 POSTs (working + working-with-message), got {len(received)}. "
        f"Posts: {received}"
    )
    second = received[1]
    assert second["status"].get("message") is not None, (
        "Bug: the second working update lost its status.message after dedup. "
        f"Payload: {second}"
    )
    flat_text = "".join(
        part.get("text", "")
        for part in second["status"]["message"].get("parts", [])
        if "text" in part
    )
    assert "MARKER-AGENT-MESSAGE" in flat_text, (
        f"Bug: agent message text not preserved. Got: {second['status']['message']}"
    )
