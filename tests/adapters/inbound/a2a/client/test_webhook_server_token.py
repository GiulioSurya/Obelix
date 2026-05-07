"""Webhook server must reject POSTs whose X-A2A-Notification-Token header
does not match the expected per-session token. TEMP-PATCH-SPEC-1.

Iron rule: NO mock httpx; uses real httpx.AsyncClient against the running WebhookServer.
"""

from __future__ import annotations

import httpx
import pytest

from obelix.adapters.inbound.a2a.client.webhook_server import (
    TaskTracker,
    WebhookServer,
)


@pytest.mark.asyncio
async def test_post_with_correct_token_accepted():
    tracker = TaskTracker()
    server = WebhookServer(
        tracker,
        webhook_host="127.0.0.1",
        webhook_port=0,  # random
        expected_token="good-token",
    )
    await server.start()
    try:
        url = server.get_url()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "kind": "task",
                    "status": {
                        "state": "completed",
                        "timestamp": "2026-05-07T12:00:00+00:00",
                    },
                },
                headers={"X-A2A-Notification-Token": "good-token"},
            )
            assert resp.status_code == 200
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_post_with_wrong_token_rejected_401():
    tracker = TaskTracker()
    server = WebhookServer(
        tracker,
        webhook_host="127.0.0.1",
        webhook_port=0,
        expected_token="good-token",
    )
    await server.start()
    try:
        url = server.get_url()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={"id": "x"},
                headers={"X-A2A-Notification-Token": "wrong"},
            )
            assert resp.status_code == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_post_without_token_header_rejected_when_token_required():
    tracker = TaskTracker()
    server = WebhookServer(
        tracker,
        webhook_host="127.0.0.1",
        webhook_port=0,
        expected_token="good-token",
    )
    await server.start()
    try:
        url = server.get_url()
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json={"id": "x"})
            assert resp.status_code == 401
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_post_accepted_when_no_token_configured():
    """Backward compat: server with expected_token=None accepts any POST."""
    tracker = TaskTracker()
    server = WebhookServer(
        tracker,
        webhook_host="127.0.0.1",
        webhook_port=0,
        # expected_token NOT passed → None
    )
    await server.start()
    try:
        url = server.get_url()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                json={
                    "id": "task-1",
                    "contextId": "ctx-1",
                    "kind": "task",
                    "status": {
                        "state": "completed",
                        "timestamp": "2026-05-07T12:00:00+00:00",
                    },
                },
            )
            assert resp.status_code == 200
    finally:
        await server.stop()
