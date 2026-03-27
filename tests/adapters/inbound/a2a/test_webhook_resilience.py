"""Tests for webhook resilience: URL rewriting, polling fallback, override.

Covers all networking scenarios:
- Local (no container): no rewrite needed
- Docker bridge: source IP rewrite 172.x
- Kubernetes pod: source IP rewrite 10.x
- Single proxy (X-Forwarded-For)
- Multi-hop proxy chain
- Reverse proxy (X-Real-IP)
- RFC 7239 Forwarded header
- Env var override (OBELIX_WEBHOOK_REWRITE_HOST)
- Client --webhook-host / OBELIX_WEBHOOK_HOST
- Polling fallback when push fails
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from obelix.adapters.inbound.a2a.server.middleware import (
    client_ip_var,
    resolve_client_ip,
)
from obelix.adapters.inbound.a2a.server.push_config_store import (
    SmartPushNotificationConfigStore,
    _maybe_rewrite_url,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. IP Resolution (middleware logic)
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveClientIP:
    """Tests for resolve_client_ip() — extracts real client IP from headers."""

    def test_direct_connection_no_headers(self):
        """No proxy headers → use direct connection IP."""
        assert resolve_client_ip({}, "192.168.1.10") == "192.168.1.10"

    def test_direct_loopback(self):
        """Direct localhost connection."""
        assert resolve_client_ip({}, "127.0.0.1") == "127.0.0.1"

    def test_x_forwarded_for_single(self):
        """Single IP in X-Forwarded-For."""
        headers = {"x-forwarded-for": "10.0.0.5"}
        assert resolve_client_ip(headers, "172.17.0.1") == "10.0.0.5"

    def test_x_forwarded_for_chain(self):
        """Multi-hop proxy: first IP is the original client."""
        headers = {"x-forwarded-for": "203.0.113.50, 70.41.3.18, 150.172.238.178"}
        assert resolve_client_ip(headers, "127.0.0.1") == "203.0.113.50"

    def test_x_forwarded_for_with_spaces(self):
        """Handles whitespace in X-Forwarded-For."""
        headers = {"x-forwarded-for": "  10.244.1.15 , 172.16.0.1"}
        assert resolve_client_ip(headers, "127.0.0.1") == "10.244.1.15"

    def test_x_real_ip(self):
        """X-Real-IP set by Nginx/Envoy."""
        headers = {"x-real-ip": "10.244.2.30"}
        assert resolve_client_ip(headers, "127.0.0.1") == "10.244.2.30"

    def test_x_real_ip_with_spaces(self):
        """Handles whitespace in X-Real-IP."""
        headers = {"x-real-ip": "  10.244.2.30  "}
        assert resolve_client_ip(headers, "127.0.0.1") == "10.244.2.30"

    def test_forwarded_rfc7239(self):
        """RFC 7239 Forwarded header."""
        headers = {"forwarded": "for=192.0.2.60;proto=http;by=203.0.113.43"}
        assert resolve_client_ip(headers, "127.0.0.1") == "192.0.2.60"

    def test_forwarded_rfc7239_quoted(self):
        """RFC 7239 with quoted IP."""
        headers = {"forwarded": 'for="192.0.2.60"'}
        assert resolve_client_ip(headers, "127.0.0.1") == "192.0.2.60"

    def test_forwarded_rfc7239_ipv6(self):
        """RFC 7239 with IPv6 in brackets."""
        headers = {"forwarded": 'for="[2001:db8::1]"'}
        assert resolve_client_ip(headers, "127.0.0.1") == "2001:db8::1"

    def test_priority_xff_over_xri(self):
        """X-Forwarded-For takes priority over X-Real-IP."""
        headers = {
            "x-forwarded-for": "10.0.0.1",
            "x-real-ip": "10.0.0.2",
        }
        assert resolve_client_ip(headers, "127.0.0.1") == "10.0.0.1"

    def test_priority_xff_over_forwarded(self):
        """X-Forwarded-For takes priority over Forwarded."""
        headers = {
            "x-forwarded-for": "10.0.0.1",
            "forwarded": "for=10.0.0.3",
        }
        assert resolve_client_ip(headers, "127.0.0.1") == "10.0.0.1"

    def test_priority_xri_over_forwarded(self):
        """X-Real-IP takes priority over Forwarded."""
        headers = {
            "x-real-ip": "10.0.0.2",
            "forwarded": "for=10.0.0.3",
        }
        assert resolve_client_ip(headers, "127.0.0.1") == "10.0.0.2"

    def test_empty_xff_falls_through(self):
        """Empty X-Forwarded-For falls through to direct IP."""
        headers = {"x-forwarded-for": ""}
        assert resolve_client_ip(headers, "192.168.1.1") == "192.168.1.1"

    def test_empty_headers(self):
        """Empty header dict."""
        assert resolve_client_ip({}, "10.0.0.1") == "10.0.0.1"

    def test_empty_direct_ip(self):
        """No client info available."""
        assert resolve_client_ip({}, "") == ""


# ═══════════════════════════════════════════════════════════════════════════
# 2. URL Rewriting (push config store logic)
# ═══════════════════════════════════════════════════════════════════════════


class TestMaybeRewriteUrl:
    """Tests for _maybe_rewrite_url() — rewrites localhost in webhook URLs."""

    def test_no_rewrite_when_local(self):
        """No rewrite when client IP is also loopback (genuinely local)."""
        token = client_ip_var.set("127.0.0.1")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == url
        finally:
            client_ip_var.reset(token)

    def test_rewrite_docker_bridge(self):
        """Docker bridge: client IP 172.x → rewrite."""
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://172.19.0.1:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_rewrite_k8s_pod(self):
        """K8s pod network: client IP 10.x → rewrite."""
        token = client_ip_var.set("10.244.1.15")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://10.244.1.15:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_rewrite_localhost_string(self):
        """Rewrites 'localhost' as well as '127.0.0.1'."""
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://localhost:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://172.19.0.1:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_rewrite_ipv6_loopback(self):
        """Rewrites ::1 loopback."""
        token = client_ip_var.set("10.0.0.5")
        try:
            url = "http://::1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://10.0.0.5:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_no_rewrite_non_loopback_url(self):
        """URL with a real IP is not rewritten."""
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://192.168.1.100:54321/webhook"
            assert _maybe_rewrite_url(url) == url
        finally:
            client_ip_var.reset(token)

    def test_no_rewrite_no_client_ip(self):
        """No client IP in context → no rewrite."""
        token = client_ip_var.set("")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == url
        finally:
            client_ip_var.reset(token)

    def test_env_var_override(self, monkeypatch):
        """OBELIX_WEBHOOK_REWRITE_HOST env var takes priority."""
        monkeypatch.setenv("OBELIX_WEBHOOK_REWRITE_HOST", "my-proxy.example.com")
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert (
                _maybe_rewrite_url(url) == "http://my-proxy.example.com:54321/webhook"
            )
        finally:
            client_ip_var.reset(token)

    def test_env_var_override_no_loopback(self, monkeypatch):
        """Env var override doesn't rewrite non-loopback URLs."""
        monkeypatch.setenv("OBELIX_WEBHOOK_REWRITE_HOST", "my-proxy.example.com")
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://10.0.0.5:54321/webhook"
            assert _maybe_rewrite_url(url) == url
        finally:
            client_ip_var.reset(token)

    def test_env_var_override_priority_over_client_ip(self, monkeypatch):
        """Env var takes priority over auto-detected client IP."""
        monkeypatch.setenv("OBELIX_WEBHOOK_REWRITE_HOST", "override.example.com")
        token = client_ip_var.set("10.244.1.15")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert (
                _maybe_rewrite_url(url) == "http://override.example.com:54321/webhook"
            )
        finally:
            client_ip_var.reset(token)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SmartPushNotificationConfigStore (integration)
# ═══════════════════════════════════════════════════════════════════════════


class TestSmartPushNotificationConfigStore:
    """Tests for SmartPushNotificationConfigStore.set_info() rewrite behavior."""

    @pytest.mark.asyncio
    async def test_stores_without_rewrite_when_local(self):
        """Local client: URL stored unchanged."""
        from a2a.types import PushNotificationConfig

        store = SmartPushNotificationConfigStore()
        config = PushNotificationConfig(url="http://127.0.0.1:5000/webhook")

        token = client_ip_var.set("127.0.0.1")
        try:
            await store.set_info("task-1", config)
            stored = await store.get_info("task-1")
            assert stored[0].url == "http://127.0.0.1:5000/webhook"
        finally:
            client_ip_var.reset(token)

    @pytest.mark.asyncio
    async def test_rewrites_when_docker(self):
        """Docker bridge client: URL rewritten to bridge IP."""
        from a2a.types import PushNotificationConfig

        store = SmartPushNotificationConfigStore()
        config = PushNotificationConfig(url="http://127.0.0.1:5000/webhook")

        token = client_ip_var.set("172.19.0.1")
        try:
            await store.set_info("task-2", config)
            stored = await store.get_info("task-2")
            assert stored[0].url == "http://172.19.0.1:5000/webhook"
        finally:
            client_ip_var.reset(token)

    @pytest.mark.asyncio
    async def test_preserves_token(self):
        """Rewrite preserves the push notification token."""
        from a2a.types import PushNotificationConfig

        store = SmartPushNotificationConfigStore()
        config = PushNotificationConfig(
            url="http://127.0.0.1:5000/webhook",
            token="secret-token-123",
        )

        token = client_ip_var.set("10.244.1.15")
        try:
            await store.set_info("task-3", config)
            stored = await store.get_info("task-3")
            assert stored[0].url == "http://10.244.1.15:5000/webhook"
            assert stored[0].token == "secret-token-123"
        finally:
            client_ip_var.reset(token)

    @pytest.mark.asyncio
    async def test_preserves_id(self):
        """Rewrite preserves the push notification config ID."""
        from a2a.types import PushNotificationConfig

        store = SmartPushNotificationConfigStore()
        config = PushNotificationConfig(
            url="http://127.0.0.1:5000/webhook",
            id="custom-id",
        )

        token = client_ip_var.set("172.19.0.1")
        try:
            await store.set_info("task-4", config)
            stored = await store.get_info("task-4")
            assert stored[0].id == "custom-id"
        finally:
            client_ip_var.reset(token)


# ═══════════════════════════════════════════════════════════════════════════
# 4. WebhookServer (client-side host configuration)
# ═══════════════════════════════════════════════════════════════════════════


class TestWebhookServerHost:
    """Tests for WebhookServer webhook_host configuration."""

    def test_default_host(self):
        """Default webhook host is 127.0.0.1."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
            WebhookServer,
        )

        tracker = TaskTracker()
        server = WebhookServer(tracker)
        assert server.webhook_host == "127.0.0.1"

    def test_explicit_host(self):
        """Explicit webhook_host parameter."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
            WebhookServer,
        )

        tracker = TaskTracker()
        server = WebhookServer(tracker, webhook_host="192.168.1.100")
        assert server.webhook_host == "192.168.1.100"

    def test_env_var_host(self, monkeypatch):
        """OBELIX_WEBHOOK_HOST env var."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
            WebhookServer,
        )

        monkeypatch.setenv("OBELIX_WEBHOOK_HOST", "host.docker.internal")
        tracker = TaskTracker()
        server = WebhookServer(tracker)
        assert server.webhook_host == "host.docker.internal"

    def test_explicit_overrides_env_var(self, monkeypatch):
        """Explicit parameter takes priority over env var."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
            WebhookServer,
        )

        monkeypatch.setenv("OBELIX_WEBHOOK_HOST", "from-env")
        tracker = TaskTracker()
        server = WebhookServer(tracker, webhook_host="explicit")
        assert server.webhook_host == "explicit"

    def test_get_url_uses_webhook_host(self):
        """get_url() includes the configured webhook host."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
            WebhookServer,
        )

        tracker = TaskTracker()
        server = WebhookServer(tracker, webhook_host="10.244.1.10")
        server._port = 12345
        assert server.get_url() == "http://10.244.1.10:12345/webhook"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Polling Fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestPollingFallback:
    """Tests for the client-side polling fallback mechanism."""

    @pytest.mark.asyncio
    async def test_active_tasks_older_than_threshold_are_polled(self):
        """Tasks non-terminal for >3s trigger a poll."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
        )

        tracker = TaskTracker()
        await tracker.register("task-1", "agent-a")

        # Simulate age > 3s
        info = tracker.get("task-1")
        info.timestamp = time.time() - 5

        active = tracker.get_active()
        old = [t for t in active if time.time() - t.timestamp > 3.0]
        assert len(old) == 1
        assert old[0].task_id == "task-1"

    @pytest.mark.asyncio
    async def test_recently_registered_tasks_not_polled(self):
        """Tasks registered <3s ago are not polled."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
        )

        tracker = TaskTracker()
        await tracker.register("task-2", "agent-b")

        active = tracker.get_active()
        old = [t for t in active if time.time() - t.timestamp > 3.0]
        assert len(old) == 0

    @pytest.mark.asyncio
    async def test_terminal_tasks_not_polled(self):
        """Terminal tasks are not in get_active()."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
        )

        tracker = TaskTracker()
        await tracker.register("task-3", "agent-c")
        await tracker.update({"id": "task-3", "status": {"state": "completed"}})

        active = tracker.get_active()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_poll_update_makes_task_terminal(self):
        """Simulating a poll result updates the tracker to terminal state."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
        )

        tracker = TaskTracker()
        await tracker.register("task-4", "agent-d")

        # Simulate poll result
        poll_result = {
            "id": "task-4",
            "status": {"state": "completed"},
        }
        await tracker.update(poll_result)

        info = tracker.get("task-4")
        assert info.is_terminal
        assert info.state == "completed"

    @pytest.mark.asyncio
    async def test_idempotent_update(self):
        """Multiple updates with same state are idempotent."""
        from obelix.adapters.inbound.a2a.client.webhook_server import (
            TaskTracker,
        )

        tracker = TaskTracker()
        await tracker.register("task-5", "agent-e")

        completed = {"id": "task-5", "status": {"state": "completed"}}
        await tracker.update(completed)
        await tracker.update(completed)  # duplicate

        info = tracker.get("task-5")
        assert info.state == "completed"
        # Still only one task
        assert len(tracker.get_all()) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. SmartPushNotificationSender (warning logging)
# ═══════════════════════════════════════════════════════════════════════════


class TestSmartPushNotificationSender:
    """Tests for SmartPushNotificationSender warning behavior."""

    @pytest.mark.asyncio
    async def test_successful_push_returns_true(self):
        """Successful push returns True."""
        import httpx
        from a2a.types import PushNotificationConfig, Task, TaskState, TaskStatus

        from obelix.adapters.inbound.a2a.server.push_sender import (
            SmartPushNotificationSender,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response

        sender = SmartPushNotificationSender(
            httpx_client=mock_client, config_store=MagicMock()
        )
        task = Task(
            id="task-1",
            status=TaskStatus(state=TaskState.completed),
            context_id="ctx-1",
        )
        config = PushNotificationConfig(url="http://10.0.0.1:5000/webhook")

        result = await sender._dispatch_notification(task, config)
        assert result is True

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self):
        """Connection error returns False (doesn't raise)."""
        import httpx
        from a2a.types import PushNotificationConfig, Task, TaskState, TaskStatus

        from obelix.adapters.inbound.a2a.server.push_sender import (
            SmartPushNotificationSender,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        sender = SmartPushNotificationSender(
            httpx_client=mock_client, config_store=MagicMock()
        )
        task = Task(
            id="task-2",
            status=TaskStatus(state=TaskState.completed),
            context_id="ctx-2",
        )
        config = PushNotificationConfig(url="http://127.0.0.1:5000/webhook")

        result = await sender._dispatch_notification(task, config)
        assert result is False

    @pytest.mark.asyncio
    async def test_timeout_error_returns_false(self):
        """Timeout returns False (doesn't raise)."""
        import httpx
        from a2a.types import PushNotificationConfig, Task, TaskState, TaskStatus

        from obelix.adapters.inbound.a2a.server.push_sender import (
            SmartPushNotificationSender,
        )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.side_effect = httpx.ReadTimeout("timed out")

        sender = SmartPushNotificationSender(
            httpx_client=mock_client, config_store=MagicMock()
        )
        task = Task(
            id="task-3",
            status=TaskStatus(state=TaskState.completed),
            context_id="ctx-3",
        )
        config = PushNotificationConfig(url="http://127.0.0.1:5000/webhook")

        result = await sender._dispatch_notification(task, config)
        assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# 7. End-to-end scenario simulations
# ═══════════════════════════════════════════════════════════════════════════


class TestScenarios:
    """High-level scenario tests combining multiple components."""

    def test_scenario_local_no_container(self):
        """Scenario: both client and server on same host, no Docker."""
        token = client_ip_var.set("127.0.0.1")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == url  # No rewrite
        finally:
            client_ip_var.reset(token)

    def test_scenario_docker_bridge(self):
        """Scenario: server in Docker, client on host via bridge network."""
        token = client_ip_var.set("172.19.0.1")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://172.19.0.1:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_scenario_k8s_pods(self):
        """Scenario: client and server in different K8s pods."""
        token = client_ip_var.set("10.244.1.10")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://10.244.1.10:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_scenario_behind_nginx_proxy(self):
        """Scenario: client behind Nginx reverse proxy."""
        headers = {"x-forwarded-for": "203.0.113.50, 10.0.0.1"}
        ip = resolve_client_ip(headers, "10.0.0.1")
        assert ip == "203.0.113.50"

        token = client_ip_var.set(ip)
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert _maybe_rewrite_url(url) == "http://203.0.113.50:54321/webhook"
        finally:
            client_ip_var.reset(token)

    def test_scenario_behind_envoy_proxy(self):
        """Scenario: client behind Envoy service mesh (X-Real-IP)."""
        headers = {"x-real-ip": "10.244.2.30"}
        ip = resolve_client_ip(headers, "10.244.0.1")
        assert ip == "10.244.2.30"

    def test_scenario_multi_hop_proxy(self):
        """Scenario: request through multiple proxies."""
        headers = {
            "x-forwarded-for": "198.51.100.42, 10.10.0.1, 172.16.0.5",
        }
        ip = resolve_client_ip(headers, "127.0.0.1")
        assert ip == "198.51.100.42"  # Original client IP

    def test_scenario_client_sends_real_ip(self):
        """Scenario: client on a different network sends its real IP."""
        token = client_ip_var.set("192.168.1.100")
        try:
            url = "http://192.168.1.100:54321/webhook"
            # URL already has the real IP — no rewrite
            assert _maybe_rewrite_url(url) == url
        finally:
            client_ip_var.reset(token)

    def test_scenario_env_var_override_production(self, monkeypatch):
        """Scenario: production setup with explicit webhook rewrite host."""
        monkeypatch.setenv("OBELIX_WEBHOOK_REWRITE_HOST", "webhook.prod.internal")
        token = client_ip_var.set("10.244.1.10")
        try:
            url = "http://127.0.0.1:54321/webhook"
            assert (
                _maybe_rewrite_url(url) == "http://webhook.prod.internal:54321/webhook"
            )
        finally:
            client_ip_var.reset(token)
