"""ASGI middleware that captures the client IP from incoming HTTP requests.

Resolves the real client IP through proxy headers (X-Forwarded-For, X-Real-IP,
Forwarded) and falls back to the direct connection IP.  The resolved IP is
stored in a ``contextvars.ContextVar`` so that downstream components (e.g. the
push notification config store) can read it without needing direct access to
the request object.

This is critical for **webhook URL rewriting**: when a client behind Docker or
Kubernetes sends ``http://127.0.0.1:<port>/webhook`` as its push notification
URL, the server can detect the mismatch and rewrite it to the client's real IP.
"""

from __future__ import annotations

import contextvars
import re
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

# ── Context variables (read by SmartPushNotificationConfigStore) ──────────

client_ip_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "client_ip", default=""
)


# ── IP resolution logic ──────────────────────────────────────────────────

# RFC 7239 ``Forwarded: for=<ip>`` pattern
_FORWARDED_FOR_RE = re.compile(r'for="?([^";,\s]+)"?', re.IGNORECASE)


def resolve_client_ip(
    headers: dict[str, str],
    direct_ip: str,
) -> str:
    """Resolve the original client IP from proxy headers or direct connection.

    Resolution order (first non-empty wins):
        1. ``X-Forwarded-For`` — first IP in the chain (closest to client)
        2. ``X-Real-IP`` — set by Nginx / Envoy
        3. ``Forwarded: for=<ip>`` — RFC 7239 standard
        4. Direct connection IP from the ASGI scope

    Args:
        headers: Lowercased header dict from the ASGI scope.
        direct_ip: The IP from ``scope["client"]``.

    Returns:
        The resolved client IP string.
    """
    # 1. X-Forwarded-For (first entry = original client)
    if xff := headers.get("x-forwarded-for"):
        return xff.split(",")[0].strip()

    # 2. X-Real-IP
    if xri := headers.get("x-real-ip"):
        return xri.strip()

    # 3. Forwarded (RFC 7239)
    if fwd := headers.get("forwarded"):
        match = _FORWARDED_FOR_RE.search(fwd)
        if match:
            return match.group(1).strip("[]")  # strip [] for IPv6

    # 4. Direct connection
    return direct_ip


# ── ASGI Middleware ──────────────────────────────────────────────────────


class ClientIPMiddleware:
    """ASGI middleware that captures the client IP into a context variable.

    Wrap the A2A FastAPI/Starlette app with this middleware so that
    ``client_ip_var.get()`` returns the resolved client IP inside any
    downstream async handler running in the same request context.

    Usage::

        app = a2a_app.build(title="...")
        app = ClientIPMiddleware(app)  # wrap
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> Any:
        if scope["type"] == "http":
            # Extract direct IP
            client = scope.get("client")
            direct_ip = client[0] if client else ""

            # Build lowercased header dict
            raw_headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}

            resolved = resolve_client_ip(headers, direct_ip)
            token = client_ip_var.set(resolved)
            try:
                await self._app(scope, receive, send)
            finally:
                client_ip_var.reset(token)
        else:
            await self._app(scope, receive, send)
