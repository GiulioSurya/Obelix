"""Smart push notification config store with automatic webhook URL rewriting.

Wraps ``InMemoryPushNotificationConfigStore`` and rewrites ``127.0.0.1`` in
webhook URLs to the real client IP (resolved by ``ClientIPMiddleware``).

This solves a fundamental networking problem: when the A2A server runs inside a
Docker container or Kubernetes pod, ``127.0.0.1`` in the webhook URL points to
the *container's* loopback — not the client's host.  By rewriting to the actual
client IP (detected from proxy headers or the TCP connection), push
notifications reach the client regardless of the network topology.

Rewrite priority:
    1. ``OBELIX_WEBHOOK_REWRITE_HOST`` env var (explicit override)
    2. Client IP from ``ClientIPMiddleware`` (automatic)
    3. No rewrite (client is genuinely on localhost)
"""

from __future__ import annotations

import os

from a2a.server.tasks.inmemory_push_notification_config_store import (
    InMemoryPushNotificationConfigStore,
)
from a2a.types import PushNotificationConfig

from obelix.adapters.inbound.a2a.server.middleware import client_ip_var
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


class SmartPushNotificationConfigStore(InMemoryPushNotificationConfigStore):
    """Config store that rewrites localhost webhook URLs to the real client IP.

    Drop-in replacement for ``InMemoryPushNotificationConfigStore``.  On
    ``set_info()``, inspects the webhook URL and rewrites the host portion
    when the client is clearly not on the same loopback interface.
    """

    async def set_info(
        self,
        task_id: str,
        notification_config: PushNotificationConfig,
    ) -> None:
        """Store push config, rewriting localhost URLs if needed."""
        url = notification_config.url
        rewritten_url = _maybe_rewrite_url(url)

        if rewritten_url != url:
            notification_config = PushNotificationConfig(
                url=rewritten_url,
                id=notification_config.id,
                token=notification_config.token,
            )
            logger.warning(
                "Webhook URL rewritten: {} → {} "
                "(client IP detected from request — "
                "server is likely running in a container)",
                url,
                rewritten_url,
            )

        await super().set_info(task_id, notification_config)


def _maybe_rewrite_url(url: str) -> str:
    """Rewrite localhost in a webhook URL to the resolved client IP.

    Returns the original URL unchanged if:
    - The URL doesn't contain a loopback address
    - The resolved client IP is also loopback (genuinely local)
    - No client IP is available
    """
    # 1. Check explicit override
    override_host = os.environ.get("OBELIX_WEBHOOK_REWRITE_HOST")
    if override_host:
        for loopback in _LOOPBACK_HOSTS:
            if loopback in url:
                rewritten = url.replace(loopback, override_host)
                logger.info(
                    "Webhook URL rewritten via OBELIX_WEBHOOK_REWRITE_HOST: {} → {}",
                    url,
                    rewritten,
                )
                return rewritten
        return url

    # 2. Check if URL contains a loopback address
    has_loopback = any(lb in url for lb in _LOOPBACK_HOSTS)
    if not has_loopback:
        return url

    # 3. Get client IP from middleware context
    client_ip = client_ip_var.get()
    if not client_ip:
        return url

    # 4. If client IP is also loopback, no rewrite needed (genuinely local)
    if client_ip in _LOOPBACK_HOSTS:
        return url

    # 5. Rewrite
    for loopback in _LOOPBACK_HOSTS:
        if loopback in url:
            return url.replace(loopback, client_ip)

    return url
