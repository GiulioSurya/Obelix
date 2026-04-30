"""Process-wide registry of known remote A2A agents and in-flight task tokens."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from obelix.adapters.outbound.a2a.state import TokenRoute
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.client import Client as A2AClient
    from a2a.types import AgentCard

logger = get_logger(__name__)


class RemoteAgentRegistry:
    """Singleton registry. Created in AgentFactory.a2a_serve before uvicorn
    starts. Owns AgentCards, A2A clients, and the in-flight token map.

    Reuses the httpx.AsyncClient already created by a2a_serve for the
    SmartPushNotificationSender — connection pooling matters and double
    clients waste fds.
    """

    def __init__(
        self,
        urls: list[str],
        httpx_client: httpx.AsyncClient,
    ) -> None:
        self._urls = urls
        self._httpx = httpx_client
        self._cards: dict[str, AgentCard] = {}
        self._clients: dict[str, A2AClient] = {}
        self._token_map: dict[str, TokenRoute] = {}
        # Used by T5 resolve_all() to serialize concurrent agent-card fetches.
        self._lock = asyncio.Lock()

    # ── Token map ─────────────────────────────────────────────────────────

    def register_token(self, token: str, *, context_id: str, agent_name: str) -> None:
        """Register a new token before send_message is called.

        Overwrites any existing entry with the same token string. For 256-bit
        random tokens (secrets.token_urlsafe(32)), collision in practice is
        impossible; a duplicate string here would indicate a programming error.
        """
        self._token_map[token] = TokenRoute(
            context_id=context_id,
            agent_name=agent_name,
            task_id=None,
            registered_at=datetime.now(UTC),
        )

    def claim_task_id(self, token: str, task_id: str) -> None:
        """Fill in task_id on the route after send_message returns.

        No-op when the token isn't registered: a webhook may arrive before
        send_message returns its task_id, in which case the route was already
        consumed (and its caller used body.id as fallback).
        """
        route = self._token_map.get(token)
        if route is not None:
            route.task_id = task_id

    def lookup(self, token: str) -> TokenRoute | None:
        return self._token_map.get(token)

    def revoke(self, token: str) -> None:
        self._token_map.pop(token, None)

    async def gc_expired(self, ttl_seconds: int = 86400) -> int:
        """Remove tokens older than TTL. Returns count removed.

        Declared async so callers (PollingWorker, periodic task) can await it
        uniformly with their other async housekeeping. Contains no I/O — all
        work is in-memory.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=ttl_seconds)
        expired = [
            tok
            for tok, route in self._token_map.items()
            if route.registered_at < cutoff
        ]
        for tok in expired:
            self._token_map.pop(tok, None)
        if expired:
            logger.info(f"[A2A] GC removed {len(expired)} expired tokens")
        return len(expired)
