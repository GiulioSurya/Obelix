"""Process-wide registry of known remote A2A agents and in-flight task tokens."""

from __future__ import annotations

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

    def touch(self, token: str) -> None:
        """Refresh registered_at on the token's route. Call when activity
        on the route signals it's still in use (e.g., respond_to_remote,
        webhook update on non-terminal state). Prevents premature TTL GC
        on long-running input_required cycles."""
        route = self._token_map.get(token)
        if route is not None:
            route.registered_at = datetime.now(UTC)

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

    # ── Card resolution ───────────────────────────────────────────────────

    async def resolve_all(self) -> None:
        """Fetch /.well-known/agent-card.json for each URL and build clients.

        - Per-URL fetch failure: log warning, skip that agent.
        - Duplicate names: log warning, append `(N)` discriminator so
          replicated topologies (multi-AZ behind separate URLs) work
          (e.g. two cards named "B" become "B" and "B (1)").
        """
        from a2a.client import A2ACardResolver, ClientConfig, ClientFactory

        client_config = ClientConfig(
            httpx_client=self._httpx,
            streaming=False,
            polling=False,
        )
        factory = ClientFactory(client_config)

        for url in self._urls:
            try:
                resolver = A2ACardResolver(httpx_client=self._httpx, base_url=url)
                card = await resolver.get_agent_card()
            # Broad exception catch is intentional: any fetch-time failure
            # (network, timeout, JSON validation, transport negotiation)
            # should soft-fail this remote without aborting the rest of the
            # registry. The catch only wraps the resolver call (factory.create
            # below is intentionally unprotected — bugs there should propagate).
            except Exception as e:
                logger.warning(f"[A2A] AgentCard fetch failed | url={url} error={e}")
                continue

            base_name = getattr(card, "name", None) or url
            unique_name = base_name
            if base_name in self._cards:
                idx = 1
                while f"{base_name} ({idx})" in self._cards:
                    idx += 1
                unique_name = f"{base_name} ({idx})"
                logger.warning(
                    f"[A2A] duplicate AgentCard name | base_name={base_name!r} "
                    f"url={url} registered_as={unique_name!r}"
                )

            self._cards[unique_name] = card
            self._clients[unique_name] = factory.create(card)
            logger.info(f"[A2A] registered remote agent | name={unique_name} url={url}")

    def names(self) -> list[str]:
        return list(self._cards.keys())

    def card_for(self, name: str) -> AgentCard:
        try:
            return self._cards[name]
        except KeyError:
            raise KeyError(
                f"unknown remote agent {name!r}; known: {list(self._cards.keys())}"
            ) from None

    def client_for(self, name: str) -> A2AClient:
        try:
            return self._clients[name]
        except KeyError:
            raise KeyError(
                f"unknown remote agent {name!r}; known: {list(self._clients.keys())}"
            ) from None

    def descriptions(self) -> dict[str, dict]:
        """Returns {name: {description, skills}} for system_prompt_fragment."""
        out: dict[str, dict] = {}
        for name, card in self._cards.items():
            skills_list = [
                getattr(s, "name", str(s)) for s in (getattr(card, "skills", []) or [])
            ]
            out[name] = {
                "description": getattr(card, "description", "") or "",
                "skills": skills_list,
            }
        return out
