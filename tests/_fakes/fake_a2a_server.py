"""In-process A2A JSON-RPC server backed by InMemoryTaskStore.

Instantiate one per test; the .client attribute is a ready-to-use
a2a.client.Client wired through httpx.ASGITransport — no network.

Iron rule: we never mock a2a.* symbols; the SDK runs against itself.
"""

from __future__ import annotations

from typing import Any

import httpx
from a2a.client import A2ACardResolver, Client, ClientConfig, ClientFactory
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers.default_request_handler import (
    DefaultRequestHandler,
)
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill


class _NoopExecutor(AgentExecutor):
    """Default executor used when a test only needs the request handlers
    (tasks/get, tasks/cancel) — never produces events, never runs an agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        return

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:  # noqa: D401
        return


def _default_card(url: str = "http://fake-a2a") -> AgentCard:
    return AgentCard(
        name="fake-agent",
        description="In-process fake A2A agent for tests.",
        url=url,
        version="0.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            push_notifications=False,  # spec-2 polling-only fixture
            supports_authenticated_extended_card=False,
        ),
        skills=[
            AgentSkill(
                id="echo",
                name="echo",
                description="Echo what you send.",
                tags=[],
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


class FakeA2AServer:
    """Test fixture: a Starlette A2A app + an SDK Client over ASGITransport.

    Public attributes:
        task_store: the SDK's InMemoryTaskStore (mutate directly to seed
            tasks or to assert on stored Task.metadata after the code under
            test ran).
        client: an a2a.client.Client wired to this app via ASGITransport.
        executor: the AgentExecutor handed to DefaultRequestHandler (override
            via constructor to inject custom behavior).

    Polling-only by default (matches the spec-2 CLI config). Pass
    ``streaming=True`` to test code that may opt into SSE.
    """

    def __init__(
        self,
        executor: AgentExecutor | None = None,
        *,
        card: AgentCard | None = None,
        streaming: bool = False,
    ) -> None:
        self.task_store = InMemoryTaskStore()
        self.executor = executor or _NoopExecutor()
        self._card = card or _default_card()
        self._handler = DefaultRequestHandler(
            agent_executor=self.executor,
            task_store=self.task_store,
        )
        self._app = A2AStarletteApplication(
            agent_card=self._card,
            http_handler=self._handler,
        ).build()
        self._transport = httpx.ASGITransport(app=self._app)
        self._httpx = httpx.AsyncClient(
            transport=self._transport,
            base_url=self._card.url,
            timeout=30.0,
        )
        self._streaming = streaming
        self.client: Client | None = None  # populated in __aenter__; use 'async with'

    async def __aenter__(self) -> FakeA2AServer:
        resolver = A2ACardResolver(httpx_client=self._httpx, base_url=self._card.url)
        # The card resolver fetches /.well-known/agent-card; we already have
        # the card object, but a fresh fetch through the ASGI transport
        # exercises the full path the way the CLI does.
        card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=self._httpx,
            streaming=self._streaming,
            polling=not self._streaming,
            push_notification_configs=[],
        )
        self.client = ClientFactory(config).create(card)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._httpx.aclose()


def make_server(
    executor: AgentExecutor | None = None,
    **kwargs: Any,
) -> FakeA2AServer:
    """Module-level helper for symmetry with pytest fixture style."""
    return FakeA2AServer(executor, **kwargs)
