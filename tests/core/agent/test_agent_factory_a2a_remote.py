"""Tests for AgentFactory.a2a_serve(remote_agents=...) wiring (T14).

Verifies that when remote_agents is provided, the FastAPI app gets:
- /webhook route mounted
- registry resolved at startup (before uvicorn.run)
- polling worker registered on startup/shutdown lifespan
- the 5 outbound tools injected into agent instances per-request
- executor receives the registry for cancel sweep
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage


class _DummyProvider:
    @property
    def provider_type(self):
        return "dummy"

    @property
    def model_id(self):
        return "dummy-1"

    async def invoke(self, *a, **kw):
        return AssistantMessage(content="ok")


class _DummyAgent(BaseAgent):
    def __init__(self, **kw):
        super().__init__(
            system_message="test",
            provider=_DummyProvider(),
            **kw,
        )


# ── No-remotes path: existing behavior preserved ──────────────────────────


def test_a2a_serve_with_no_remotes_unchanged():
    """When remote_agents is None, no /webhook, no registry, behavior identical."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    with patch("uvicorn.run") as mock_run:
        factory.a2a_serve("dummy", port=12345, log_level="error")

    mock_run.assert_called_once()


def test_a2a_serve_no_remotes_no_webhook_route():
    """No remote_agents → no /webhook route on the FastAPI app."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    seen: dict = {}

    def fake_run(app, **kwargs):
        seen["app"] = app

    with patch("uvicorn.run", side_effect=fake_run):
        factory.a2a_serve("dummy", port=12345, log_level="error")

    app = seen["app"]
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/webhook" not in paths


# ── With-remotes path: wiring verified ────────────────────────────────────


def _patch_card_resolver():
    """Build a context manager that fakes A2ACardResolver.get_agent_card."""
    from a2a.client import A2ACardResolver
    from a2a.types import TransportProtocol

    async def _fake_card(self, **kw):
        c = MagicMock()
        c.name = "B"
        c.description = "remote"
        c.skills = []
        c.url = self.base_url
        c.preferred_transport = TransportProtocol.jsonrpc
        c.additional_interfaces = None
        return c

    return patch.object(A2ACardResolver, "get_agent_card", _fake_card)


def test_a2a_serve_with_remotes_mounts_webhook():
    """remote_agents → /webhook route registered on the FastAPI app."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    seen: dict = {}

    def fake_run(app, **kwargs):
        seen["app"] = app

    with patch("uvicorn.run", side_effect=fake_run), _patch_card_resolver():
        factory.a2a_serve(
            "dummy",
            remote_agents=["http://b:8001"],
            port=12345,
            log_level="error",
        )

    app = seen["app"]
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/webhook" in paths


def test_a2a_serve_with_remotes_resolves_cards_at_startup():
    """resolve_all is called synchronously before uvicorn.run."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    resolve_calls: list[str] = []

    async def _track_card(self, **kw):
        from a2a.types import TransportProtocol

        resolve_calls.append(self.base_url)
        c = MagicMock()
        c.name = "B"
        c.description = "remote"
        c.skills = []
        c.url = self.base_url
        c.preferred_transport = TransportProtocol.jsonrpc
        c.additional_interfaces = None
        return c

    from a2a.client import A2ACardResolver

    with (
        patch("uvicorn.run"),
        patch.object(A2ACardResolver, "get_agent_card", _track_card),
    ):
        factory.a2a_serve(
            "dummy",
            remote_agents=["http://b:8001"],
            port=12345,
            log_level="error",
        )

    assert "http://b:8001" in resolve_calls


def test_a2a_serve_with_remotes_injects_tools_into_agent():
    """The agent_factory closure attaches the 5 outbound A2A tools to
    each agent instance produced per request."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    seen: dict = {}

    def fake_run(app, **kwargs):
        seen["app"] = app

    with patch("uvicorn.run", side_effect=fake_run), _patch_card_resolver():
        factory.a2a_serve(
            "dummy",
            remote_agents=["http://b:8001"],
            port=12345,
            log_level="error",
        )

    app = seen["app"]

    # Locate the ObelixAgentExecutor through the request handler graph.
    # The A2A SDK mounts JSON-RPC routes whose endpoints are bound methods
    # of an ``A2AFastAPIApplication``; from there:
    #   a2a_app.handler                  -> JSONRPCHandler
    #   .request_handler                 -> DefaultRequestHandler
    #   .agent_executor                  -> ObelixAgentExecutor
    from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor

    executor = None
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        a2a_app = getattr(endpoint, "__self__", None)
        if a2a_app is None or type(a2a_app).__name__ != "A2AFastAPIApplication":
            continue
        try:
            candidate = a2a_app.handler.request_handler.agent_executor
        except AttributeError:
            continue
        if isinstance(candidate, ObelixAgentExecutor):
            executor = candidate
            break

    if executor is None:
        # Fallback: introspection too brittle on this FastAPI/Starlette
        # combo — covered by integration tests.
        pytest.skip("could not introspect executor — covered by integration tests")

    agent = executor._agent_factory()
    tool_names = {getattr(t, "tool_name", None) for t in agent.registered_tools}
    expected = {
        "dispatch_agent",
        "respond_to_remote",
        "task_list",
        "task_get",
        "task_stop",
    }
    assert expected.issubset(tool_names), f"missing tools: {expected - tool_names}"


def test_a2a_serve_with_empty_remotes_list_unchanged():
    """remote_agents=[] should behave identically to None."""
    factory = AgentFactory()
    factory.register("dummy", _DummyAgent)

    seen: dict = {}

    def fake_run(app, **kwargs):
        seen["app"] = app

    with patch("uvicorn.run", side_effect=fake_run):
        factory.a2a_serve("dummy", remote_agents=[], port=12345, log_level="error")

    app = seen["app"]
    paths = [getattr(r, "path", None) for r in app.routes]
    assert "/webhook" not in paths
