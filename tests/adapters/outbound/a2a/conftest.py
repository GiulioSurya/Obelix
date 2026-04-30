"""Shared fixtures for outbound A2A unit tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry


def make_fake_card(
    name: str, *, description: str = "", skills: list[str] | None = None
):
    """Build a minimal AgentCard double for tests.

    a2a-sdk's real AgentCard has many required fields; we build only what
    the registry uses (name, description, skills, url).
    """
    from a2a.types import TransportProtocol

    skills_list = []
    for s in skills or []:
        skill = MagicMock()
        skill.name = s
        skill.description = f"skill {s}"
        skills_list.append(skill)
    card = MagicMock()
    card.name = name
    card.description = description
    card.skills = skills_list
    card.url = f"http://localhost/{name}"
    # ClientFactory.create() reads preferred_transport + additional_interfaces
    # to negotiate a transport. Default to JSONRPC (matches the client default).
    card.preferred_transport = TransportProtocol.jsonrpc
    card.additional_interfaces = None
    return card


@pytest.fixture
async def httpx_client():
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
async def registry(httpx_client: httpx.AsyncClient) -> RemoteAgentRegistry:
    return RemoteAgentRegistry(urls=[], httpx_client=httpx_client)


@pytest.fixture
def patched_resolver(monkeypatch: pytest.MonkeyPatch):
    """Replace A2ACardResolver.get_agent_card with a configurable side_effect."""
    cards: dict[str, Any] = {}
    calls: list[str] = []

    async def _fake_get_agent_card(self, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(self.base_url)
        if self.base_url in cards:
            return cards[self.base_url]
        raise RuntimeError(f"no fake card for {self.base_url}")

    from a2a.client import A2ACardResolver

    monkeypatch.setattr(A2ACardResolver, "get_agent_card", _fake_get_agent_card)

    class Patched:
        def add(self, base_url: str, card: Any) -> None:
            cards[base_url] = card

        @property
        def calls(self) -> list[str]:
            return calls

    return Patched()
