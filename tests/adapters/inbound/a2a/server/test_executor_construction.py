"""Verify ObelixAgentExecutor accepts a task_store kwarg and exposes it
internally for the metadata-patch helpers (introduced in later tasks).

Iron rule: real SDK types only - no MagicMock substitution for TaskStore.
"""

from __future__ import annotations

from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore

from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.core.agent.base_agent import BaseAgent
from obelix.infrastructure.providers import Providers


class _MinimalProvider:
    """Hand-written minimal provider for construction-only tests.

    BaseAgent only reads ``provider.model_id`` at construction time
    (to seed ``AgentUsage``). No invoke is exercised by these tests, so
    we satisfy that single attribute and nothing else - per the iron
    rule no mock libraries are used.
    """

    @property
    def provider_type(self) -> Providers:
        return Providers.ANTHROPIC

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def invoke(self, messages, tools=None, response_schema=None):  # noqa: ANN001
        raise NotImplementedError

    async def invoke_stream(self, messages, tools=None, response_schema=None):  # noqa: ANN001
        raise NotImplementedError
        yield  # pragma: no cover


def _agent_factory() -> BaseAgent:
    # A trivial agent - real construction, no mocks.
    return BaseAgent(system_message="x", provider=_MinimalProvider())


def test_executor_accepts_task_store():
    store = InMemoryTaskStore()
    executor = ObelixAgentExecutor(
        agent_factory=_agent_factory,
        task_store=store,
    )
    assert executor._task_store is store


def test_executor_no_longer_accepts_httpx_client():
    """Regression: spec 2 removes the httpx_client slot. Passing it must
    raise (TypeError on unexpected kwarg)."""
    import httpx
    import pytest

    store = InMemoryTaskStore()
    with pytest.raises(TypeError, match="httpx_client"):
        ObelixAgentExecutor(
            agent_factory=_agent_factory,
            task_store=store,
            httpx_client=httpx.AsyncClient(),
        )


def test_executor_defaults_task_store_to_none():
    """Spec 2 keeps task_store optional so non-A2A code paths can construct
    the executor without an SDK store. Tasks 4-7 must guard against the
    None case; this test pins the default."""
    executor = ObelixAgentExecutor(agent_factory=_agent_factory)
    assert executor._task_store is None
