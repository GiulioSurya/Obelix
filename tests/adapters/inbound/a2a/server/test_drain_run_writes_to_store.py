"""When _run_drain_task drives a real agent loop, the produced artifact and
final state must land in the SDK TaskStore — observable via tasks/get(T3).

Iron rule: real SDK + real BaseAgent + hand-written canned provider (no mocks).
"""

from __future__ import annotations

import asyncio

import pytest
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import (
    Task,
    TaskState,
    TaskStatus,
)

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


def _build_executor_with_agent(task_store, agent_response_text: str):
    """Build a real ObelixAgentExecutor whose agent_factory returns a real
    BaseAgent that replies with a fixed string. We avoid LLM dependence by
    using a hand-written provider that returns a canned AssistantMessage.
    """
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import (
        AssistantMessage,
    )
    from obelix.infrastructure.providers import Providers
    from obelix.ports.outbound.llm_provider import AbstractLLMProvider

    class _CannedProvider(AbstractLLMProvider):
        @property
        def provider_type(self):
            return Providers.ANTHROPIC

        @property
        def model_id(self) -> str:
            return "canned-1"

        async def invoke(self, messages, tools=None, response_schema=None):
            # Return a plain AssistantMessage with text content and no tool calls;
            # BaseAgent will detect "no tool calls" as the terminal condition and
            # build the final AssistantResponse itself.
            return AssistantMessage(content=agent_response_text)

    def _factory():
        return BaseAgent(
            system_message="you are a test agent",
            provider=_CannedProvider(),
        )

    return ObelixAgentExecutor(
        agent_factory=_factory,
        task_store=task_store,
    )


@pytest.mark.asyncio
async def test_drain_run_artifact_lands_in_task_store():
    store = InMemoryTaskStore()
    parent = Task(
        id="t1",
        context_id="ctx-x",
        status=TaskStatus(state=TaskState.completed),
        metadata=None,
    )
    await store.save(parent)

    executor = _build_executor_with_agent(store, "drain-result-text")
    entry = ContextEntry()
    entry.history = []
    # Seed a synthetic remote-task notification so the drainer doesn't
    # short-circuit on "nothing to drain".
    from obelix.core.model.human_message import HumanMessage

    entry.pending_notifications = [
        HumanMessage(content="<remote_task_update>foo</remote_task_update>")
    ]
    entry.current_task_id = "t1"

    new_id = await executor.spawn_drain_task(
        entry=entry,
        context_id="ctx-x",
        parent_task_id="t1",
    )

    # Wait for fire-and-forget background task to finish writing its result.
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        t3 = await store.get(new_id)
        if t3 and t3.status.state in (
            TaskState.completed,
            TaskState.failed,
            TaskState.canceled,
            TaskState.rejected,
        ):
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f"drain task {new_id} did not reach terminal state in 5s")

    refreshed = await store.get(new_id)
    assert refreshed is not None
    assert refreshed.status.state == TaskState.completed, (
        f"expected completed, got {refreshed.status.state}"
    )
    # Artifact should contain the canned text.
    arts = refreshed.artifacts or []
    flat = ""
    for a in arts:
        for p in a.parts:
            if hasattr(p.root, "text") and p.root.text:
                flat += p.root.text
    assert "drain-result-text" in flat, (
        f"expected canned text in artifact, got artifacts={arts!r}"
    )
