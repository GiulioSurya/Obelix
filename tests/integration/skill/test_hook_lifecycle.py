"""End-to-end: a skill-with-hooks registers, fires, and is cleaned up per query."""

from pathlib import Path

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider(AbstractLLMProvider):
    """Scripted provider."""

    model_id = "stub-model"

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    @property
    def provider_type(self):
        return Providers.ANTHROPIC

    async def invoke(self, messages, tools, response_schema=None):
        if self._idx >= len(self._responses):
            raise RuntimeError("Unscripted provider call")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


@pytest.mark.asyncio
async def test_hook_registered_during_query_and_cleaned_at_end():
    """Skill with hooks: hook added on invoke, removed at QUERY_END."""
    provider = StubProvider(
        [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="Skill",
                        arguments={"name": "with_all_hooks", "args": ""},
                    )
                ],
            ),
            AssistantMessage(content="Done."),
        ]
    )
    agent = BaseAgent(
        system_message="X",
        provider=provider,
        skills_config=str(FIXTURES / "with_all_hooks"),
    )
    pre_counts = {event: len(agent._hooks[event]) for event in AgentEvent}

    await agent.execute_query_async("go")

    post_counts = {event: len(agent._hooks[event]) for event in AgentEvent}
    assert post_counts == pre_counts


@pytest.mark.asyncio
async def test_hook_fires_when_skill_event_triggers():
    """A skill declaring on_tool_error sees HumanMessage injected when a tool errors."""
    provider = StubProvider(
        [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="Skill",
                        arguments={"name": "with_all_hooks", "args": ""},
                    )
                ],
            ),
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t2",
                        name="DefinitelyNotARegisteredTool",
                        arguments={},
                    )
                ],
            ),
            AssistantMessage(content="Recovered and done."),
        ]
    )
    agent = BaseAgent(
        system_message="X",
        provider=provider,
        skills_config=str(FIXTURES / "with_all_hooks"),
    )

    await agent.execute_query_async("go")

    msgs = [str(getattr(m, "content", "")) for m in agent.conversation_history]
    assert any("tool-error" in m for m in msgs), (
        f"expected 'tool-error' hook-injected message in history, got: {msgs!r}"
    )


@pytest.mark.asyncio
async def test_two_queries_start_clean():
    """After Query 1 cleanup, Query 2 re-installs hooks from scratch."""
    provider_q1 = [
        AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="Skill",
                    arguments={"name": "with_all_hooks", "args": ""},
                )
            ],
        ),
        AssistantMessage(content="Q1 done."),
    ]
    provider_q2 = [
        AssistantMessage(content="Q2 just answers"),
    ]
    provider = StubProvider(provider_q1 + provider_q2)
    agent = BaseAgent(
        system_message="X",
        provider=provider,
        skills_config=str(FIXTURES / "with_all_hooks"),
    )

    pre_counts = {event: len(agent._hooks[event]) for event in AgentEvent}

    # Query 1
    await agent.execute_query_async("q1")
    # After Q1: hook counts back to pre-query baseline
    post_q1_counts = {event: len(agent._hooks[event]) for event in AgentEvent}
    assert post_q1_counts == pre_counts

    # Query 2 - skill not invoked, should work normally
    r2 = await agent.execute_query_async("q2")
    assert "Q2 just answers" in r2.content
    post_q2_counts = {event: len(agent._hooks[event]) for event in AgentEvent}
    assert post_q2_counts == pre_counts
