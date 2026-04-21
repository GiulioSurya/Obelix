"""End-to-end: agent with skills_config invokes a skill inline."""

from pathlib import Path

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider(AbstractLLMProvider):
    """Replays scripted responses; raises if asked for more."""

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
async def test_agent_invokes_inline_skill_end_to_end():
    """Script: turn 1 -> LLM calls Skill(minimal). Turn 2 -> LLM emits final text."""
    provider = StubProvider(
        [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        id="t1",
                        name="Skill",
                        arguments={"name": "minimal", "args": ""},
                    )
                ],
            ),
            AssistantMessage(content="Done with the minimal skill."),
        ]
    )

    agent = BaseAgent(
        system_message="You are a test agent.",
        provider=provider,
        skills_config=str(FIXTURES / "minimal"),
    )

    result = await agent.execute_query_async("please run minimal")
    assert "Done with the minimal skill" in result.content

    # The skill body must have appeared in the conversation history
    # (either as AssistantMessage content or a ToolMessage tool_results entry).
    def _extract_text(m):
        texts = [str(getattr(m, "content", "") or "")]
        for tr in getattr(m, "tool_results", []) or []:
            texts.append(str(getattr(tr, "result", "") or ""))
            texts.append(str(getattr(tr, "error", "") or ""))
        return "\n".join(texts)

    has_body = any(
        "Body of the minimal skill" in _extract_text(m)
        for m in agent.conversation_history
    )
    assert has_body, "skill body not injected into conversation history"


@pytest.mark.asyncio
async def test_agent_without_skills_config_unaffected():
    """Sanity: skills_config=None does not add anything."""
    provider = StubProvider([AssistantMessage(content="hi")])
    agent = BaseAgent(
        system_message="You are X.",
        provider=provider,
        skills_config=None,
    )
    assert "## Available skills" not in agent.system_message.content
    result = await agent.execute_query_async("ping")
    assert result.content == "hi"
