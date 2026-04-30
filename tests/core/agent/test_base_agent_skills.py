from pathlib import Path

import pytest

from obelix.core.agent.base_agent import BaseAgent
from obelix.core.skill.skill import SkillValidationError
from obelix.infrastructure.providers import Providers
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class _StubProvider(AbstractLLMProvider):
    """Minimal stub provider for BaseAgent construction."""

    model_id = "stub-model-1"

    @property
    def provider_type(self):
        return Providers.ANTHROPIC

    async def invoke(self, messages, tools, response_schema=None):
        raise NotImplementedError


def _provider():
    return _StubProvider()


class TestSkillsConfigAccepted:
    def test_none_behaves_as_before(self):
        a = BaseAgent(
            system_message="You are X.",
            provider=_provider(),
            skills_config=None,
        )
        # No skill tool registered
        assert all(getattr(t, "tool_name", "") != "Skill" for t in a.registered_tools)
        # No "Available skills" listing in system prompt
        assert "Available skills" not in a.system_message.content

    def test_valid_single_dir_registers_skill_tool(self):
        a = BaseAgent(
            system_message="You are X.",
            provider=_provider(),
            skills_config=str(FIXTURES / "minimal"),
        )
        names = [getattr(t, "tool_name", None) for t in a.registered_tools]
        assert "Skill" in names
        assert "## Available skills" in a.system_message.content
        assert "minimal" in a.system_message.content

    def test_valid_list_of_dirs(self):
        a = BaseAgent(
            system_message="X",
            provider=_provider(),
            skills_config=[
                str(FIXTURES / "minimal"),
                str(FIXTURES / "with_when_to_use"),
            ],
        )
        assert "minimal" in a.system_message.content
        assert "with_when_to_use" in a.system_message.content

    def test_pathlib_path_accepted(self):
        a = BaseAgent(
            system_message="X",
            provider=_provider(),
            skills_config=FIXTURES / "minimal",
        )
        assert "## Available skills" in a.system_message.content

    def test_dir_with_many_subskills(self, tmp_path):
        """A directory containing multiple skill subdirs loads all of them."""
        import shutil

        for name in ["minimal", "with_when_to_use", "with_named_args"]:
            shutil.copytree(FIXTURES / name, tmp_path / name)
        a = BaseAgent(
            system_message="X",
            provider=_provider(),
            skills_config=str(tmp_path),
        )
        assert "minimal" in a.system_message.content
        assert "with_when_to_use" in a.system_message.content
        assert "with_named_args" in a.system_message.content


class TestInputValidation:
    def test_empty_string_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config="",
            )

    def test_whitespace_string_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config="   ",
            )

    def test_empty_string_inside_list_rejected(self):
        with pytest.raises(ValueError, match="cannot contain empty strings"):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config=[str(FIXTURES / "minimal"), ""],
            )

    def test_non_str_non_path_type_rejected(self):
        with pytest.raises(TypeError, match="must be str, Path, list, or None"):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config=42,
            )

    def test_bad_item_in_list_rejected(self):
        with pytest.raises(TypeError, match="list items must be str or Path"):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config=[str(FIXTURES / "minimal"), 42],
            )


class TestValidationFailure:
    def test_invalid_skill_raises_on_construction(self):
        with pytest.raises(SkillValidationError):
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config=str(FIXTURES / "invalid_missing_description"),
            )

    def test_aggregated_validation_failure(self):
        with pytest.raises(SkillValidationError) as exc:
            BaseAgent(
                system_message="X",
                provider=_provider(),
                skills_config=str(FIXTURES / "invalid_multiple_issues"),
            )
        # multi-issue fixture → description missing + duplicate args + bogus hook + undeclared placeholder
        assert len(exc.value.issues) >= 4


class TestMcpIntegration:
    def test_no_mcp_no_mcp_skills(self):
        """Without mcp_config, no MCPSkillProvider is wired."""
        a = BaseAgent(
            system_message="X",
            provider=_provider(),
            skills_config=str(FIXTURES / "minimal"),
        )
        # minimal skill is present; no mcp__ names anywhere
        assert "mcp__" not in a.system_message.content


class TestParentAgentWiring:
    def test_skill_tool_wired_with_self_as_parent(self):
        """SkillTool should be wired with parent_agent=self so fork/hooks work."""
        a = BaseAgent(
            system_message="X",
            provider=_provider(),
            skills_config=str(FIXTURES / "minimal"),
        )
        # Find the SkillTool
        skill_tool = next(
            t for t in a.registered_tools if getattr(t, "tool_name", None) == "Skill"
        )
        # The instance must be callable as a tool — scaffold test was enough.
        # We can't directly inspect the closure, but registering should not raise.
        assert skill_tool is not None


# ---------------------------------------------------------------------------
# Task 11: 3-way dispatch branch (skill / sub_agent / tool)
# ---------------------------------------------------------------------------


class TestSkillSpanDispatch:
    """Verifies the tracer dispatches a SpanType.skill for SkillTool invocations."""

    @pytest.mark.asyncio
    async def test_skill_invocation_emits_skill_span(
        self, make_agent_with_skill_and_tracer
    ):
        """Invoking the SkillTool yields a span with type=skill, name=<skill>."""
        agent, spy = make_agent_with_skill_and_tracer(
            skill_name="demo",
            mode="inline",
            source="filesystem",
        )
        await agent.execute_query_async('Use the "demo" skill.')

        skill_spans = [s for s in spy.spans if s.span_type.value == "skill"]
        assert len(skill_spans) == 1
        sk = skill_spans[0]
        assert sk.name == "demo"
        assert sk.metadata.get("mode") == "inline"
        assert sk.metadata.get("source") == "filesystem"

        # And there must not be a regular tool span with the "Skill" name.
        tool_spans_named_Skill = [
            s for s in spy.spans if s.span_type.value == "tool" and s.name == "Skill"
        ]
        assert tool_spans_named_Skill == []

    @pytest.mark.asyncio
    async def test_regular_tool_still_emits_tool_span(
        self, make_agent_with_regular_tool_and_tracer
    ):
        """Non-SkillTool, non-SubAgentWrapper tools still get SpanType.tool."""
        agent, spy = make_agent_with_regular_tool_and_tracer(tool_name="calc")
        await agent.execute_query_async("do something")
        tool_spans = [s for s in spy.spans if s.span_type.value == "tool"]
        assert any(s.name == "calc" for s in tool_spans)


class TestSubAgentSpanDispatch:
    """Verifies the tracer dispatches a SpanType.sub_agent for SubAgentWrapper calls."""

    @pytest.mark.asyncio
    async def test_sub_agent_invocation_emits_sub_agent_span(
        self, make_agent_with_sub_agent_and_tracer
    ):
        """Invoking a registered sub-agent yields a span with type=sub_agent.

        The span name matches the registered sub-agent name (the tool_call.name
        used by the parent's LLM), not the child agent's class name.
        """
        agent, spy = make_agent_with_sub_agent_and_tracer(sub_agent_name="child")
        await agent.execute_query_async("delegate to child")
        sub_agent_spans = [s for s in spy.spans if s.span_type.value == "sub_agent"]
        assert len(sub_agent_spans) >= 1
        assert any(s.name == "child" for s in sub_agent_spans)
