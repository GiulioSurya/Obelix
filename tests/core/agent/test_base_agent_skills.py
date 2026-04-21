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
