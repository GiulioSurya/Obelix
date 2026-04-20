from unittest.mock import MagicMock

from obelix.adapters.outbound.mcp.manager import MCPPrompt
from obelix.adapters.outbound.skill.mcp import MCPSkillProvider


class TestMCPSkillProviderEmpty:
    def test_no_manager_empty(self):
        provider = MCPSkillProvider(None)
        assert provider.discover() == []

    def test_disconnected_manager_empty(self):
        mgr = MagicMock()
        mgr.list_prompts.return_value = []
        provider = MCPSkillProvider(mgr)
        assert provider.discover() == []


class TestMCPSkillProviderHappy:
    def test_well_formed_prompt_becomes_skill(self):
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="review",
            description="Reviews code",
            arguments=[],
            server_name="tracer",
            template="# Review\n\nDo a review.",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "mcp__tracer__review"
        assert s.description == "Reviews code"
        assert s.source == "mcp"
        assert s.base_dir is None


class TestMCPSkillProviderSkipMalformed:
    def test_missing_description_skipped_with_log(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="bad",
            description="",
            arguments=[],
            server_name="s",
            template="body",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert skills == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_empty_template_skipped_with_log(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="bad",
            description="hi",
            arguments=[],
            server_name="s",
            template="",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert skills == []

    def test_whitespace_only_template_skipped(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="bad",
            description="hi",
            arguments=[],
            server_name="s",
            template="   \n\n",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        assert provider.discover() == []


class TestMCPSkillProviderArguments:
    def test_prompt_arguments_become_skill_arguments(self):
        mgr = MagicMock()
        arg1 = MagicMock()
        arg1.name = "topic"
        arg2 = MagicMock()
        arg2.name = "depth"
        prompt = MCPPrompt(
            name="review",
            description="d",
            arguments=[arg1, arg2],
            server_name="srv",
            template="Review $topic at $depth",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert len(skills) == 1
        assert skills[0].arguments == ("topic", "depth")

    def test_arguments_without_name_attr_skipped(self):
        mgr = MagicMock()
        bad_arg = MagicMock(spec=[])  # no .name attr
        prompt = MCPPrompt(
            name="p",
            description="d",
            arguments=[bad_arg],
            server_name="srv",
            template="body",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert len(skills) == 1
        # No valid arg names extracted
        assert skills[0].arguments == ()


class TestMCPSkillProviderMulti:
    def test_multiple_prompts_multiple_skills(self):
        mgr = MagicMock()
        p1 = MCPPrompt(
            name="a",
            description="da",
            arguments=[],
            server_name="s1",
            template="body a",
        )
        p2 = MCPPrompt(
            name="b",
            description="db",
            arguments=[],
            server_name="s2",
            template="body b",
        )
        mgr.list_prompts.return_value = [p1, p2]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"mcp__s1__a", "mcp__s2__b"}

    def test_mix_valid_and_invalid(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        mgr = MagicMock()
        good = MCPPrompt(
            name="good",
            description="hi",
            arguments=[],
            server_name="s",
            template="body",
        )
        bad = MCPPrompt(
            name="bad",
            description="",  # missing
            arguments=[],
            server_name="s",
            template="body",
        )
        mgr.list_prompts.return_value = [good, bad]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        # Good one passes, bad one skipped
        assert len(skills) == 1
        assert skills[0].name == "mcp__s__good"
