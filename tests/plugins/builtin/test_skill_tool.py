from pathlib import Path
from unittest.mock import MagicMock

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.plugins.builtin.skill_tool import DEFAULT_LISTING_BUDGET, make_skill_tool


def _mgr_with(*skills: Skill) -> SkillManager:
    provider = MagicMock()
    provider.discover.return_value = list(skills)
    return SkillManager(providers=[provider])


def _skill(
    name: str = "s",
    desc: str = "d",
    when: str | None = None,
    body: str = "body",
) -> Skill:
    return Skill(
        name=name,
        description=desc,
        body=body,
        base_dir=Path("/tmp/skills") / name,
        when_to_use=when,
    )


class TestSystemPromptFragment:
    def test_no_skills_empty_string(self):
        tool = make_skill_tool(_mgr_with())
        assert tool.system_prompt_fragment() == ""

    def test_with_skills_lists_all(self):
        mgr = _mgr_with(_skill("a", "desc a"), _skill("b", "desc b"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        assert "a" in frag
        assert "b" in frag
        assert "desc a" in frag
        assert "desc b" in frag

    def test_fragment_mentions_skill_tool_invocation(self):
        mgr = _mgr_with(_skill("a"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        assert "Skill" in frag  # references the tool name
        assert "Available skills" in frag

    def test_fragment_includes_when_to_use(self):
        mgr = _mgr_with(_skill("a", "desc", when="user asks X"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        assert "user asks X" in frag

    def test_fragment_tells_llm_when_to_use_tool(self):
        mgr = _mgr_with(_skill("a"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        # Nudges the model toward proactive invocation
        assert "invoke" in frag.lower() or "use" in frag.lower()

    def test_custom_budget_respected(self):
        long_desc = "x" * 5000
        mgr = _mgr_with(_skill("a", long_desc))
        tool = make_skill_tool(mgr, listing_budget=200)
        frag = tool.system_prompt_fragment()
        # Budget truncation kicks in
        assert len(frag) < 2000  # nowhere near 5000


class TestToolExposure:
    def test_has_tool_name(self):
        tool = make_skill_tool(_mgr_with(_skill("a")))
        assert getattr(tool, "tool_name", None) == "Skill"

    def test_has_tool_description(self):
        tool = make_skill_tool(_mgr_with(_skill("a")))
        desc = getattr(tool, "tool_description", "")
        assert desc  # non-empty

    def test_default_budget_is_8000_chars(self):
        assert DEFAULT_LISTING_BUDGET == 8_000
