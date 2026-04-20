from pathlib import Path

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.ports.outbound.skill_provider import AbstractSkillProvider


def _skill(name: str, desc: str = "d", when: str | None = None, source="filesystem"):
    return Skill(
        name=name,
        description=desc,
        body="body",
        base_dir=Path("/"),
        when_to_use=when,
        source=source,
    )


class _FakeProvider(AbstractSkillProvider):
    def __init__(self, skills):
        self._skills = skills

    def discover(self):
        return list(self._skills)


class TestConstructionAndLookup:
    def test_empty(self):
        mgr = SkillManager(providers=[])
        assert mgr.list_all() == []
        assert mgr.load("x") is None
        assert mgr.format_listing(1000) == ""

    def test_single_provider(self):
        p = _FakeProvider([_skill("a"), _skill("b")])
        mgr = SkillManager(providers=[p])
        names = {s.name for s in mgr.list_all()}
        assert names == {"a", "b"}

    def test_load_known_and_unknown(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a")])])
        assert mgr.load("a") is not None
        assert mgr.load("missing") is None


class TestDedup:
    def test_disjoint_names(self):
        p1 = _FakeProvider([_skill("a")])
        p2 = _FakeProvider([_skill("b")])
        mgr = SkillManager(providers=[p1, p2])
        assert len(mgr.list_all()) == 2

    def test_collision_filesystem_wins_over_mcp(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        fs = _FakeProvider([_skill("a", desc="fs", source="filesystem")])
        mcp = _FakeProvider([_skill("a", desc="mcp", source="mcp")])
        mgr = SkillManager(providers=[fs, mcp])
        assert len(mgr.list_all()) == 1
        assert mgr.load("a").description == "fs"
        # Warning logged
        assert any(
            "collision" in r.message.lower() or "kept" in r.message.lower()
            for r in caplog.records
        )

    def test_collision_mcp_loses_when_filesystem_comes_second(self, caplog):
        """Even if MCP comes first in providers list, filesystem wins by source priority."""
        import logging

        caplog.set_level(logging.WARNING)
        mcp = _FakeProvider([_skill("a", desc="mcp", source="mcp")])
        fs = _FakeProvider([_skill("a", desc="fs", source="filesystem")])
        mgr = SkillManager(providers=[mcp, fs])
        assert len(mgr.list_all()) == 1
        assert mgr.load("a").description == "fs"

    def test_collision_provider_order_wins_within_same_source(self):
        p1 = _FakeProvider([_skill("a", desc="first")])
        p2 = _FakeProvider([_skill("a", desc="second")])
        mgr = SkillManager(providers=[p1, p2])
        assert mgr.load("a").description == "first"


class TestListingDeterminism:
    def test_deterministic_order(self):
        mgr = SkillManager(
            providers=[_FakeProvider([_skill("c"), _skill("a"), _skill("b")])]
        )
        call1 = mgr.format_listing(1000)
        call2 = mgr.format_listing(1000)
        assert call1 == call2

    def test_sorted_alphabetically(self):
        mgr = SkillManager(
            providers=[_FakeProvider([_skill("c"), _skill("a"), _skill("b")])]
        )
        names = [s.name for s in mgr.list_all()]
        assert names == ["a", "b", "c"]


class TestListingBudget:
    def test_under_budget_all_full(self):
        skills = [_skill(f"s{i}", desc="short description") for i in range(3)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(10_000)
        assert "short description" in out
        for i in range(3):
            assert f"s{i}" in out

    def test_over_budget_truncates_descriptions(self):
        long_desc = "a" * 500
        skills = [_skill(f"s{i}", desc=long_desc) for i in range(20)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(200)  # small
        # All names preserved
        for i in range(20):
            assert f"s{i}" in out

    def test_extreme_budget_names_only(self):
        skills = [_skill(f"s{i}", desc="a" * 100) for i in range(50)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(10)
        # With budget this small, names-only fallback
        assert "s0" in out

    def test_medium_budget_truncates_with_ellipsis(self):
        """Budget that allows truncated descriptions but not full ones."""
        long_desc = "a" * 200
        skills = [_skill(f"s{i}", desc=long_desc) for i in range(3)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        # 3 skills: name_overhead ~= 3 * (2+4) + 2 = 20, budget 200 -> ~60 each desc
        out = mgr.format_listing(200)
        # Names present
        for i in range(3):
            assert f"s{i}" in out
        # Descriptions truncated with ellipsis char
        assert "\u2026" in out
        # Description not full 200 chars long
        assert "a" * 200 not in out

    def test_zero_budget_empty_string(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a")])])
        assert mgr.format_listing(0) == ""

    def test_negative_budget_empty_string(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a")])])
        assert mgr.format_listing(-5) == ""


class TestWhenToUseRendered:
    def test_when_to_use_included(self):
        mgr = SkillManager(
            providers=[_FakeProvider([_skill("a", desc="d", when="use when X")])]
        )
        out = mgr.format_listing(1000)
        assert "use when X" in out

    def test_no_when_to_use_only_description(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a", desc="just desc")])])
        out = mgr.format_listing(1000)
        assert "just desc" in out
        # No trailing em-dash
        for line in out.splitlines():
            assert not line.endswith("— ")


class TestEntryFormat:
    def test_entry_line_starts_with_dash_and_name(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("mysk", desc="desc")])])
        out = mgr.format_listing(1000)
        assert out.startswith("- mysk:") or out.startswith("- mysk :")
