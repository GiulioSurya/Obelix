"""Snapshot tests lock the skill listing output format.

Changes here force explicit review of any format change (prompt drift
protection).
"""

from pathlib import Path
from unittest.mock import MagicMock

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill


def _mk(name: str, desc: str, when: str | None = None) -> Skill:
    return Skill(
        name=name,
        description=desc,
        body="body",
        base_dir=Path("/"),
        when_to_use=when,
    )


def _fake_provider(skills: list[Skill]):
    p = MagicMock()
    p.discover.return_value = list(skills)
    return p


def test_canonical_listing_shape():
    """Three skills under generous budget — exact full rendering locked in."""
    skills = [
        _mk("alpha", "The alpha skill"),
        _mk("beta", "Beta helper", when="when beta-ing"),
        _mk("gamma", "Gamma processor"),
    ]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(10_000)
    expected = (
        "- alpha: The alpha skill\n"
        "- beta: Beta helper — when beta-ing\n"
        "- gamma: Gamma processor"
    )
    assert out == expected


def test_names_only_fallback_shape():
    """At extreme budget all descriptions drop; only `- <name>` remains."""
    long = "x" * 400
    skills = [_mk(f"s{i}", long) for i in range(10)]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(20)  # extremely tight
    for i in range(10):
        assert f"- s{i}" in out
    # No description tail
    assert "x" not in out


def test_empty_listing_returns_empty_string():
    mgr = SkillManager(providers=[_fake_provider([])])
    assert mgr.format_listing(10_000) == ""


def test_listing_sorted_alphabetically():
    """Discovery order must not affect final listing order."""
    skills = [_mk("zeta", "z"), _mk("alpha", "a"), _mk("mu", "m")]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(10_000)
    # Order: alpha, mu, zeta
    assert out.index("alpha") < out.index("mu") < out.index("zeta")


def test_truncation_preserves_all_names():
    """When budget forces description truncation, names stay intact."""
    long = "x" * 400
    skills = [_mk(f"sk{i:02d}", long) for i in range(5)]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    # Enough per-entry budget that descriptions truncate with "…"
    out = mgr.format_listing(200)
    for i in range(5):
        assert f"sk{i:02d}" in out


def test_truncation_marker_is_ellipsis():
    """The truncation marker is U+2026 (…), not ASCII '...'. Locked-in glyph."""
    long = "x" * 400
    skills = [_mk(f"sk{i}", long) for i in range(5)]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(200)
    assert "…" in out
    assert "..." not in out


def test_listing_never_leaks_body():
    """The skill body must never appear in the listing — only name/description/when_to_use."""
    body_marker = "SECRET_BODY_CONTENT_XYZ"
    skills = [
        Skill(
            name="alpha",
            description="desc",
            body=body_marker,
            base_dir=None,
            when_to_use="when Xing",
        )
    ]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(10_000)
    assert body_marker not in out


def test_when_to_use_uses_em_dash_separator():
    """When `when_to_use` is present, the separator is '—' (em-dash)."""
    skills = [_mk("a", "desc", when="use X")]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(10_000)
    assert "desc — use X" in out
