import dataclasses
from pathlib import Path

import pytest

from obelix.core.skill.skill import (
    Skill,
    SkillCandidate,
    SkillIssue,
    SkillValidationError,
)


class TestSkillCandidate:
    def test_holds_raw_frontmatter_and_body(self):
        c = SkillCandidate(
            file_path=Path("x.md"),
            frontmatter={"description": "foo"},
            body="hello",
        )
        assert c.frontmatter == {"description": "foo"}
        assert c.body == "hello"
        assert c.file_path == Path("x.md")

    def test_file_path_optional_for_mcp(self):
        c = SkillCandidate(file_path=None, frontmatter={}, body="")
        assert c.file_path is None


class TestSkillIssue:
    def test_requires_field_and_message(self):
        issue = SkillIssue(
            file_path=Path("x.md"),
            field="frontmatter.description",
            message="missing",
        )
        assert issue.field == "frontmatter.description"
        assert issue.message == "missing"
        assert issue.line is None

    def test_hashable(self):
        a = SkillIssue(file_path=None, field="x", message="y")
        b = SkillIssue(file_path=None, field="x", message="y")
        assert hash(a) == hash(b)


class TestSkill:
    def test_basic_construction(self):
        s = Skill(
            name="reviewer",
            description="Reviews code",
            body="# Review",
            base_dir=Path("/skills/reviewer"),
        )
        assert s.name == "reviewer"
        assert s.description == "Reviews code"
        assert s.body == "# Review"
        assert s.context == "inline"
        assert s.source == "filesystem"
        assert s.arguments == ()
        assert s.hooks == {}
        assert s.when_to_use is None

    def test_frozen(self):
        s = Skill(name="a", description="b", body="c", base_dir=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.name = "x"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = Skill(name="x", description="y", body="z", base_dir=None)
        b = Skill(name="x", description="y", body="z", base_dir=None)
        assert a == b

    def test_from_candidate(self):
        candidate = SkillCandidate(
            file_path=Path("skills/r/SKILL.md"),
            frontmatter={
                "description": "Reviews",
                "when_to_use": "when needed",
                "arguments": ["path"],
                "context": "fork",
                "hooks": {"on_tool_error": "retry"},
                "allowed_tools": ["Read"],
            },
            body="# body",
        )
        s = Skill.from_candidate(candidate, name="r", base_dir=Path("skills/r"))
        assert s.name == "r"
        assert s.description == "Reviews"
        assert s.when_to_use == "when needed"
        assert s.arguments == ("path",)
        assert s.context == "fork"
        assert s.hooks == {"on_tool_error": "retry"}
        assert s.allowed_tools == ("Read",)
        assert s.base_dir == Path("skills/r")
        assert s.file_path == Path("skills/r/SKILL.md")
        assert s.source == "filesystem"

    def test_from_candidate_minimal_frontmatter(self):
        candidate = SkillCandidate(
            file_path=None,
            frontmatter={"description": "Only description"},
            body="body",
        )
        s = Skill.from_candidate(candidate, name="x", base_dir=None)
        assert s.description == "Only description"
        assert s.when_to_use is None
        assert s.arguments == ()
        assert s.allowed_tools == ()
        assert s.context == "inline"
        assert s.hooks == {}
        assert s.source == "filesystem"
        assert s.file_path is None
        assert s.base_dir is None

    def test_from_candidate_missing_description_raises(self):
        candidate = SkillCandidate(
            file_path=None,
            frontmatter={},  # no description
            body="body",
        )
        with pytest.raises(KeyError):
            Skill.from_candidate(candidate, name="x", base_dir=None)


class TestSkillValidationError:
    def test_empty_issues_raises_value_error(self):
        with pytest.raises(ValueError):
            SkillValidationError([])

    def test_carries_issues(self):
        issues = [SkillIssue(file_path=None, field="x", message="y")]
        err = SkillValidationError(issues)
        assert err.issues == issues

    def test_is_exception(self):
        err = SkillValidationError([SkillIssue(file_path=None, field="x", message="y")])
        assert isinstance(err, Exception)
