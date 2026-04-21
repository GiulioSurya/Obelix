from pathlib import Path

import pytest

from obelix.core.skill.reporting import format_validation_error
from obelix.core.skill.skill import SkillIssue


class TestFormatValidationError:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            format_validation_error([])

    def test_single_issue_renders_header(self):
        out = format_validation_error(
            [
                SkillIssue(
                    file_path=Path("skills/a/SKILL.md"),
                    field="frontmatter.description",
                    message="missing",
                )
            ]
        )
        assert "1 skill(s) failed validation" in out
        assert "1 issue(s)" in out
        assert "skills/a/SKILL.md" in out
        assert "[frontmatter.description]" in out
        assert "missing" in out

    def test_multiple_issues_same_file_grouped(self):
        issues = [
            SkillIssue(file_path=Path("a/SKILL.md"), field="x", message="m1"),
            SkillIssue(file_path=Path("a/SKILL.md"), field="y", message="m2"),
        ]
        out = format_validation_error(issues)
        assert out.count("a/SKILL.md") == 1
        assert "[x]" in out
        assert "[y]" in out

    def test_multi_file_sorted_alphabetically(self):
        issues = [
            SkillIssue(file_path=Path("b/SKILL.md"), field="x", message="m"),
            SkillIssue(file_path=Path("a/SKILL.md"), field="x", message="m"),
        ]
        out = format_validation_error(issues)
        a_idx = out.index("a/SKILL.md")
        b_idx = out.index("b/SKILL.md")
        assert a_idx < b_idx

    def test_issue_with_line_rendered(self):
        out = format_validation_error(
            [
                SkillIssue(
                    file_path=Path("a.md"), field="frontmatter", message="bad", line=5
                )
            ]
        )
        assert "line 5" in out

    def test_issue_without_line_no_line_suffix(self):
        out = format_validation_error(
            [SkillIssue(file_path=Path("a.md"), field="x", message="bad")]
        )
        assert "(line " not in out

    def test_mcp_issue_has_mcp_section(self):
        out = format_validation_error(
            [SkillIssue(file_path=None, field="x", message="bad")]
        )
        assert "[mcp]" in out

    def test_counts_skills_not_issues(self):
        issues = [
            SkillIssue(file_path=Path("a.md"), field="x", message="m"),
            SkillIssue(file_path=Path("a.md"), field="y", message="m"),
            SkillIssue(file_path=Path("b.md"), field="z", message="m"),
        ]
        out = format_validation_error(issues)
        assert "2 skill(s)" in out
        assert "3 issue(s)" in out

    def test_preserves_issue_order_within_group(self):
        issues = [
            SkillIssue(file_path=Path("a.md"), field="z", message="mz"),
            SkillIssue(file_path=Path("a.md"), field="a", message="ma"),
        ]
        out = format_validation_error(issues)
        z_idx = out.index("[z]")
        a_idx = out.index("[a]")
        assert z_idx < a_idx

    def test_mcp_and_fs_mixed(self):
        issues = [
            SkillIssue(file_path=None, field="mcp_field", message="bad"),
            SkillIssue(file_path=Path("a.md"), field="fs_field", message="bad"),
        ]
        out = format_validation_error(issues)
        assert "[mcp]" in out
        assert "a.md" in out

    def test_path_rendered_as_posix_cross_platform(self):
        """Locks in the as_posix() fix — no backslashes on Windows."""
        out = format_validation_error(
            [
                SkillIssue(
                    file_path=Path("skills") / "deep" / "SKILL.md",
                    field="x",
                    message="bad",
                )
            ]
        )
        assert "skills/deep/SKILL.md" in out
        assert "\\" not in out  # no Windows-style separator leaks

    def test_mcp_section_sorts_before_lowercase_paths(self):
        """'[mcp]' starts with '[' (0x5B) so it sorts before lowercase file paths."""
        issues = [
            SkillIssue(file_path=Path("skills/a.md"), field="fs", message="m"),
            SkillIssue(file_path=None, field="mcp_field", message="m"),
        ]
        out = format_validation_error(issues)
        mcp_idx = out.index("[mcp]")
        fs_idx = out.index("skills/a.md")
        assert mcp_idx < fs_idx
