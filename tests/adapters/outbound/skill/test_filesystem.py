from pathlib import Path

import pytest

from obelix.adapters.outbound.skill.filesystem import FilesystemSkillProvider
from obelix.core.skill.skill import SkillValidationError

# Fixture dir is tests/fixtures/skills/
# __file__ = .../tests/adapters/outbound/skill/test_filesystem.py
# fixtures = .../tests/fixtures/skills/
FIXTURES = Path(__file__).parent.parent.parent.parent / "fixtures" / "skills"


class TestSingleFilePath:
    def test_md_file_directly(self):
        p = FIXTURES / "minimal" / "SKILL.md"
        provider = FilesystemSkillProvider([p])
        skills = provider.discover()
        assert len(skills) == 1
        assert skills[0].name == "minimal"
        assert skills[0].description == "Minimal skill"
        assert skills[0].base_dir == (FIXTURES / "minimal").resolve()


class TestSingleDirPath:
    def test_dir_with_skill_md(self):
        provider = FilesystemSkillProvider([FIXTURES / "minimal"])
        skills = provider.discover()
        assert len(skills) == 1
        assert skills[0].name == "minimal"


class TestValidationAggregation:
    def test_single_invalid_raises(self):
        provider = FilesystemSkillProvider([FIXTURES / "invalid_missing_description"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert len(exc.value.issues) >= 1
        assert any("description" in i.field for i in exc.value.issues)

    def test_multi_issue_skill_all_reported(self):
        provider = FilesystemSkillProvider([FIXTURES / "invalid_multiple_issues"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        # missing description + duplicate args + bogus hook + undeclared placeholder
        assert len(exc.value.issues) >= 4

    def test_mix_of_valid_and_invalid_all_invalid_aggregated(self):
        provider = FilesystemSkillProvider(
            [
                FIXTURES / "minimal",
                FIXTURES / "invalid_missing_description",
                FIXTURES / "invalid_duplicate_args",
            ]
        )
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        # Issues span 2 files (the two invalid ones)
        files = {i.file_path for i in exc.value.issues}
        assert len(files) == 2

    def test_invalid_yaml_reported_as_parse_issue(self):
        provider = FilesystemSkillProvider([FIXTURES / "invalid_yaml"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any("parse" in i.field for i in exc.value.issues)

    def test_empty_body_reported(self):
        provider = FilesystemSkillProvider([FIXTURES / "invalid_empty_body"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any(
            "body" in i.field and "empty" in i.message.lower() for i in exc.value.issues
        )


class TestEmptyAndMissing:
    def test_nonexistent_path_issue(self, tmp_path):
        provider = FilesystemSkillProvider([tmp_path / "does_not_exist"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any(
            "does not exist" in i.message.lower() or "not found" in i.message.lower()
            for i in exc.value.issues
        )

    def test_empty_dir_no_skills(self, tmp_path):
        provider = FilesystemSkillProvider([tmp_path])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any(
            "no skill.md" in i.message.lower() or "no skills" in i.message.lower()
            for i in exc.value.issues
        )

    def test_non_md_file_issue(self, tmp_path):
        txt = tmp_path / "x.txt"
        txt.write_text("not a skill", encoding="utf-8")
        provider = FilesystemSkillProvider([txt])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any(
            ".md" in i.message.lower() or "expected" in i.message.lower()
            for i in exc.value.issues
        )


class TestDirOfSkills:
    def test_dir_without_skill_md_scans_subdirs(self, tmp_path):
        """A directory with no direct SKILL.md but with subdirs each containing
        SKILL.md should yield one Skill per subdir."""
        import shutil

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        shutil.copy(FIXTURES / "minimal" / "SKILL.md", tmp_path / "a" / "SKILL.md")
        shutil.copy(
            FIXTURES / "with_when_to_use" / "SKILL.md",
            tmp_path / "b" / "SKILL.md",
        )
        provider = FilesystemSkillProvider([tmp_path])
        skills = provider.discover()
        names = {s.name for s in skills}
        assert names == {"a", "b"}


class TestAllValidFixtures:
    def test_all_valid_fixtures_load(self, tmp_path):
        """Copy valid fixtures into a tmp dir and discover."""
        import shutil

        valid_names = [
            "minimal",
            "with_when_to_use",
            "with_named_args",
            "with_flat_args_only",
            "with_all_hooks",
            "fork_context",
        ]
        for name in valid_names:
            shutil.copytree(FIXTURES / name, tmp_path / name)

        provider = FilesystemSkillProvider([tmp_path])
        skills = provider.discover()
        names = {s.name for s in skills}
        assert names == set(valid_names)

    def test_fork_context_parsed(self):
        provider = FilesystemSkillProvider([FIXTURES / "fork_context"])
        skills = provider.discover()
        assert skills[0].context == "fork"

    def test_named_args_parsed(self):
        provider = FilesystemSkillProvider([FIXTURES / "with_named_args"])
        skills = provider.discover()
        assert skills[0].arguments == ("path", "depth")

    def test_all_hooks_parsed(self):
        provider = FilesystemSkillProvider([FIXTURES / "with_all_hooks"])
        skills = provider.discover()
        assert len(skills[0].hooks) == 7


class TestNameOverride:
    def test_frontmatter_name_overrides_dirname(self, tmp_path):
        skill_dir = tmp_path / "original"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: override\ndescription: test\n---\nBody.",
            encoding="utf-8",
        )
        provider = FilesystemSkillProvider([skill_dir])
        skills = provider.discover()
        assert skills[0].name == "override"
