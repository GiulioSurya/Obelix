"""Filesystem-backed skill provider.

Walks configured paths, parses SKILL.md files, runs validators,
aggregates all issues across all files, raises SkillValidationError if any.
"""

from __future__ import annotations

from pathlib import Path

from obelix.core.skill.parsing import ParseError, parse_skill_file
from obelix.core.skill.skill import (
    Skill,
    SkillCandidate,
    SkillIssue,
    SkillValidationError,
)
from obelix.core.skill.validation import DEFAULT_VALIDATORS, run_validators
from obelix.ports.outbound.skill_provider import AbstractSkillProvider


class FilesystemSkillProvider(AbstractSkillProvider):
    """Discovers skills from explicit filesystem paths.

    Accepted path shapes:
      - path to a `.md` file -> treated as a skill (name from parent dir).
      - path to a directory containing `SKILL.md` -> one skill (name from dir).
      - path to a directory without `SKILL.md` -> scans immediate subdirs for `SKILL.md`.
    """

    def __init__(self, paths: list[Path | str]):
        self._paths = [Path(p) for p in paths]

    def discover(self) -> list[Skill]:
        all_issues: list[SkillIssue] = []
        validated: list[Skill] = []

        candidates = self._resolve_candidates(all_issues)
        for cand_path in candidates:
            skill = self._load_one(cand_path, all_issues)
            if skill is not None:
                validated.append(skill)

        if all_issues:
            raise SkillValidationError(all_issues)
        return validated

    def _resolve_candidates(self, bag: list[SkillIssue]) -> list[Path]:
        """Resolve each configured path into a list of SKILL.md file paths."""
        out: list[Path] = []
        for p in self._paths:
            if not p.exists():
                bag.append(
                    SkillIssue(
                        file_path=p,
                        field="path",
                        message=f"path does not exist: {p}",
                    )
                )
                continue

            if p.is_file():
                if p.suffix != ".md":
                    bag.append(
                        SkillIssue(
                            file_path=p,
                            field="path",
                            message=f"expected a .md file, got {p.suffix or '(no extension)'}",
                        )
                    )
                    continue
                out.append(p)
                continue

            # directory
            direct = p / "SKILL.md"
            if direct.exists():
                out.append(direct)
                continue

            # scan immediate subdirs for SKILL.md
            nested = sorted(p.glob("*/SKILL.md"))
            if not nested:
                bag.append(
                    SkillIssue(
                        file_path=p,
                        field="path",
                        message=f"no SKILL.md found in {p} or its subdirs",
                    )
                )
                continue
            out.extend(nested)

        return out

    def _load_one(self, skill_md: Path, bag: list[SkillIssue]) -> Skill | None:
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except OSError as e:
            bag.append(
                SkillIssue(
                    file_path=skill_md,
                    field="io",
                    message=f"read failed: {e}",
                )
            )
            return None

        try:
            candidate: SkillCandidate = parse_skill_file(raw, skill_md)
        except ParseError as e:
            bag.append(
                SkillIssue(
                    file_path=skill_md,
                    field="parse",
                    message=str(e),
                    line=e.line,
                )
            )
            return None

        issues = run_validators(candidate, DEFAULT_VALIDATORS)
        if issues:
            bag.extend(issues)
            return None

        base_dir = skill_md.parent.resolve()
        name = base_dir.name
        # Frontmatter `name:` overrides directory name
        fm_name = candidate.frontmatter.get("name")
        if isinstance(fm_name, str) and fm_name:
            name = fm_name

        return Skill.from_candidate(
            candidate, name=name, base_dir=base_dir, source="filesystem"
        )
