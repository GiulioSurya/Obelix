"""Data shapes for the skills subsystem.

Pure data. No IO, no logic. Leaf of the dependency DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SkillIssue:
    """A single validation issue found in a candidate skill."""

    file_path: Path | None
    field: str
    message: str
    line: int | None = None


@dataclass
class SkillCandidate:
    """Raw parsed skill, pre-validation.

    Produced by parsing.parse_skill_file(), consumed by validation.run_validators().
    Mutable only so parsers can populate it field-by-field. Not hashable.
    """

    file_path: Path | None
    frontmatter: dict
    body: str


@dataclass(frozen=True)
class Skill:
    """A validated skill, ready to be invoked."""

    name: str
    description: str
    body: str
    base_dir: Path | None
    when_to_use: str | None = None
    allowed_tools: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = field(default_factory=dict)
    source: Literal["filesystem", "mcp"] = "filesystem"
    file_path: Path | None = None

    @classmethod
    def from_candidate(
        cls,
        candidate: SkillCandidate,
        name: str,
        base_dir: Path | None,
        source: Literal["filesystem", "mcp"] = "filesystem",
    ) -> Skill:
        fm = candidate.frontmatter
        return cls(
            name=name,
            description=fm["description"],
            body=candidate.body,
            base_dir=base_dir,
            when_to_use=fm.get("when_to_use"),
            allowed_tools=tuple(fm.get("allowed_tools", [])),
            arguments=tuple(fm.get("arguments", [])),
            context=fm.get("context", "inline"),
            hooks=dict(fm.get("hooks", {})),
            source=source,
            file_path=candidate.file_path,
        )


class SkillValidationError(Exception):
    """Raised at agent boot when one or more skills fail validation.

    Carries all aggregated issues so the user sees every problem at once.
    """

    def __init__(self, issues: list[SkillIssue]):
        if not issues:
            raise ValueError("SkillValidationError requires at least one issue")
        self.issues = issues
        super().__init__(self._default_message())

    def _default_message(self) -> str:
        # reporting.py will be added in a later task; fall back gracefully
        # until it exists, so this module can be used in isolation.
        try:
            from obelix.core.skill.reporting import format_validation_error

            return format_validation_error(self.issues)
        except ImportError:
            return f"{len(self.issues)} skill issue(s)"
