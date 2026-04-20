"""Validator chain for skill candidates.

Pure functions. No IO. One Validator class per rule.
run_validators() aggregates all issues, never short-circuits.

Additional validator classes are added in subsequent tasks (3.2-3.6).
"""

from __future__ import annotations

import re
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from obelix.core.agent.hooks import AgentEvent
from obelix.core.skill.skill import SkillCandidate, SkillIssue


@runtime_checkable
class Validator(Protocol):
    """One validator = one rule. Stateless. Pure."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...


def run_validators(
    candidate: SkillCandidate,
    validators: tuple[Validator, ...],
) -> list[SkillIssue]:
    """Run every validator against the candidate and aggregate issues.

    Never short-circuits. Validators must return issue lists; raising
    is considered a bug and will propagate.
    """
    issues: list[SkillIssue] = []
    for v in validators:
        issues.extend(v.check(candidate))
    return issues


class _FrontmatterSchema(BaseModel):
    """Pydantic model capturing the schema of valid frontmatter."""

    model_config = ConfigDict(extra="ignore")  # forward-compat: ignore unknown keys

    name: str | None = None
    description: str = Field(..., min_length=1)
    when_to_use: str | None = None
    arguments: list[str] = []
    allowed_tools: list[str] = []
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = {}


class FrontmatterSchemaValidator:
    """Validate required fields, types, and Literal values using Pydantic."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        try:
            _FrontmatterSchema.model_validate(candidate.frontmatter)
            return []
        except ValidationError as e:
            issues: list[SkillIssue] = []
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "frontmatter"
                issues.append(
                    SkillIssue(
                        file_path=candidate.file_path,
                        field=f"frontmatter.{loc}",
                        message=err["msg"],
                    )
                )
            return issues


_VALID_HOOK_EVENTS: frozenset[str] = frozenset(e.value for e in AgentEvent)


class HookEventValidator:
    """Every hooks.<key> must map to a valid AgentEvent enum value.

    Catches typos like `on_tool_erorr` that would otherwise silently never fire.
    """

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        hooks = candidate.frontmatter.get("hooks", {})
        if not isinstance(hooks, dict):
            return []  # type error is reported by FrontmatterSchemaValidator

        valid_list = ", ".join(sorted(_VALID_HOOK_EVENTS))
        issues: list[SkillIssue] = []
        # Issue order mirrors YAML source order (PyYAML preserves insertion order).
        for key in hooks.keys():
            if key not in _VALID_HOOK_EVENTS:
                issues.append(
                    SkillIssue(
                        file_path=candidate.file_path,
                        field=f"hooks.{key}",
                        message=f"unknown event. Valid: {valid_list}",
                    )
                )
        return issues


_RESERVED_ARG_NAMES: frozenset[str] = frozenset({"ARGUMENTS"})


class ArgumentUniquenessValidator:
    """Arguments must be unique and not clash with reserved names."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        args = candidate.frontmatter.get("arguments", [])
        if not isinstance(args, list):
            return []

        issues: list[SkillIssue] = []

        # Duplicates
        seen: set[str] = set()
        reported_dups: set[str] = set()
        for name in args:
            if not isinstance(name, str):
                continue
            if name in seen and name not in reported_dups:
                issues.append(
                    SkillIssue(
                        file_path=candidate.file_path,
                        field="arguments",
                        message=f"duplicate name '{name}'",
                    )
                )
                reported_dups.add(name)
            seen.add(name)

        # Reserved — one issue per distinct reserved name (not per occurrence)
        reported_reserved: set[str] = set()
        for name in args:
            if (
                isinstance(name, str)
                and name in _RESERVED_ARG_NAMES
                and name not in reported_reserved
            ):
                reported_reserved.add(name)
                issues.append(
                    SkillIssue(
                        file_path=candidate.file_path,
                        field="arguments",
                        message=f"'{name}' is reserved, use a different name",
                    )
                )

        return issues


# Placeholder: $<identifier> where identifier is [A-Za-z_][A-Za-z0-9_]*
# Excludes ${...} form (those are meta placeholders, handled separately).
# A preceding backslash escapes the dollar sign: `\$foo` is a literal, not a placeholder.
_PLACEHOLDER_RE = re.compile(r"(?<!\\)\$(?!\{)([A-Za-z_][A-Za-z0-9_]*)")

# Placeholder names that are always valid regardless of `arguments:`
_META_PLACEHOLDER_NAMES: frozenset[str] = frozenset({"ARGUMENTS"})


class PlaceholderConsistencyValidator:
    """Every $name in body must be declared in `arguments` (or be ARGUMENTS)."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        declared_raw = candidate.frontmatter.get("arguments", [])
        declared: set[str] = (
            set(declared_raw) if isinstance(declared_raw, list) else set()
        )
        allowed = declared | _META_PLACEHOLDER_NAMES

        found = _PLACEHOLDER_RE.findall(candidate.body)
        reported: set[str] = set()
        issues: list[SkillIssue] = []
        for name in found:
            if name in allowed or name in reported:
                continue
            reported.add(name)
            issues.append(
                SkillIssue(
                    file_path=candidate.file_path,
                    field="body",
                    message=f"placeholder '${name}' referenced but not declared in 'arguments'",
                )
            )
        return issues


class BodyNonEmptyValidator:
    """Body (after stripping whitespace) must be non-empty."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        if candidate.body.strip():
            return []
        return [
            SkillIssue(
                file_path=candidate.file_path,
                field="body",
                message="empty body",
            )
        ]


DEFAULT_VALIDATORS: tuple[Validator, ...] = (
    FrontmatterSchemaValidator(),
    HookEventValidator(),
    ArgumentUniquenessValidator(),
    PlaceholderConsistencyValidator(),
    BodyNonEmptyValidator(),
)
