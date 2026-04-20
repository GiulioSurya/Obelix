"""Validator chain for skill candidates.

Pure functions. No IO. One Validator class per rule.
run_validators() aggregates all issues, never short-circuits.

Additional validator classes are added in subsequent tasks (3.2-3.6).
"""

from __future__ import annotations

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
    """Every hooks.<key> must map to a valid AgentEvent enum value."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        hooks = candidate.frontmatter.get("hooks", {})
        if not isinstance(hooks, dict):
            return []  # type error is reported by FrontmatterSchemaValidator

        valid_list = ", ".join(sorted(_VALID_HOOK_EVENTS))
        issues: list[SkillIssue] = []
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
