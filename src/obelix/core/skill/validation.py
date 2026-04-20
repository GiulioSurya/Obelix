"""Validator chain for skill candidates.

Pure functions. No IO. One Validator class per rule.
run_validators() aggregates all issues, never short-circuits.

Additional validator classes are added in subsequent tasks (3.2-3.6).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
