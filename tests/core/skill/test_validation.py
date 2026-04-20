from pathlib import Path

from obelix.core.skill.skill import SkillCandidate, SkillIssue
from obelix.core.skill.validation import Validator, run_validators


class _Fixed:
    """Test-only validator returning a fixed list of issues."""

    def __init__(self, issues: list[SkillIssue]):
        self._issues = issues

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        return list(self._issues)


def _cand() -> SkillCandidate:
    return SkillCandidate(file_path=Path("x.md"), frontmatter={}, body="")


class TestRunValidators:
    def test_empty_validator_tuple_no_issues(self):
        assert run_validators(_cand(), validators=()) == []

    def test_single_validator_issues_propagated(self):
        v = _Fixed([SkillIssue(file_path=None, field="a", message="b")])
        issues = run_validators(_cand(), validators=(v,))
        assert len(issues) == 1
        assert issues[0].field == "a"

    def test_multiple_validators_all_issues_aggregated(self):
        v1 = _Fixed([SkillIssue(file_path=None, field="a", message="b")])
        v2 = _Fixed(
            [
                SkillIssue(file_path=None, field="c", message="d"),
                SkillIssue(file_path=None, field="e", message="f"),
            ]
        )
        issues = run_validators(_cand(), validators=(v1, v2))
        assert len(issues) == 3
        assert [i.field for i in issues] == ["a", "c", "e"]

    def test_validator_returning_empty_is_fine(self):
        issues = run_validators(_cand(), validators=(_Fixed([]),))
        assert issues == []

    def test_never_short_circuits_on_many_issues(self):
        v1 = _Fixed([SkillIssue(file_path=None, field="x", message="1")])
        v2 = _Fixed([SkillIssue(file_path=None, field="y", message="2")])
        v3 = _Fixed([SkillIssue(file_path=None, field="z", message="3")])
        issues = run_validators(_cand(), validators=(v1, v2, v3))
        assert len(issues) == 3


class TestValidatorProtocol:
    def test_any_class_with_check_method_satisfies(self):
        """Protocol is structural — duck typing is enough."""

        class MyValidator:
            def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
                return []

        v: Validator = MyValidator()
        assert run_validators(_cand(), validators=(v,)) == []

    def test_runtime_checkable_isinstance(self):
        """Protocol is runtime_checkable so isinstance() works."""

        class MyValidator:
            def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
                return []

        assert isinstance(MyValidator(), Validator)

    def test_missing_check_fails_isinstance(self):
        class NotAValidator:
            def something_else(self): ...

        assert not isinstance(NotAValidator(), Validator)
