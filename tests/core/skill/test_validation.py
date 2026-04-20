from pathlib import Path

from obelix.core.skill.skill import SkillCandidate, SkillIssue
from obelix.core.skill.validation import (
    DEFAULT_VALIDATORS,
    ArgumentUniquenessValidator,
    BodyNonEmptyValidator,
    FrontmatterSchemaValidator,
    HookEventValidator,
    PlaceholderConsistencyValidator,
    Validator,
    run_validators,
)


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


def _cand_fm(**fm) -> SkillCandidate:
    return SkillCandidate(file_path=Path("x.md"), frontmatter=fm, body="nonempty body")


class TestFrontmatterSchemaValidator:
    def setup_method(self):
        self.v = FrontmatterSchemaValidator()

    def test_happy_path(self):
        assert self.v.check(_cand_fm(description="hi")) == []

    def test_missing_description(self):
        issues = self.v.check(_cand_fm())
        assert len(issues) == 1
        assert "description" in issues[0].field
        assert (
            "missing" in issues[0].message.lower()
            or "required" in issues[0].message.lower()
            or "field required" in issues[0].message.lower()
        )

    def test_description_empty_string(self):
        issues = self.v.check(_cand_fm(description=""))
        assert len(issues) == 1

    def test_description_wrong_type(self):
        issues = self.v.check(_cand_fm(description=42))
        assert len(issues) == 1

    def test_unknown_field_ignored_forward_compat(self):
        """Unknown frontmatter keys are silently ignored (forward compat)."""
        assert self.v.check(_cand_fm(description="hi", future_field="xyz")) == []

    def test_full_valid_frontmatter(self):
        fm = {
            "description": "Reviews Python",
            "when_to_use": "on demand",
            "arguments": ["path", "depth"],
            "allowed_tools": ["Read", "Grep"],
            "context": "inline",
            "hooks": {"on_tool_error": "retry"},
            "name": "custom-name",
        }
        assert self.v.check(_cand_fm(**fm)) == []

    def test_invalid_context_value(self):
        issues = self.v.check(_cand_fm(description="hi", context="parallel"))
        assert len(issues) >= 1
        assert any("context" in i.field for i in issues)

    def test_arguments_wrong_type(self):
        issues = self.v.check(_cand_fm(description="hi", arguments="not a list"))
        assert len(issues) >= 1

    def test_hooks_wrong_type(self):
        issues = self.v.check(_cand_fm(description="hi", hooks=["not a dict"]))
        assert len(issues) >= 1

    def test_allowed_tools_wrong_type(self):
        issues = self.v.check(_cand_fm(description="hi", allowed_tools="Read"))
        assert len(issues) >= 1

    def test_issue_file_path_propagated(self):
        issues = self.v.check(_cand_fm())
        assert issues[0].file_path == Path("x.md")

    def test_no_issue_raised_just_returned(self):
        """Validator returns list — never raises on invalid input."""
        # Should not raise for any of these:
        self.v.check(_cand_fm(description=42, arguments="x", hooks="y"))


class TestHookEventValidator:
    def setup_method(self):
        self.v = HookEventValidator()

    def test_no_hooks_no_issues(self):
        assert self.v.check(_cand_fm()) == []

    def test_empty_hooks_dict_no_issues(self):
        assert self.v.check(_cand_fm(hooks={})) == []

    def test_all_valid_events_no_issues(self):
        hooks = {
            "before_llm_call": "x",
            "after_llm_call": "x",
            "before_tool_execution": "x",
            "after_tool_execution": "x",
            "on_tool_error": "x",
            "before_final_response": "x",
            "query_end": "x",
        }
        assert self.v.check(_cand_fm(hooks=hooks)) == []

    def test_unknown_event_one_issue(self):
        issues = self.v.check(_cand_fm(hooks={"on_random": "x"}))
        assert len(issues) == 1
        assert "hooks.on_random" in issues[0].field
        assert "query_end" in issues[0].message

    def test_multiple_unknown_events_all_reported(self):
        issues = self.v.check(_cand_fm(hooks={"on_a": "x", "on_b": "y", "on_c": "z"}))
        assert len(issues) == 3
        fields = {i.field for i in issues}
        assert fields == {"hooks.on_a", "hooks.on_b", "hooks.on_c"}

    def test_mix_of_valid_and_invalid(self):
        issues = self.v.check(_cand_fm(hooks={"on_tool_error": "x", "on_bogus": "y"}))
        assert len(issues) == 1
        assert "hooks.on_bogus" in issues[0].field

    def test_hooks_not_dict_no_crash(self):
        """If hooks is not a dict (FrontmatterSchemaValidator catches it),
        this validator gracefully returns no issues."""
        assert self.v.check(_cand_fm(hooks="not a dict")) == []


class TestArgumentUniquenessValidator:
    def setup_method(self):
        self.v = ArgumentUniquenessValidator()

    def test_no_arguments_ok(self):
        assert self.v.check(_cand_fm()) == []

    def test_empty_list_ok(self):
        assert self.v.check(_cand_fm(arguments=[])) == []

    def test_unique_names_ok(self):
        assert self.v.check(_cand_fm(arguments=["path", "depth"])) == []

    def test_single_duplicate_one_issue(self):
        issues = self.v.check(_cand_fm(arguments=["path", "path"]))
        assert len(issues) == 1
        assert "path" in issues[0].message
        assert "duplicate" in issues[0].message.lower()

    def test_multiple_duplicates_all_reported(self):
        issues = self.v.check(_cand_fm(arguments=["a", "b", "a", "c", "b"]))
        messages = [i.message for i in issues]
        # At least one mentioning 'a' and one mentioning 'b'
        assert any("'a'" in m for m in messages)
        assert any("'b'" in m for m in messages)

    def test_reserved_arguments_name(self):
        issues = self.v.check(_cand_fm(arguments=["path", "ARGUMENTS"]))
        assert any("reserved" in i.message.lower() for i in issues)

    def test_reserved_and_duplicate_both_reported(self):
        issues = self.v.check(_cand_fm(arguments=["ARGUMENTS", "ARGUMENTS"]))
        assert len(issues) >= 2

    def test_non_list_arguments_no_crash(self):
        """If arguments is not a list (schema validator catches it),
        this returns empty."""
        assert self.v.check(_cand_fm(arguments="not a list")) == []

    def test_non_string_item_skipped(self):
        """Non-string items skipped defensively (schema validator reports them)."""
        # Should not raise
        self.v.check(_cand_fm(arguments=["path", 42, "depth"]))

    def test_issue_field_is_arguments(self):
        issues = self.v.check(_cand_fm(arguments=["x", "x"]))
        assert issues[0].field == "arguments"


def _cand_body(body: str, **fm) -> SkillCandidate:
    return SkillCandidate(file_path=Path("x.md"), frontmatter=fm, body=body)


class TestPlaceholderConsistencyValidator:
    def setup_method(self):
        self.v = PlaceholderConsistencyValidator()

    def test_body_no_placeholders_ok(self):
        assert self.v.check(_cand_body("Just text.")) == []

    def test_arguments_placeholder_always_ok(self):
        assert self.v.check(_cand_body("use $ARGUMENTS here")) == []

    def test_meta_placeholders_always_ok(self):
        body = "dir=${OBELIX_SKILL_DIR} sid=${OBELIX_SESSION_ID}"
        assert self.v.check(_cand_body(body)) == []

    def test_declared_placeholder_ok(self):
        assert self.v.check(_cand_body("path=$path", arguments=["path"])) == []

    def test_undeclared_placeholder_one_issue(self):
        issues = self.v.check(_cand_body("depth=$depth", arguments=["path"]))
        assert len(issues) == 1
        assert "$depth" in issues[0].message

    def test_multiple_undeclared_all_reported(self):
        issues = self.v.check(_cand_body("$a $b $c", arguments=["a"]))
        msgs = [i.message for i in issues]
        assert any("$b" in m for m in msgs)
        assert any("$c" in m for m in msgs)

    def test_same_placeholder_multiple_times_reported_once(self):
        issues = self.v.check(_cand_body("$x then $x", arguments=[]))
        assert len(issues) == 1

    def test_placeholder_name_with_underscore(self):
        assert self.v.check(_cand_body("$my_var", arguments=["my_var"])) == []

    def test_placeholder_followed_by_underscore_word(self):
        """$pa must be recognized distinctly from $path."""
        issues = self.v.check(_cand_body("x=$pa", arguments=["path"]))
        assert len(issues) == 1
        assert "$pa" in issues[0].message

    def test_non_list_arguments_defensive(self):
        """If arguments isn't a list, treat as empty — schema validator reports type."""
        issues = self.v.check(_cand_body("$x", arguments="not a list"))
        # $x is unrecognized
        assert len(issues) == 1

    def test_issue_field_is_body(self):
        issues = self.v.check(_cand_body("$unknown", arguments=[]))
        assert issues[0].field == "body"


class TestBodyNonEmptyValidator:
    def setup_method(self):
        self.v = BodyNonEmptyValidator()

    def test_normal_body_ok(self):
        assert self.v.check(_cand_body("hello")) == []

    def test_empty_body_one_issue(self):
        issues = self.v.check(_cand_body(""))
        assert len(issues) == 1
        assert issues[0].field == "body"
        assert "empty" in issues[0].message.lower()

    def test_whitespace_only_body_one_issue(self):
        issues = self.v.check(_cand_body("   \n\n  "))
        assert len(issues) == 1

    def test_issue_file_path_propagated(self):
        issues = self.v.check(_cand_body(""))
        assert issues[0].file_path == Path("x.md")


class TestDefaultValidators:
    def test_default_validators_is_tuple(self):
        assert isinstance(DEFAULT_VALIDATORS, tuple)
        # Has all 5 concrete validators
        assert len(DEFAULT_VALIDATORS) >= 5

    def test_default_catches_multi_issue_candidate(self):
        candidate = SkillCandidate(
            file_path=Path("x.md"),
            frontmatter={
                "description": "",  # fails schema (empty)
                "arguments": ["a", "a"],  # duplicate
                "hooks": {"bogus": "x"},  # unknown event
            },
            body="$undeclared_thing",
        )
        issues = run_validators(candidate, DEFAULT_VALIDATORS)
        # Expect at least: empty description + duplicate arg + unknown hook + undeclared placeholder
        assert len(issues) >= 4

    def test_default_happy_path(self):
        candidate = SkillCandidate(
            file_path=Path("x.md"),
            frontmatter={
                "description": "All good",
                "arguments": ["path"],
                "hooks": {"on_tool_error": "retry"},
            },
            body="Review $path",
        )
        assert run_validators(candidate, DEFAULT_VALIDATORS) == []

    def test_default_validators_contains_all_5(self):
        """Each of the 5 concrete validator types is represented."""
        from obelix.core.skill.validation import (
            ArgumentUniquenessValidator,
            BodyNonEmptyValidator,
            FrontmatterSchemaValidator,
            HookEventValidator,
            PlaceholderConsistencyValidator,
        )

        types = {type(v) for v in DEFAULT_VALIDATORS}
        expected = {
            FrontmatterSchemaValidator,
            HookEventValidator,
            ArgumentUniquenessValidator,
            PlaceholderConsistencyValidator,
            BodyNonEmptyValidator,
        }
        assert expected <= types
