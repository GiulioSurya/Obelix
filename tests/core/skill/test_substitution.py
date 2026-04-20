from pathlib import Path

import pytest

from obelix.core.skill.substitution import (
    SkillInvocationError,
    substitute_placeholders,
)


class TestFlatArguments:
    def test_arguments_replaced(self):
        out = substitute_placeholders(
            body="Run on $ARGUMENTS",
            args="src/foo.py",
            declared_args=(),
            base_dir=None,
            session_id="s",
        )
        assert out == "Run on src/foo.py"

    def test_no_args_empty_arguments(self):
        out = substitute_placeholders(
            body="Args: [$ARGUMENTS]",
            args="",
            declared_args=(),
            base_dir=None,
            session_id="s",
        )
        assert out == "Args: []"

    def test_body_without_placeholders_unchanged(self):
        body = "Just some text."
        out = substitute_placeholders(
            body=body, args="ignored", declared_args=(), base_dir=None, session_id="s"
        )
        assert out == body

    def test_empty_body_with_args_no_op(self):
        out = substitute_placeholders(
            body="",
            args="anything",
            declared_args=(),
            base_dir=None,
            session_id="s",
        )
        assert out == ""


class TestNamedPositional:
    def test_single_named(self):
        out = substitute_placeholders(
            body="path=$path",
            args="src/foo.py",
            declared_args=("path",),
            base_dir=None,
            session_id="s",
        )
        assert out == "path=src/foo.py"

    def test_multiple_named(self):
        out = substitute_placeholders(
            body="$path @ $depth",
            args="foo.py 3",
            declared_args=("path", "depth"),
            base_dir=None,
            session_id="s",
        )
        assert out == "foo.py @ 3"

    def test_missing_arg_becomes_empty(self):
        out = substitute_placeholders(
            body="[$path][$depth]",
            args="foo.py",
            declared_args=("path", "depth"),
            base_dir=None,
            session_id="s",
        )
        assert out == "[foo.py][]"

    def test_quoted_arg_with_spaces(self):
        out = substitute_placeholders(
            body="file=$path count=$depth",
            args='"my file.py" 3',
            declared_args=("path", "depth"),
            base_dir=None,
            session_id="s",
        )
        assert out == "file=my file.py count=3"

    def test_too_many_args_raises(self):
        with pytest.raises(SkillInvocationError) as exc:
            substitute_placeholders(
                body="$x",
                args="a b c d",
                declared_args=("x",),
                base_dir=None,
                session_id="s",
            )
        assert "expects 1" in str(exc.value)
        assert "got 4" in str(exc.value)

    def test_arguments_flat_still_available(self):
        out = substitute_placeholders(
            body="path=$path raw=$ARGUMENTS",
            args="foo.py",
            declared_args=("path",),
            base_dir=None,
            session_id="s",
        )
        assert out == "path=foo.py raw=foo.py"

    def test_prefix_safety(self):
        """$pa must not be partially replaced when $path is declared."""
        out = substitute_placeholders(
            body="$path-$pa",
            args="foo",
            declared_args=("path", "pa"),
            base_dir=None,
            session_id="s",
        )
        assert out == "foo-"

    def test_unterminated_quote_raises_invocation_error(self):
        with pytest.raises(SkillInvocationError) as exc:
            substitute_placeholders(
                body="$p",
                args='"oops',
                declared_args=("p",),
                base_dir=None,
                session_id="s",
            )
        assert "failed to parse" in str(exc.value).lower()


class TestMetaPlaceholders:
    def test_skill_dir(self):
        out = substitute_placeholders(
            body="dir=${OBELIX_SKILL_DIR}",
            args="",
            declared_args=(),
            base_dir=Path("/abs/skills/x"),
            session_id="s",
        )
        # Path stringification is platform-dependent; tolerate both separators.
        assert "abs" in out and "skills" in out and "x" in out
        assert "${OBELIX_SKILL_DIR}" not in out

    def test_skill_dir_none_replaced_with_empty(self):
        out = substitute_placeholders(
            body="dir=[${OBELIX_SKILL_DIR}]",
            args="",
            declared_args=(),
            base_dir=None,
            session_id="s",
        )
        assert out == "dir=[]"

    def test_session_id(self):
        out = substitute_placeholders(
            body="sid=${OBELIX_SESSION_ID}",
            args="",
            declared_args=(),
            base_dir=None,
            session_id="abc-123",
        )
        assert out == "sid=abc-123"


class TestUnicode:
    def test_unicode_args_preserved(self):
        out = substitute_placeholders(
            body="name=$name",
            args="café",
            declared_args=("name",),
            base_dir=None,
            session_id="s",
        )
        assert out == "name=café"


class TestPropertyIdempotence:
    def test_meta_already_substituted_no_op(self):
        """Running substitution twice on meta placeholders only produces the same output."""
        body = "dir=/abs sid=abc"
        out1 = substitute_placeholders(
            body=body,
            args="",
            declared_args=(),
            base_dir=None,
            session_id="abc",
        )
        out2 = substitute_placeholders(
            body=out1,
            args="",
            declared_args=(),
            base_dir=None,
            session_id="abc",
        )
        assert out1 == out2
