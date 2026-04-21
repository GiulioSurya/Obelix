from pathlib import Path

import pytest

from obelix.core.skill.parsing import ParseError, parse_skill_file


class TestFrontmatterSplit:
    def test_valid_frontmatter_and_body(self):
        raw = "---\ndescription: hi\n---\n# Body\n\nhello"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter == {"description": "hi"}
        assert c.body == "# Body\n\nhello"

    def test_no_frontmatter_raises(self):
        with pytest.raises(ParseError) as exc:
            parse_skill_file("Just a body", Path("x.md"))
        assert "missing frontmatter" in str(exc.value).lower()

    def test_unclosed_frontmatter_raises(self):
        raw = "---\ndescription: hi\n# Body\n"
        with pytest.raises(ParseError):
            parse_skill_file(raw, Path("x.md"))

    def test_empty_body(self):
        raw = "---\ndescription: hi\n---\n"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.body == ""

    def test_body_only_whitespace(self):
        raw = "---\ndescription: hi\n---\n   \n\n"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.body.strip() == ""


class TestYAMLErrors:
    def test_invalid_yaml_raises_with_line(self):
        raw = "---\ndescription: hi\nkey:\n  bad: : :\n---\nbody"
        with pytest.raises(ParseError) as exc:
            parse_skill_file(raw, Path("x.md"))
        assert exc.value.line is not None

    def test_non_mapping_frontmatter_raises(self):
        raw = "---\n- item1\n- item2\n---\nbody"
        with pytest.raises(ParseError) as exc:
            parse_skill_file(raw, Path("x.md"))
        assert "mapping" in str(exc.value).lower() or "dict" in str(exc.value).lower()


class TestEdgeCases:
    def test_crlf_line_endings(self):
        raw = "---\r\ndescription: hi\r\n---\r\n# Body\r\nhello"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter == {"description": "hi"}
        assert "hello" in c.body

    def test_bom_prefix(self):
        raw = "\ufeff---\ndescription: hi\n---\nbody"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter == {"description": "hi"}

    def test_unicode_in_body(self):
        raw = "---\ndescription: café\n---\nBody with é, ñ, 日本語"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter["description"] == "café"
        assert "日本語" in c.body

    def test_extra_dashes_after_open(self):
        """Only 3 dashes on first line are accepted; 4+ dashes do not open frontmatter."""
        raw = "----\ndescription: nope\n---\nbody"
        with pytest.raises(ParseError):
            parse_skill_file(raw, Path("x.md"))

    def test_frontmatter_with_nested_mapping(self):
        raw = "---\ndescription: hi\nhooks:\n  on_tool_error: retry\n---\nbody"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter["hooks"] == {"on_tool_error": "retry"}

    def test_frontmatter_with_list(self):
        raw = "---\ndescription: hi\narguments:\n  - path\n  - depth\n---\nbody"
        c = parse_skill_file(raw, Path("x.md"))
        assert c.frontmatter["arguments"] == ["path", "depth"]


class TestParseError:
    def test_has_line_attribute(self):
        err = ParseError("msg", line=5)
        assert err.line == 5
        assert str(err) == "msg"

    def test_line_optional(self):
        err = ParseError("msg")
        assert err.line is None
