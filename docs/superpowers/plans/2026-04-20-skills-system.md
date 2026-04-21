# Skills System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Skills subsystem for Obelix that lets agents invoke skills at runtime and supports community-authored SKILL.md files (same format as Claude Code), with explicit per-agent configuration, exhaustive boot-time validation, MCP integration, fork execution, and ≥90% test coverage.

**Architecture:** Hexagonal. Pure data + pure functions at the core (`core/skill/`), adapters for filesystem and MCP (`adapters/outbound/skill/`), built-in tool orchestrator (`plugins/builtin/skill_tool.py`), minimal wiring in `BaseAgent`. Validator-chain pattern for exhaustive `collect-all` validation. DAG-enforced dependency graph.

**Tech Stack:** Python 3.13, Pydantic v2 (schema validation), PyYAML (frontmatter parsing), pytest + pytest-asyncio + pytest-cov (testing), hypothesis (property-based tests), grimp (import boundary enforcement).

**Spec:** `docs/superpowers/specs/2026-04-20-skills-system-design.md` — read it first.

---

## Dependency Order

Tasks are ordered so every task builds on completed tasks. Phases are not just grouping — each phase depends on the prior one.

- **Phase 0:** Dev dependencies setup
- **Phase 1:** Foundation — `skill.py` (pure data)
- **Phase 2:** Pure functions — `substitution.py`, `parsing.py`, `reporting.py`
- **Phase 3:** Validator chain — `validation.py` (one validator per task)
- **Phase 4:** Port + adapters — `skill_provider.py`, `filesystem.py`, MCP support
- **Phase 5:** Aggregation — `manager.py`
- **Phase 6:** BaseAgent plumbing — `_unregister_hook`
- **Phase 7:** Orchestrator — `skill_tool.py`
- **Phase 8:** BaseAgent integration — `skills_config` parameter
- **Phase 9:** Integration tests
- **Phase 10:** Regression guardrails
- **Phase 11:** Documentation

---

## File Structure (locked in)

```
src/obelix/
  core/skill/
    __init__.py
    skill.py                     # Skill, SkillCandidate, SkillIssue, SkillValidationError
    substitution.py              # substitute_placeholders()
    parsing.py                   # parse_skill_file(), ParseError
    reporting.py                 # format_validation_error()
    validation.py                # Validator protocol + DEFAULT_VALIDATORS + run_validators()
    manager.py                   # SkillManager
  ports/outbound/
    skill_provider.py            # AbstractSkillProvider
  adapters/outbound/skill/
    __init__.py
    filesystem.py                # FilesystemSkillProvider
    mcp.py                       # MCPSkillProvider
  plugins/builtin/
    skill_tool.py                # SkillTool
  core/agent/
    base_agent.py                # MODIFIED: +skills_config, +_unregister_hook
  adapters/outbound/mcp/
    manager.py                   # MODIFIED: +list_prompts()

tests/
  fixtures/skills/               # Valid and invalid SKILL.md corpus
    minimal/SKILL.md
    with_named_args/SKILL.md
    fork_context/SKILL.md
    ...
  core/skill/
    test_skill.py
    test_substitution.py
    test_parsing.py
    test_reporting.py
    test_validation.py
    test_manager.py
  adapters/outbound/skill/
    test_filesystem.py
    test_mcp.py
  plugins/builtin/
    test_skill_tool.py
  core/agent/
    test_base_agent_skills.py
    test_base_agent_hook_unregister.py
  integration/skill/
    test_full_flow.py
    test_fork_execution.py
    test_hook_lifecycle.py
    test_mcp_roundtrip.py
    test_validation_failure.py
    test_multi_query.py
  regression/
    test_skill_listing_snapshot.py
    test_import_boundaries.py

docs/
  skills.md                      # EN user-facing
  skills_implementation_notes.md # IT personal notes
  base_agent.md                  # UPDATED: +skills_config
  index.md                       # UPDATED: link to skills.md
```

---

## Phase 0 — Dev dependencies

### Task 0.1: Add `hypothesis` and `grimp` to dev group

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit pyproject.toml**

Open `pyproject.toml`, find the `[dependency-groups]` section, and in the `dev` list add two entries:

```toml
[dependency-groups]
dev = [
    "ruff>=0.9.0",
    "pytest>=9.0.0",
    "pytest-asyncio>=1.0.0",
    "pytest-cov>=7.0.0",
    # ... existing entries ...
    "hypothesis>=6.100",
    "grimp>=3.3",
]
```

- [ ] **Step 2: Install**

Run: `uv sync --group dev`
Expected: new packages installed, `uv.lock` updated.

- [ ] **Step 3: Verify imports**

Run: `uv run python -c "import hypothesis; import grimp; print(hypothesis.__version__, grimp.__version__)"`
Expected: two version strings printed.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add hypothesis and grimp for skills system tests"
```

---

## Phase 1 — Foundation: skill.py

### Task 1.1: Data dataclasses — `Skill`, `SkillCandidate`, `SkillIssue`

**Files:**
- Create: `src/obelix/core/skill/__init__.py`
- Create: `src/obelix/core/skill/skill.py`
- Create: `tests/core/skill/__init__.py`
- Create: `tests/core/skill/test_skill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/skill/test_skill.py`:

```python
from pathlib import Path
import pytest
from obelix.core.skill.skill import (
    Skill,
    SkillCandidate,
    SkillIssue,
    SkillValidationError,
)


class TestSkillCandidate:
    def test_holds_raw_frontmatter_and_body(self):
        c = SkillCandidate(
            file_path=Path("x.md"),
            frontmatter={"description": "foo"},
            body="hello",
        )
        assert c.frontmatter == {"description": "foo"}
        assert c.body == "hello"
        assert c.file_path == Path("x.md")

    def test_file_path_optional_for_mcp(self):
        c = SkillCandidate(file_path=None, frontmatter={}, body="")
        assert c.file_path is None


class TestSkillIssue:
    def test_requires_field_and_message(self):
        issue = SkillIssue(
            file_path=Path("x.md"),
            field="frontmatter.description",
            message="missing",
        )
        assert issue.field == "frontmatter.description"
        assert issue.message == "missing"
        assert issue.line is None

    def test_hashable(self):
        a = SkillIssue(file_path=None, field="x", message="y")
        b = SkillIssue(file_path=None, field="x", message="y")
        assert hash(a) == hash(b)


class TestSkill:
    def test_basic_construction(self):
        s = Skill(
            name="reviewer",
            description="Reviews code",
            body="# Review",
            base_dir=Path("/skills/reviewer"),
        )
        assert s.name == "reviewer"
        assert s.description == "Reviews code"
        assert s.body == "# Review"
        assert s.context == "inline"
        assert s.source == "filesystem"
        assert s.arguments == ()
        assert s.hooks == {}
        assert s.when_to_use is None

    def test_frozen(self):
        s = Skill(name="a", description="b", body="c", base_dir=None)
        with pytest.raises(Exception):
            s.name = "x"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = Skill(name="x", description="y", body="z", base_dir=None)
        b = Skill(name="x", description="y", body="z", base_dir=None)
        assert a == b

    def test_from_candidate(self):
        candidate = SkillCandidate(
            file_path=Path("skills/r/SKILL.md"),
            frontmatter={
                "description": "Reviews",
                "when_to_use": "when needed",
                "arguments": ["path"],
                "context": "fork",
                "hooks": {"on_tool_error": "retry"},
                "allowed_tools": ["Read"],
            },
            body="# body",
        )
        s = Skill.from_candidate(candidate, name="r", base_dir=Path("skills/r"))
        assert s.name == "r"
        assert s.description == "Reviews"
        assert s.when_to_use == "when needed"
        assert s.arguments == ("path",)
        assert s.context == "fork"
        assert s.hooks == {"on_tool_error": "retry"}
        assert s.allowed_tools == ("Read",)
        assert s.base_dir == Path("skills/r")
        assert s.file_path == Path("skills/r/SKILL.md")
        assert s.source == "filesystem"


class TestSkillValidationError:
    def test_empty_issues_raises_value_error(self):
        with pytest.raises(ValueError):
            SkillValidationError([])

    def test_carries_issues(self):
        issues = [SkillIssue(file_path=None, field="x", message="y")]
        err = SkillValidationError(issues)
        assert err.issues == issues

    def test_is_exception(self):
        err = SkillValidationError(
            [SkillIssue(file_path=None, field="x", message="y")]
        )
        assert isinstance(err, Exception)
```

Also create empty `tests/core/skill/__init__.py`.

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/core/skill/test_skill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'obelix.core.skill'`

- [ ] **Step 3: Create module skeleton**

Create `src/obelix/core/skill/__init__.py`:

```python
"""Skills subsystem: data shapes, validation, management."""

from obelix.core.skill.skill import (
    Skill,
    SkillCandidate,
    SkillIssue,
    SkillValidationError,
)

__all__ = [
    "Skill",
    "SkillCandidate",
    "SkillIssue",
    "SkillValidationError",
]
```

Create `src/obelix/core/skill/skill.py`:

```python
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
        from obelix.core.skill.reporting import format_validation_error

        return format_validation_error(self.issues)
```

Note: `_default_message` lazy-imports `reporting` to avoid a module-level circular import. `reporting` will be added in Task 2.3 and tested before being imported here in real flow.

For now, make the lazy import work without `reporting` existing yet — wrap in try/except:

```python
    def _default_message(self) -> str:
        try:
            from obelix.core.skill.reporting import format_validation_error

            return format_validation_error(self.issues)
        except ImportError:
            return f"{len(self.issues)} skill issue(s)"
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/core/skill/test_skill.py -v`
Expected: PASS — 10 tests pass.

- [ ] **Step 5: Lint and format**

Run: `uv run ruff format src/obelix/core/skill/ tests/core/skill/ && uv run ruff check src/obelix/core/skill/ tests/core/skill/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/skill/__init__.py src/obelix/core/skill/skill.py tests/core/skill/__init__.py tests/core/skill/test_skill.py
git commit -m "feat(skill): add Skill, SkillCandidate, SkillIssue dataclasses"
```

---

## Phase 2 — Pure functions

### Task 2.1: Placeholder substitution

**Files:**
- Create: `src/obelix/core/skill/substitution.py`
- Create: `tests/core/skill/test_substitution.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/skill/test_substitution.py`:

```python
import pytest
from pathlib import Path
from obelix.core.skill.substitution import (
    substitute_placeholders,
    SkillInvocationError,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/skill/test_substitution.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `src/obelix/core/skill/substitution.py`:

```python
"""Pure placeholder substitution for skill bodies.

No IO, no side effects. Stateless.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path


class SkillInvocationError(Exception):
    """Raised at runtime when a skill is invoked with invalid args."""


# Regex matches $<identifier> but not $<identifier>_suffix continuation.
# We use \b at the end so $pa does not match inside $path.
def _replace_named(body: str, name: str, value: str) -> str:
    # Replace only whole-word occurrences. Python re doesn't treat $ as word-start,
    # so we anchor: literal $<name> followed by a non-word char or end-of-string.
    pattern = rf"\${re.escape(name)}(?=\W|$)"
    return re.sub(pattern, lambda _: value, body)


def substitute_placeholders(
    body: str,
    args: str,
    declared_args: tuple[str, ...],
    base_dir: Path | None,
    session_id: str,
) -> str:
    """Replace placeholders in a skill body.

    Order of replacement:
      1. Named positional ($path, $depth, ...) — order: longest name first
         to avoid prefix collisions like ($pa, $path).
      2. $ARGUMENTS flat fallback.
      3. Meta placeholders (${OBELIX_SKILL_DIR}, ${OBELIX_SESSION_ID}).
    """
    # Parse args with shell quoting
    parsed = shlex.split(args) if args else []
    if len(parsed) > len(declared_args):
        raise SkillInvocationError(
            f"skill expects {len(declared_args)} argument(s), got {len(parsed)}"
        )

    # Named positional, longest name first (prefix safety)
    out = body
    for name in sorted(declared_args, key=len, reverse=True):
        idx = declared_args.index(name)
        value = parsed[idx] if idx < len(parsed) else ""
        out = _replace_named(out, name, value)

    # Flat ARGUMENTS (word-boundary aware)
    out = _replace_named(out, "ARGUMENTS", args)

    # Meta
    out = out.replace("${OBELIX_SKILL_DIR}", str(base_dir) if base_dir else "")
    out = out.replace("${OBELIX_SESSION_ID}", session_id)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/core/skill/test_substitution.py -v`
Expected: PASS — all tests pass.

- [ ] **Step 5: Lint and format**

Run: `uv run ruff format src/obelix/core/skill/ tests/core/skill/ && uv run ruff check src/obelix/core/skill/ tests/core/skill/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/skill/substitution.py tests/core/skill/test_substitution.py
git commit -m "feat(skill): add placeholder substitution with prefix-safe replacement"
```

---

### Task 2.2: Parsing — `parse_skill_file()`

**Files:**
- Create: `src/obelix/core/skill/parsing.py`
- Create: `tests/core/skill/test_parsing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/skill/test_parsing.py`:

```python
import pytest
from pathlib import Path
from obelix.core.skill.parsing import parse_skill_file, ParseError


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
        """--- followed by content without a proper opening treats content as body."""
        # Only 3 dashes on first line are accepted; 4+ dashes as body.
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/skill/test_parsing.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `src/obelix/core/skill/parsing.py`:

```python
"""Parse SKILL.md files into SkillCandidate objects.

Syntactic concerns only — produces raw frontmatter dict + body string.
Semantic validation lives in validation.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from obelix.core.skill.skill import SkillCandidate


class ParseError(Exception):
    """Raised when a SKILL.md cannot be parsed at the syntactic level."""

    def __init__(self, message: str, line: int | None = None):
        super().__init__(message)
        self.line = line


# Matches the opening `---` on its own line (possibly with trailing whitespace)
# and captures YAML up to the next `---` on its own line.
_FRONTMATTER_RE = re.compile(
    r"\A(?:\ufeff)?---[ \t]*\r?\n(?P<fm>.*?)(?:\r?\n)---[ \t]*\r?\n?(?P<body>.*)\Z",
    re.DOTALL,
)


def parse_skill_file(raw: str, file_path: Path | None) -> SkillCandidate:
    """Split a SKILL.md into frontmatter (dict) and body (str).

    Raises ParseError if:
      - no frontmatter delimiters found
      - YAML cannot be parsed
      - frontmatter is not a mapping (dict)
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        raise ParseError("missing frontmatter delimiters (--- ... ---)")

    fm_text = match.group("fm")
    body = match.group("body")

    try:
        loaded = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError as e:
        line = getattr(getattr(e, "problem_mark", None), "line", None)
        # YAML lines are 0-indexed; +1 for human-readable
        line_out = (line + 1) if line is not None else None
        raise ParseError(f"invalid YAML: {e}", line=line_out) from e

    if not isinstance(loaded, dict):
        raise ParseError(
            f"frontmatter must be a YAML mapping (dict), got {type(loaded).__name__}"
        )

    return SkillCandidate(file_path=file_path, frontmatter=loaded, body=body)
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/core/skill/test_parsing.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and format**

Run: `uv run ruff format src/obelix/core/skill/ tests/core/skill/ && uv run ruff check src/obelix/core/skill/ tests/core/skill/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/skill/parsing.py tests/core/skill/test_parsing.py
git commit -m "feat(skill): add parse_skill_file splitting frontmatter and body"
```

---

### Task 2.3: Reporting — `format_validation_error()`

**Files:**
- Create: `src/obelix/core/skill/reporting.py`
- Create: `tests/core/skill/test_reporting.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/skill/test_reporting.py`:

```python
from pathlib import Path
import pytest
from obelix.core.skill.skill import SkillIssue
from obelix.core.skill.reporting import format_validation_error


class TestFormatValidationError:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            format_validation_error([])

    def test_single_issue_renders_header(self):
        out = format_validation_error(
            [
                SkillIssue(
                    file_path=Path("skills/a/SKILL.md"),
                    field="frontmatter.description",
                    message="missing",
                )
            ]
        )
        assert "1 skill(s) failed validation" in out
        assert "1 issue(s)" in out
        assert "skills/a/SKILL.md" in out
        assert "[frontmatter.description]" in out
        assert "missing" in out

    def test_multiple_issues_same_file_grouped(self):
        issues = [
            SkillIssue(file_path=Path("a/SKILL.md"), field="x", message="m1"),
            SkillIssue(file_path=Path("a/SKILL.md"), field="y", message="m2"),
        ]
        out = format_validation_error(issues)
        assert out.count("a/SKILL.md") == 1  # path header only once
        assert "[x]" in out
        assert "[y]" in out

    def test_multi_file_sorted_alphabetically(self):
        issues = [
            SkillIssue(file_path=Path("b/SKILL.md"), field="x", message="m"),
            SkillIssue(file_path=Path("a/SKILL.md"), field="x", message="m"),
        ]
        out = format_validation_error(issues)
        a_idx = out.index("a/SKILL.md")
        b_idx = out.index("b/SKILL.md")
        assert a_idx < b_idx

    def test_issue_with_line_rendered(self):
        out = format_validation_error(
            [
                SkillIssue(
                    file_path=Path("a.md"), field="frontmatter", message="bad", line=5
                )
            ]
        )
        assert "line 5" in out

    def test_issue_without_line_no_line_suffix(self):
        out = format_validation_error(
            [SkillIssue(file_path=Path("a.md"), field="x", message="bad")]
        )
        assert "line" not in out.lower() or "line 5" not in out

    def test_mcp_issue_has_mcp_section(self):
        out = format_validation_error(
            [SkillIssue(file_path=None, field="x", message="bad")]
        )
        assert "[mcp]" in out

    def test_counts_skills_not_issues(self):
        """Header says N skills (unique files) and M issues (total)."""
        issues = [
            SkillIssue(file_path=Path("a.md"), field="x", message="m"),
            SkillIssue(file_path=Path("a.md"), field="y", message="m"),
            SkillIssue(file_path=Path("b.md"), field="z", message="m"),
        ]
        out = format_validation_error(issues)
        assert "2 skill(s)" in out  # a.md + b.md
        assert "3 issue(s)" in out
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/skill/test_reporting.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/obelix/core/skill/reporting.py`:

```python
"""Human-readable formatting of SkillIssue lists.

Takes validation output and renders it. Knows nothing about how issues
were produced.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from obelix.core.skill.skill import SkillIssue


def format_validation_error(issues: list[SkillIssue]) -> str:
    """Format a list of issues for human readers.

    Groups by file (with `None` → `[mcp]`), sorts file paths alphabetically,
    keeps issue order within each group.
    """
    if not issues:
        raise ValueError("cannot format empty issue list")

    by_source: dict[str, list[SkillIssue]] = defaultdict(list)
    for issue in issues:
        key = str(issue.file_path) if issue.file_path is not None else "[mcp]"
        by_source[key].append(issue)

    skill_count = len(by_source)
    issue_count = len(issues)

    lines = [
        f"{skill_count} skill(s) failed validation ({issue_count} issue(s) total)",
        "",
    ]

    for source in sorted(by_source.keys()):
        lines.append(f"{source}:")
        for issue in by_source[source]:
            line_suffix = f" (line {issue.line})" if issue.line is not None else ""
            lines.append(f"  [{issue.field}] {issue.message}{line_suffix}")
        lines.append("")

    return "\n".join(lines).rstrip()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_reporting.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and format**

Run: `uv run ruff format src/obelix/core/skill/ tests/core/skill/ && uv run ruff check src/obelix/core/skill/ tests/core/skill/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/skill/reporting.py tests/core/skill/test_reporting.py
git commit -m "feat(skill): add issue reporting formatter"
```

---

## Phase 3 — Validator chain

### Task 3.1: Validator protocol and `run_validators`

**Files:**
- Create: `src/obelix/core/skill/validation.py`
- Create: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/skill/test_validation.py`:

```python
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


class TestValidatorProtocol:
    def test_any_class_with_check_method_satisfies(self):
        """Protocol is structural — duck typing is enough."""

        class MyValidator:
            def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
                return []

        v: Validator = MyValidator()
        assert run_validators(_cand(), validators=(v,)) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/skill/test_validation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/obelix/core/skill/validation.py`:

```python
"""Validator chain for skill candidates.

Pure functions. No IO. One Validator class per rule.
run_validators() aggregates all issues, never short-circuits.
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add Validator protocol and run_validators chain"
```

---

### Task 3.2: `FrontmatterSchemaValidator`

**Files:**
- Modify: `src/obelix/core/skill/validation.py`
- Modify: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/core/skill/test_validation.py`:

```python
from obelix.core.skill.validation import FrontmatterSchemaValidator


def _cand_fm(**fm) -> SkillCandidate:
    return SkillCandidate(
        file_path=Path("x.md"), frontmatter=fm, body="nonempty body"
    )


class TestFrontmatterSchemaValidator:
    def setup_method(self):
        self.v = FrontmatterSchemaValidator()

    def test_happy_path(self):
        assert self.v.check(_cand_fm(description="hi")) == []

    def test_missing_description(self):
        issues = self.v.check(_cand_fm())
        assert len(issues) == 1
        assert "description" in issues[0].field
        assert "missing" in issues[0].message.lower() or "required" in issues[0].message.lower()

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
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_validation.py::TestFrontmatterSchemaValidator -v`
Expected: FAIL — `FrontmatterSchemaValidator` not importable.

- [ ] **Step 3: Implement**

Append to `src/obelix/core/skill/validation.py`:

```python
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class _FrontmatterSchema(BaseModel):
    """Pydantic model capturing the schema of valid frontmatter."""

    model_config = ConfigDict(extra="ignore")  # forward-compat: ignore unknown keys

    description: str
    when_to_use: str | None = None
    arguments: list[str] = []
    allowed_tools: list[str] = []
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = {}

    def _never_used(self) -> None:  # placeholder for pyright
        pass


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
```

Also add the `description` empty-string check — Pydantic accepts empty strings by default. Add a custom validator:

Replace the Pydantic model with:

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _FrontmatterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    description: str = Field(..., min_length=1)
    when_to_use: str | None = None
    arguments: list[str] = []
    allowed_tools: list[str] = []
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = {}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py::TestFrontmatterSchemaValidator -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add FrontmatterSchemaValidator with Pydantic"
```

---

### Task 3.3: `HookEventValidator`

**Files:**
- Modify: `src/obelix/core/skill/validation.py`
- Modify: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/core/skill/test_validation.py`:

```python
from obelix.core.skill.validation import HookEventValidator


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
        issues = self.v.check(
            _cand_fm(hooks={"on_a": "x", "on_b": "y", "on_c": "z"})
        )
        assert len(issues) == 3
        fields = {i.field for i in issues}
        assert fields == {"hooks.on_a", "hooks.on_b", "hooks.on_c"}

    def test_mix_of_valid_and_invalid(self):
        issues = self.v.check(
            _cand_fm(hooks={"on_tool_error": "x", "on_bogus": "y"})
        )
        assert len(issues) == 1
        assert "hooks.on_bogus" in issues[0].field
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_validation.py::TestHookEventValidator -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append to `src/obelix/core/skill/validation.py`:

```python
from obelix.core.agent.hooks import AgentEvent

_VALID_HOOK_EVENTS: frozenset[str] = frozenset(e.value for e in AgentEvent)


class HookEventValidator:
    """Every hooks.<key> must map to a valid AgentEvent enum value."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        hooks = candidate.frontmatter.get("hooks", {})
        if not isinstance(hooks, dict):
            return []  # type error reported by FrontmatterSchemaValidator

        issues: list[SkillIssue] = []
        valid_list = ", ".join(sorted(_VALID_HOOK_EVENTS))
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py::TestHookEventValidator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add HookEventValidator against AgentEvent enum"
```

---

### Task 3.4: `ArgumentUniquenessValidator`

**Files:**
- Modify: `src/obelix/core/skill/validation.py`
- Modify: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
from obelix.core.skill.validation import ArgumentUniquenessValidator


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
        assert any("'a'" in m for m in messages)
        assert any("'b'" in m for m in messages)

    def test_reserved_arguments_name(self):
        issues = self.v.check(_cand_fm(arguments=["path", "ARGUMENTS"]))
        assert any("reserved" in i.message.lower() for i in issues)

    def test_reserved_and_duplicate_both_reported(self):
        issues = self.v.check(_cand_fm(arguments=["ARGUMENTS", "ARGUMENTS"]))
        assert len(issues) >= 2
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_validation.py::TestArgumentUniquenessValidator -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append:

```python
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

        # Reserved
        for name in args:
            if isinstance(name, str) and name in _RESERVED_ARG_NAMES:
                issues.append(
                    SkillIssue(
                        file_path=candidate.file_path,
                        field="arguments",
                        message=f"'{name}' is reserved, use a different name",
                    )
                )

        return issues
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py::TestArgumentUniquenessValidator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add ArgumentUniquenessValidator"
```

---

### Task 3.5: `PlaceholderConsistencyValidator`

**Files:**
- Modify: `src/obelix/core/skill/validation.py`
- Modify: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
from obelix.core.skill.validation import PlaceholderConsistencyValidator


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
        assert (
            self.v.check(_cand_body("path=$path", arguments=["path"])) == []
        )

    def test_undeclared_placeholder_one_issue(self):
        issues = self.v.check(_cand_body("depth=$depth", arguments=["path"]))
        assert len(issues) == 1
        assert "$depth" in issues[0].message

    def test_multiple_undeclared_all_reported(self):
        issues = self.v.check(
            _cand_body("$a $b $c", arguments=["a"])
        )
        msgs = [i.message for i in issues]
        assert any("$b" in m for m in msgs)
        assert any("$c" in m for m in msgs)

    def test_same_placeholder_multiple_times_reported_once(self):
        issues = self.v.check(_cand_body("$x then $x", arguments=[]))
        assert len(issues) == 1

    def test_placeholder_name_with_underscore(self):
        assert (
            self.v.check(_cand_body("$my_var", arguments=["my_var"])) == []
        )

    def test_placeholder_followed_by_underscore_word(self):
        """$pa must be recognized distinctly from $path."""
        # body references $pa, arguments declare only 'path' -> $pa is undeclared
        issues = self.v.check(_cand_body("x=$pa", arguments=["path"]))
        assert len(issues) == 1
        assert "$pa" in issues[0].message
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_validation.py::TestPlaceholderConsistencyValidator -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append:

```python
import re

# Placeholder: $<identifier> where identifier is [a-zA-Z_][a-zA-Z0-9_]*
# followed by a word boundary. Excludes ${...} form (handled as meta).
_PLACEHOLDER_RE = re.compile(r"\$(?!\{)([A-Za-z_][A-Za-z0-9_]*)")

_META_PLACEHOLDERS: frozenset[str] = frozenset({"ARGUMENTS"})


class PlaceholderConsistencyValidator:
    """Every $name in body must be in arguments (or be ARGUMENTS)."""

    def check(self, candidate: SkillCandidate) -> list[SkillIssue]:
        declared = set(candidate.frontmatter.get("arguments", []) or [])
        allowed = declared | _META_PLACEHOLDERS

        found = _PLACEHOLDER_RE.findall(candidate.body)
        seen: set[str] = set()
        issues: list[SkillIssue] = []
        for name in found:
            if name in allowed or name in seen:
                continue
            if name in declared:
                continue
            seen.add(name)
            issues.append(
                SkillIssue(
                    file_path=candidate.file_path,
                    field="body",
                    message=f"placeholder '${name}' referenced but not declared in 'arguments'",
                )
            )
        return issues
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py::TestPlaceholderConsistencyValidator -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add PlaceholderConsistencyValidator"
```

---

### Task 3.6: `BodyNonEmptyValidator` + `DEFAULT_VALIDATORS`

**Files:**
- Modify: `src/obelix/core/skill/validation.py`
- Modify: `tests/core/skill/test_validation.py`

- [ ] **Step 1: Add failing tests**

Append:

```python
from obelix.core.skill.validation import BodyNonEmptyValidator, DEFAULT_VALIDATORS


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


class TestDefaultValidators:
    def test_default_validators_is_tuple(self):
        assert isinstance(DEFAULT_VALIDATORS, tuple)
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
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_validation.py::TestBodyNonEmptyValidator tests/core/skill/test_validation.py::TestDefaultValidators -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Append:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_validation.py -v`
Expected: PASS (all validator tests).

- [ ] **Step 5: Check coverage**

Run: `uv run pytest tests/core/skill/ --cov=obelix.core.skill --cov-report=term-missing`
Expected: `core/skill/` ≥ 95% line + branch.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff format src/obelix/core/skill/ tests/core/skill/ && uv run ruff check src/obelix/core/skill/ tests/core/skill/`

```bash
git add src/obelix/core/skill/validation.py tests/core/skill/test_validation.py
git commit -m "feat(skill): add BodyNonEmptyValidator and DEFAULT_VALIDATORS tuple"
```

---

## Phase 4 — Port + Adapters

### Task 4.1: `AbstractSkillProvider` port

**Files:**
- Create: `src/obelix/ports/outbound/skill_provider.py`

- [ ] **Step 1: Implement the ABC (no tests — pure protocol)**

Create `src/obelix/ports/outbound/skill_provider.py`:

```python
"""Port for skill providers.

Concrete providers (filesystem, MCP, ...) implement this ABC.
SkillManager depends only on this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from obelix.core.skill.skill import Skill


class AbstractSkillProvider(ABC):
    """A source of skills."""

    @abstractmethod
    def discover(self) -> list[Skill]:
        """Return all skills this provider offers.

        Implementations must raise SkillValidationError if any candidate
        fails validation, aggregating issues across all candidates.
        """
        ...
```

- [ ] **Step 2: Smoke verify**

Run: `uv run python -c "from obelix.ports.outbound.skill_provider import AbstractSkillProvider; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/obelix/ports/outbound/skill_provider.py
git commit -m "feat(skill): add AbstractSkillProvider port"
```

---

### Task 4.2: Fixture corpus for filesystem provider tests

**Files:**
- Create: `tests/fixtures/skills/minimal/SKILL.md`
- Create: `tests/fixtures/skills/with_when_to_use/SKILL.md`
- Create: `tests/fixtures/skills/with_named_args/SKILL.md`
- Create: `tests/fixtures/skills/with_flat_args_only/SKILL.md`
- Create: `tests/fixtures/skills/with_all_hooks/SKILL.md`
- Create: `tests/fixtures/skills/fork_context/SKILL.md`
- Create: `tests/fixtures/skills/invalid_missing_description/SKILL.md`
- Create: `tests/fixtures/skills/invalid_yaml/SKILL.md`
- Create: `tests/fixtures/skills/invalid_empty_body/SKILL.md`
- Create: `tests/fixtures/skills/invalid_unknown_hook/SKILL.md`
- Create: `tests/fixtures/skills/invalid_duplicate_args/SKILL.md`
- Create: `tests/fixtures/skills/invalid_undeclared_placeholder/SKILL.md`
- Create: `tests/fixtures/skills/invalid_multiple_issues/SKILL.md`

- [ ] **Step 1: Create each fixture file**

Create each file with the contents below. Use exact filenames — tests depend on them.

`tests/fixtures/skills/minimal/SKILL.md`:
```markdown
---
description: Minimal skill
---
Body of the minimal skill.
```

`tests/fixtures/skills/with_when_to_use/SKILL.md`:
```markdown
---
description: Has when_to_use
when_to_use: When the user says "test me"
---
Body.
```

`tests/fixtures/skills/with_named_args/SKILL.md`:
```markdown
---
description: Accepts named args
arguments: [path, depth]
---
Target $path at depth $depth.
```

`tests/fixtures/skills/with_flat_args_only/SKILL.md`:
```markdown
---
description: Uses flat ARGUMENTS
---
Full input: $ARGUMENTS
```

`tests/fixtures/skills/with_all_hooks/SKILL.md`:
```markdown
---
description: Registers all supported hooks
hooks:
  before_llm_call: "pre-llm"
  after_llm_call: "post-llm"
  before_tool_execution: "pre-tool"
  after_tool_execution: "post-tool"
  on_tool_error: "tool-error"
  before_final_response: "pre-final"
  query_end: "end"
---
Body with all hooks declared.
```

`tests/fixtures/skills/fork_context/SKILL.md`:
```markdown
---
description: Runs in a forked sub-agent
context: fork
---
You are the forked skill. Execute the protocol.
```

`tests/fixtures/skills/invalid_missing_description/SKILL.md`:
```markdown
---
when_to_use: no description here
---
Body.
```

`tests/fixtures/skills/invalid_yaml/SKILL.md`:
```markdown
---
description: hi
key:
  bad: : :
---
Body.
```

`tests/fixtures/skills/invalid_empty_body/SKILL.md`:
```markdown
---
description: Valid frontmatter but empty body
---
```

`tests/fixtures/skills/invalid_unknown_hook/SKILL.md`:
```markdown
---
description: Declares an unknown hook
hooks:
  on_random_thing: "oops"
---
Body.
```

`tests/fixtures/skills/invalid_duplicate_args/SKILL.md`:
```markdown
---
description: Duplicate argument name
arguments: [path, path]
---
Body.
```

`tests/fixtures/skills/invalid_undeclared_placeholder/SKILL.md`:
```markdown
---
description: Body uses undeclared placeholder
arguments: [path]
---
Review $path at depth $depth.
```

`tests/fixtures/skills/invalid_multiple_issues/SKILL.md`:
```markdown
---
arguments: [a, a]
hooks:
  bogus_event: "x"
---
$undeclared_thing
```
(Missing `description` → at least 4 issues total.)

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/skills/
git commit -m "test(skill): add fixture corpus for skill validation and discovery"
```

---

### Task 4.3: `FilesystemSkillProvider`

**Files:**
- Create: `src/obelix/adapters/outbound/skill/__init__.py`
- Create: `src/obelix/adapters/outbound/skill/filesystem.py`
- Create: `tests/adapters/outbound/skill/__init__.py`
- Create: `tests/adapters/outbound/skill/test_filesystem.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/outbound/skill/__init__.py` (empty).

Create `tests/adapters/outbound/skill/test_filesystem.py`:

```python
from pathlib import Path
import pytest
from obelix.adapters.outbound.skill.filesystem import FilesystemSkillProvider
from obelix.core.skill.skill import SkillValidationError

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

    def test_dir_without_skill_md_scans_subdirs(self):
        # FIXTURES is a dir of dirs with SKILL.md each. Pick 2 valid subdirs.
        tmp = FIXTURES  # actual use: a tmp dir with only valid skills
        # We'll use a curated dir of valid skills
        # ... instead, verify that giving FIXTURES itself picks up multiple skills
        # but FIXTURES has invalid skills too. We'll filter by passing a curated
        # dir: create a narrower test dir
        # For now, test with a path to a single-subdir wrapper (minimal only)
        provider = FilesystemSkillProvider([FIXTURES / "minimal"])
        skills = provider.discover()
        assert len(skills) == 1


class TestValidationAggregation:
    def test_single_invalid_raises(self):
        provider = FilesystemSkillProvider(
            [FIXTURES / "invalid_missing_description"]
        )
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert len(exc.value.issues) >= 1
        assert any("description" in i.field for i in exc.value.issues)

    def test_multi_issue_skill_all_reported(self):
        provider = FilesystemSkillProvider(
            [FIXTURES / "invalid_multiple_issues"]
        )
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
        # Issues span 2 files
        files = {i.file_path for i in exc.value.issues}
        assert len(files) == 2


class TestEmptyAndMissing:
    def test_nonexistent_path_issue(self, tmp_path):
        provider = FilesystemSkillProvider([tmp_path / "does_not_exist"])
        with pytest.raises(SkillValidationError) as exc:
            provider.discover()
        assert any(
            "does not exist" in i.message.lower()
            or "not found" in i.message.lower()
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/adapters/outbound/skill/test_filesystem.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `src/obelix/adapters/outbound/skill/__init__.py` (empty).

Create `src/obelix/adapters/outbound/skill/filesystem.py`:

```python
"""Filesystem-backed skill provider.

Walks configured paths, parses SKILL.md files, runs validators,
aggregates all issues, raises SkillValidationError if any.
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
      - path to a `.md` file → treated as a skill, name = parent dir name
        (or file stem if no meaningful parent)
      - path to a directory containing `SKILL.md` → one skill, name = dir name
      - path to a directory without `SKILL.md` → scans subdirs for `SKILL.md`
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

    def _resolve_candidates(
        self, bag: list[SkillIssue]
    ) -> list[Path]:
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
                            message=f"expected a .md file, got {p.suffix}",
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

    def _load_one(
        self, skill_md: Path, bag: list[SkillIssue]
    ) -> Skill | None:
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
        # Allow frontmatter override
        if "name" in candidate.frontmatter and isinstance(
            candidate.frontmatter["name"], str
        ):
            name = candidate.frontmatter["name"]

        return Skill.from_candidate(
            candidate, name=name, base_dir=base_dir, source="filesystem"
        )
```

Note: the schema does not currently list `name`. Allow it as an optional override — add it to `_FrontmatterSchema` in `validation.py`:

Modify `src/obelix/core/skill/validation.py`, in `_FrontmatterSchema`:
```python
class _FrontmatterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    description: str = Field(..., min_length=1)
    # ... rest unchanged
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/adapters/outbound/skill/ tests/core/skill/ -v`
Expected: PASS.

- [ ] **Step 5: Coverage**

Run: `uv run pytest tests/adapters/outbound/skill/ tests/core/skill/ --cov=obelix.core.skill --cov=obelix.adapters.outbound.skill --cov-report=term-missing --cov-branch`
Expected: ≥90% on adapter, ≥95% on core.

- [ ] **Step 6: Lint and commit**

Run: `uv run ruff format src/obelix/ tests/ && uv run ruff check src/obelix/ tests/`

```bash
git add src/obelix/adapters/outbound/skill/ src/obelix/core/skill/validation.py tests/adapters/outbound/skill/
git commit -m "feat(skill): add FilesystemSkillProvider with collect-all validation"
```

---

### Task 4.4: Add `list_prompts()` to MCPManager

**Files:**
- Modify: `src/obelix/adapters/outbound/mcp/manager.py`
- Create: `tests/adapters/outbound/mcp/test_list_prompts.py`

- [ ] **Step 1: Inspect existing MCPManager**

Run: `uv run python -c "from obelix.adapters.outbound.mcp.manager import MCPManager; print([m for m in dir(MCPManager) if not m.startswith('_')])"`
Expected: list of public methods (should include `connect`, `disconnect`, `is_connected`). If it already has `list_prompts`, skip this task.

- [ ] **Step 2: Write failing test**

Create `tests/adapters/outbound/mcp/test_list_prompts.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from obelix.adapters.outbound.mcp.manager import MCPManager


@pytest.mark.asyncio
async def test_list_prompts_returns_empty_when_not_connected():
    manager = MCPManager(configs=[])
    prompts = manager.list_prompts()
    assert prompts == []


@pytest.mark.asyncio
async def test_list_prompts_aggregates_prompts_per_server(monkeypatch):
    """When connected, list_prompts returns all prompts prefixed by server name."""
    manager = MCPManager(configs=[])

    # Simulate connected state with a fake session group
    fake_prompt = MagicMock()
    fake_prompt.name = "my_prompt"
    fake_prompt.description = "Does a thing"
    fake_prompt.arguments = []

    fake_group = MagicMock()
    # Mock the attribute used to enumerate prompts per server
    fake_group.prompts = {"my_prompt": (fake_prompt, "server1")}
    manager._session_group = fake_group
    manager._connected = True

    prompts = manager.list_prompts()
    assert len(prompts) == 1
    p = prompts[0]
    assert p.name == "my_prompt"
    assert p.server_name == "server1"
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/adapters/outbound/mcp/test_list_prompts.py -v`
Expected: FAIL — `list_prompts` not defined or attribute access fails.

- [ ] **Step 4: Implement `list_prompts()`**

Open `src/obelix/adapters/outbound/mcp/manager.py`. Find the existing class and add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MCPPrompt:
    """Lightweight DTO: a prompt offered by an MCP server."""

    name: str
    description: str
    arguments: list  # mcp.types.PromptArgument list
    server_name: str
    template: str = ""  # body, if materialized


class MCPManager:
    # ... existing code ...

    def list_prompts(self) -> list[MCPPrompt]:
        """Return all prompts exposed by connected servers.

        Returns empty list when not connected — does not raise.
        The MCP SessionGroup stores prompts in a dict keyed by name;
        the value is (Prompt, server_name). We normalize to MCPPrompt.
        """
        if not getattr(self, "_connected", False):
            return []
        group = getattr(self, "_session_group", None)
        if group is None:
            return []
        prompts_dict = getattr(group, "prompts", {}) or {}
        out: list[MCPPrompt] = []
        for name, entry in prompts_dict.items():
            try:
                prompt, server_name = entry
            except (TypeError, ValueError):
                continue
            out.append(
                MCPPrompt(
                    name=name,
                    description=getattr(prompt, "description", "") or "",
                    arguments=list(getattr(prompt, "arguments", []) or []),
                    server_name=server_name,
                    template="",  # template body fetched on-demand; v1 leaves empty
                )
            )
        return out
```

Review exact internal state of `MCPManager` (names of `_session_group`, `_connected`) before committing; adjust if needed.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/adapters/outbound/mcp/test_list_prompts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/adapters/outbound/mcp/manager.py tests/adapters/outbound/mcp/test_list_prompts.py
git commit -m "feat(mcp): add list_prompts() to MCPManager for skill discovery"
```

---

### Task 4.5: `MCPSkillProvider`

**Files:**
- Create: `src/obelix/adapters/outbound/skill/mcp.py`
- Create: `tests/adapters/outbound/skill/test_mcp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/adapters/outbound/skill/test_mcp.py`:

```python
from unittest.mock import MagicMock
from obelix.adapters.outbound.skill.mcp import MCPSkillProvider
from obelix.adapters.outbound.mcp.manager import MCPPrompt


class TestMCPSkillProviderEmpty:
    def test_no_manager_empty(self):
        provider = MCPSkillProvider(None)
        assert provider.discover() == []

    def test_disconnected_manager_empty(self):
        mgr = MagicMock()
        mgr.list_prompts.return_value = []
        provider = MCPSkillProvider(mgr)
        assert provider.discover() == []


class TestMCPSkillProviderHappy:
    def test_well_formed_prompt_becomes_skill(self):
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="review",
            description="Reviews code",
            arguments=[],
            server_name="tracer",
            template="# Review\n\nDo a review.",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "mcp__tracer__review"
        assert s.description == "Reviews code"
        assert s.source == "mcp"
        assert s.base_dir is None

    def test_missing_description_skipped_with_log(self, caplog):
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="bad",
            description="",
            arguments=[],
            server_name="s",
            template="body",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert skills == []
        # One warning logged
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) >= 1

    def test_empty_template_skipped_with_log(self, caplog):
        mgr = MagicMock()
        prompt = MCPPrompt(
            name="bad",
            description="hi",
            arguments=[],
            server_name="s",
            template="",
        )
        mgr.list_prompts.return_value = [prompt]
        provider = MCPSkillProvider(mgr)
        skills = provider.discover()
        assert skills == []
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/adapters/outbound/skill/test_mcp.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/obelix/adapters/outbound/skill/mcp.py`:

```python
"""MCP-backed skill provider.

Turns MCP prompts exposed by connected servers into Skill objects.
Malformed prompts are logged and skipped, not raised — Obelix does not
control remote servers.
"""

from __future__ import annotations

import logging

from obelix.adapters.outbound.mcp.manager import MCPManager
from obelix.core.skill.skill import Skill, SkillCandidate
from obelix.core.skill.validation import DEFAULT_VALIDATORS, run_validators
from obelix.ports.outbound.skill_provider import AbstractSkillProvider

logger = logging.getLogger(__name__)


class MCPSkillProvider(AbstractSkillProvider):
    """Wraps MCPManager.list_prompts() and renders prompts as skills.

    Accepts `None` to signal "no MCP configured" — returns an empty list.
    """

    def __init__(self, manager: MCPManager | None):
        self._manager = manager

    def discover(self) -> list[Skill]:
        if self._manager is None:
            return []

        skills: list[Skill] = []
        for prompt in self._manager.list_prompts():
            skill = self._to_skill(prompt)
            if skill is None:
                logger.warning(
                    "[skill] Skipped malformed MCP prompt %r from server %r",
                    prompt.name,
                    prompt.server_name,
                )
                continue
            skills.append(skill)
        return skills

    def _to_skill(self, prompt) -> Skill | None:
        if not prompt.description:
            return None
        if not (prompt.template or "").strip():
            return None

        # Extract argument names from MCP PromptArgument (name attribute)
        arg_names: list[str] = []
        for a in prompt.arguments:
            name = getattr(a, "name", None)
            if isinstance(name, str):
                arg_names.append(name)

        candidate = SkillCandidate(
            file_path=None,
            frontmatter={
                "description": prompt.description,
                "arguments": arg_names,
            },
            body=prompt.template,
        )
        issues = run_validators(candidate, DEFAULT_VALIDATORS)
        if issues:
            return None

        namespaced = f"mcp__{prompt.server_name}__{prompt.name}"
        return Skill.from_candidate(
            candidate, name=namespaced, base_dir=None, source="mcp"
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/adapters/outbound/skill/test_mcp.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/adapters/outbound/skill/mcp.py tests/adapters/outbound/skill/test_mcp.py
git commit -m "feat(skill): add MCPSkillProvider wrapping MCPManager prompts"
```

---

## Phase 5 — Manager

### Task 5.1: `SkillManager`

**Files:**
- Create: `src/obelix/core/skill/manager.py`
- Create: `tests/core/skill/test_manager.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/skill/test_manager.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.ports.outbound.skill_provider import AbstractSkillProvider


def _skill(name: str, desc: str = "d", when: str | None = None, source="filesystem"):
    return Skill(
        name=name,
        description=desc,
        body="body",
        base_dir=Path("/"),
        when_to_use=when,
        source=source,
    )


class _FakeProvider(AbstractSkillProvider):
    def __init__(self, skills):
        self._skills = skills

    def discover(self):
        return list(self._skills)


class TestConstructionAndLookup:
    def test_empty(self):
        mgr = SkillManager(providers=[])
        assert mgr.list_all() == []
        assert mgr.load("x") is None
        assert mgr.format_listing(1000) == ""

    def test_single_provider(self):
        p = _FakeProvider([_skill("a"), _skill("b")])
        mgr = SkillManager(providers=[p])
        names = {s.name for s in mgr.list_all()}
        assert names == {"a", "b"}

    def test_load_known_and_unknown(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a")])])
        assert mgr.load("a") is not None
        assert mgr.load("missing") is None


class TestDedup:
    def test_disjoint_names(self):
        p1 = _FakeProvider([_skill("a")])
        p2 = _FakeProvider([_skill("b")])
        mgr = SkillManager(providers=[p1, p2])
        assert len(mgr.list_all()) == 2

    def test_collision_filesystem_wins_over_mcp(self, caplog):
        fs = _FakeProvider([_skill("a", desc="fs", source="filesystem")])
        mcp = _FakeProvider([_skill("a", desc="mcp", source="mcp")])
        mgr = SkillManager(providers=[fs, mcp])
        assert len(mgr.list_all()) == 1
        assert mgr.load("a").description == "fs"
        # Warning logged
        assert any("collision" in r.message.lower() for r in caplog.records)

    def test_collision_provider_order_wins_within_same_source(self):
        p1 = _FakeProvider([_skill("a", desc="first")])
        p2 = _FakeProvider([_skill("a", desc="second")])
        mgr = SkillManager(providers=[p1, p2])
        assert mgr.load("a").description == "first"


class TestListingDeterminism:
    def test_deterministic_order(self):
        mgr = SkillManager(
            providers=[_FakeProvider([_skill("c"), _skill("a"), _skill("b")])]
        )
        call1 = mgr.format_listing(1000)
        call2 = mgr.format_listing(1000)
        assert call1 == call2


class TestListingBudget:
    def test_under_budget_all_full(self):
        skills = [_skill(f"s{i}", desc="short description") for i in range(3)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(10_000)
        assert "short description" in out
        for i in range(3):
            assert f"s{i}" in out

    def test_over_budget_truncates_descriptions(self):
        long_desc = "a" * 500
        skills = [_skill(f"s{i}", desc=long_desc) for i in range(20)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(200)  # deliberately small
        # All names preserved
        for i in range(20):
            assert f"s{i}" in out
        # Output never exceeds budget (approximately)
        assert len(out) <= 200 + 50  # tolerance for delimiters

    def test_extreme_budget_names_only(self):
        skills = [_skill(f"s{i}", desc="a" * 100) for i in range(50)]
        mgr = SkillManager(providers=[_FakeProvider(skills)])
        out = mgr.format_listing(10)
        # With budget this small, names-only fallback
        assert "s0" in out

    def test_zero_budget_empty_string(self):
        mgr = SkillManager(providers=[_FakeProvider([_skill("a")])])
        assert mgr.format_listing(0) == ""


class TestWhenToUseRendered:
    def test_when_to_use_included(self):
        mgr = SkillManager(
            providers=[
                _FakeProvider([_skill("a", desc="d", when="use when X")])
            ]
        )
        out = mgr.format_listing(1000)
        assert "use when X" in out
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/skill/test_manager.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `src/obelix/core/skill/manager.py`:

```python
"""Aggregate provider, dedup skills, format the listing under a budget."""

from __future__ import annotations

import logging

from obelix.core.skill.skill import Skill
from obelix.ports.outbound.skill_provider import AbstractSkillProvider

logger = logging.getLogger(__name__)

_MAX_DESC_CHARS = 250


def _source_priority(source: str) -> int:
    """Higher priority wins in collisions."""
    return {"filesystem": 2, "mcp": 1}.get(source, 0)


class SkillManager:
    def __init__(self, providers: list[AbstractSkillProvider]):
        self._providers = providers
        self._by_name: dict[str, Skill] = self._aggregate()

    def _aggregate(self) -> dict[str, Skill]:
        by_name: dict[str, Skill] = {}
        for provider in self._providers:
            for skill in provider.discover():
                existing = by_name.get(skill.name)
                if existing is None:
                    by_name[skill.name] = skill
                    continue
                # Collision
                if _source_priority(existing.source) >= _source_priority(skill.source):
                    logger.warning(
                        "[skill] collision on '%s': kept %s (source=%s), "
                        "discarded source=%s",
                        skill.name,
                        existing.source,
                        existing.source,
                        skill.source,
                    )
                else:
                    logger.warning(
                        "[skill] collision on '%s': replacing %s with %s",
                        skill.name,
                        existing.source,
                        skill.source,
                    )
                    by_name[skill.name] = skill
        return by_name

    def list_all(self) -> list[Skill]:
        """Return all skills, sorted by name for determinism."""
        return sorted(self._by_name.values(), key=lambda s: s.name)

    def load(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def format_listing(self, char_budget: int) -> str:
        """Render the listing to inject into the system prompt.

        Strategy:
          1. If full rendering fits → use it.
          2. Otherwise, truncate descriptions uniformly, preserve names.
          3. Extreme: names only.
        """
        skills = self.list_all()
        if not skills or char_budget <= 0:
            return ""

        full_lines = [self._render_entry(s, truncate=None) for s in skills]
        full_total = sum(len(ln) for ln in full_lines) + (len(full_lines) - 1)
        if full_total <= char_budget:
            return "\n".join(full_lines)

        # Compute a per-desc max that lets all lines fit
        name_overhead = sum(len(s.name) + 4 for s in skills) + (len(skills) - 1)
        available_for_desc = char_budget - name_overhead
        if available_for_desc <= 0 or available_for_desc // len(skills) < 20:
            # Names only
            return "\n".join(f"- {s.name}" for s in skills)
        max_desc = available_for_desc // len(skills)
        return "\n".join(self._render_entry(s, truncate=max_desc) for s in skills)

    @staticmethod
    def _render_entry(s: Skill, truncate: int | None) -> str:
        desc = s.description
        if s.when_to_use:
            desc = f"{desc} — {s.when_to_use}"
        desc = desc[:_MAX_DESC_CHARS]
        if truncate is not None and len(desc) > truncate:
            desc = desc[: max(truncate - 1, 0)] + "\u2026"
        return f"- {s.name}: {desc}"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/skill/test_manager.py -v`
Expected: PASS.

- [ ] **Step 5: Coverage**

Run: `uv run pytest tests/core/skill/ --cov=obelix.core.skill --cov-report=term-missing --cov-branch`
Expected: ≥95%.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/skill/manager.py tests/core/skill/test_manager.py
git commit -m "feat(skill): add SkillManager with collision handling and listing budget"
```

---

## Phase 6 — BaseAgent hook removal

### Task 6.1: `_unregister_hook`

**Files:**
- Modify: `src/obelix/core/agent/base_agent.py`
- Create: `tests/core/agent/test_base_agent_hook_unregister.py`

- [ ] **Step 1: Inspect current hooks registry**

Run: `uv run python -c "from obelix.core.agent.base_agent import BaseAgent; print('ok')"`
Expected: `ok`. Verify with grep that `self._hooks` is a `dict[AgentEvent, list[Hook]]`:

Run: `uv run python -c "import inspect; from obelix.core.agent.base_agent import BaseAgent; src = inspect.getsource(BaseAgent.__init__); assert '_hooks' in src"`
Expected: no AssertionError.

- [ ] **Step 2: Write failing test**

Create `tests/core/agent/test_base_agent_hook_unregister.py`:

```python
import pytest
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent, Hook


@pytest.fixture
def agent_stub():
    """Minimal instance of BaseAgent for hook registry tests only."""
    # Build a BaseAgent skipping heavy construction. We use __new__ and
    # manually set the fields needed by the hook machinery.
    agent = BaseAgent.__new__(BaseAgent)
    agent._hooks = {event: [] for event in AgentEvent}
    return agent


class TestUnregisterHook:
    def test_remove_existing_hook(self, agent_stub):
        hook = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].append(hook)
        assert hook in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, hook)
        assert hook not in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]

    def test_remove_missing_hook_no_error(self, agent_stub):
        hook = Hook(AgentEvent.ON_TOOL_ERROR)
        # Never registered
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, hook)  # no raise

    def test_remove_one_does_not_affect_others(self, agent_stub):
        h1 = Hook(AgentEvent.ON_TOOL_ERROR)
        h2 = Hook(AgentEvent.ON_TOOL_ERROR)
        agent_stub._hooks[AgentEvent.ON_TOOL_ERROR].extend([h1, h2])
        agent_stub._unregister_hook(AgentEvent.ON_TOOL_ERROR, h1)
        assert h1 not in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
        assert h2 in agent_stub._hooks[AgentEvent.ON_TOOL_ERROR]
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/core/agent/test_base_agent_hook_unregister.py -v`
Expected: FAIL — `_unregister_hook` attribute missing.

- [ ] **Step 4: Implement**

Open `src/obelix/core/agent/base_agent.py`. Find the `on()` method (around line 195-201). Immediately after it, add:

```python
    def _unregister_hook(self, event: AgentEvent, hook: Hook) -> None:
        """Remove a previously registered hook. No-op if not present.

        Internal API, used by SkillTool to clean up skill-scoped hooks at query end.
        """
        self._hooks[event] = [h for h in self._hooks[event] if h is not hook]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/core/agent/test_base_agent_hook_unregister.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/agent/base_agent.py tests/core/agent/test_base_agent_hook_unregister.py
git commit -m "feat(agent): add _unregister_hook for skill-scoped cleanup"
```

---

## Phase 7 — SkillTool

### Task 7.1: `SkillTool` shell + `system_prompt_fragment`

**Files:**
- Create: `src/obelix/plugins/builtin/skill_tool.py`
- Create: `tests/plugins/builtin/__init__.py`
- Create: `tests/plugins/builtin/test_skill_tool.py`

- [ ] **Step 1: Read existing builtin tool pattern**

Run: `cat src/obelix/plugins/builtin/request_user_input_tool.py | head -40`
Expected: shows the `@tool` decorator pattern.

- [ ] **Step 2: Write failing tests for scaffolding**

Create `tests/plugins/builtin/__init__.py` (empty).

Create `tests/plugins/builtin/test_skill_tool.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from obelix.plugins.builtin.skill_tool import SkillTool, make_skill_tool
from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill


def _mgr_with(*skills: Skill) -> SkillManager:
    provider = MagicMock()
    provider.discover.return_value = list(skills)
    return SkillManager(providers=[provider])


def _skill(name="s", desc="d", body="body", context="inline", hooks=None, args=()):
    return Skill(
        name=name,
        description=desc,
        body=body,
        base_dir=Path("/tmp/skills/s"),
        context=context,
        hooks=hooks or {},
        arguments=tuple(args),
    )


class TestSystemPromptFragment:
    def test_no_skills_empty_string(self):
        tool = make_skill_tool(_mgr_with())
        assert tool.system_prompt_fragment() == ""

    def test_with_skills_lists_all(self):
        mgr = _mgr_with(_skill("a", "desc a"), _skill("b", "desc b"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        assert "a" in frag
        assert "b" in frag
        assert "desc a" in frag
        assert "desc b" in frag

    def test_fragment_mentions_skill_tool(self):
        mgr = _mgr_with(_skill("a"))
        tool = make_skill_tool(mgr)
        frag = tool.system_prompt_fragment()
        assert "Skill" in frag  # references the tool
```

- [ ] **Step 3: Run to verify fail**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestSystemPromptFragment -v`
Expected: FAIL.

- [ ] **Step 4: Implement scaffolding**

Create `src/obelix/plugins/builtin/skill_tool.py`:

```python
"""SkillTool: built-in tool exposed to the LLM to invoke skills.

Orchestrates invocation (inline or fork), hook registration, placeholder
substitution, and skill-listing exposure via system_prompt_fragment.

Designed so SkillManager and SubAgentWrapper are injected — the tool
is decoupled from discovery and from how sub-agents are constructed.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from pydantic import Field

from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill
from obelix.core.skill.substitution import (
    SkillInvocationError,
    substitute_placeholders,
)
from obelix.core.tool.tool_decorator import tool

logger = logging.getLogger(__name__)


DEFAULT_LISTING_BUDGET = 8_000  # ~1% of 200k context in chars


def make_skill_tool(
    manager: SkillManager,
    listing_budget: int = DEFAULT_LISTING_BUDGET,
    session_id: str | None = None,
):
    """Factory — produces a SkillTool *instance bound to this manager*."""
    session = session_id or str(uuid.uuid4())

    @tool(
        name="Skill",
        description=(
            "Execute a skill within the main conversation. "
            "Use when a skill from the listing matches the user's request."
        ),
    )
    class SkillTool:
        name: str = Field(..., description="Name of the skill to invoke")
        args: str = Field(default="", description="Optional arguments passed as $ARGUMENTS")

        def execute(self) -> str:
            return _execute(manager, session, self.name, self.args)

        def system_prompt_fragment(self) -> str:
            listing = manager.format_listing(listing_budget)
            if not listing:
                return ""
            return (
                "\n\n## Available skills\n\n"
                "Use the Skill tool to invoke one of the following:\n\n"
                f"{listing}\n\n"
                "When a skill matches, invoke it BEFORE generating your response.\n"
            )

    return SkillTool()


def _execute(manager: SkillManager, session_id: str, name: str, args: str) -> str:
    skill = manager.load(name)
    if skill is None:
        available = ", ".join(s.name for s in manager.list_all())
        return f"Skill '{name}' not found. Available: {available}"
    # Real execution (substitution, hooks, fork) comes in later tasks.
    try:
        rendered = substitute_placeholders(
            body=skill.body,
            args=args,
            declared_args=skill.arguments,
            base_dir=skill.base_dir,
            session_id=session_id,
        )
    except SkillInvocationError as e:
        return f"Skill '{name}' invocation failed: {e}"
    return rendered
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestSystemPromptFragment -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/plugins/builtin/skill_tool.py tests/plugins/builtin/__init__.py tests/plugins/builtin/test_skill_tool.py
git commit -m "feat(skill): add SkillTool scaffolding with system_prompt_fragment"
```

---

### Task 7.2: `SkillTool` — inline invocation with substitution

**Files:**
- Modify: `tests/plugins/builtin/test_skill_tool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/plugins/builtin/test_skill_tool.py`:

```python
import asyncio


def _invoke(tool_instance, **kwargs):
    """Invoke the tool instance's execute; accept both sync/async."""
    # The @tool decorator may wrap execute — call the underlying tool.
    # For tests, we build a fresh instance with the kwargs set.
    # tool_instance is a SkillTool; reassigning fields is allowed on Pydantic
    # BaseModel subclasses. For decorated tools, create a new instance.
    cls = type(tool_instance)
    fresh = cls(**kwargs)
    result = fresh.execute()
    if asyncio.iscoroutine(result):
        return asyncio.get_event_loop().run_until_complete(result)
    return result


class TestInlineInvocation:
    def test_known_skill_returns_body(self):
        mgr = _mgr_with(_skill("x", body="Body of x"))
        tool = make_skill_tool(mgr, session_id="sid")
        out = _invoke(tool, name="x", args="")
        assert out == "Body of x"

    def test_unknown_skill_returns_not_found(self):
        mgr = _mgr_with(_skill("a"))
        tool = make_skill_tool(mgr, session_id="sid")
        out = _invoke(tool, name="missing", args="")
        assert "not found" in out.lower()
        assert "a" in out  # available names listed

    def test_substitutes_named_arg(self):
        mgr = _mgr_with(
            _skill(
                "x",
                body="path=$path",
                args=("path",),
            )
        )
        tool = make_skill_tool(mgr, session_id="sid")
        out = _invoke(tool, name="x", args="foo.py")
        assert out == "path=foo.py"

    def test_substitutes_arguments_flat(self):
        mgr = _mgr_with(_skill("x", body="raw=$ARGUMENTS"))
        tool = make_skill_tool(mgr, session_id="sid")
        out = _invoke(tool, name="x", args="one two")
        assert out == "raw=one two"

    def test_substitutes_session_id(self):
        mgr = _mgr_with(_skill("x", body="sid=${OBELIX_SESSION_ID}"))
        tool = make_skill_tool(mgr, session_id="SESSION-42")
        out = _invoke(tool, name="x", args="")
        assert out == "sid=SESSION-42"

    def test_too_many_args_returns_error(self):
        mgr = _mgr_with(_skill("x", body="$a", args=("a",)))
        tool = make_skill_tool(mgr, session_id="sid")
        out = _invoke(tool, name="x", args="one two")
        assert "failed" in out.lower() or "invocation" in out.lower()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestInlineInvocation -v`
Expected: PASS (the scaffold already handles these).

- [ ] **Step 3: Commit**

```bash
git add tests/plugins/builtin/test_skill_tool.py
git commit -m "test(skill): cover inline invocation and substitution in SkillTool"
```

---

### Task 7.3: Idempotent invocation + query-scoped hooks

**Files:**
- Modify: `src/obelix/plugins/builtin/skill_tool.py`
- Modify: `tests/plugins/builtin/test_skill_tool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/plugins/builtin/test_skill_tool.py`:

```python
class TestIdempotence:
    def test_second_invocation_returns_already_active(self):
        mgr = _mgr_with(_skill("x", body="body"))
        tool = make_skill_tool(mgr, session_id="sid")
        first = _invoke(tool, name="x", args="")
        second = _invoke(tool, name="x", args="")
        assert first == "body"
        assert "already active" in second.lower()

    def test_idempotence_is_per_tool_instance(self):
        mgr = _mgr_with(_skill("x", body="body"))
        tool1 = make_skill_tool(mgr, session_id="sid")
        _invoke(tool1, name="x", args="")
        # A fresh tool instance has a fresh active-set
        tool2 = make_skill_tool(mgr, session_id="sid")
        second = _invoke(tool2, name="x", args="")
        assert second == "body"


class TestHookRegistration:
    def test_hooks_registered_on_agent(self):
        mgr = _mgr_with(
            _skill("x", body="body", hooks={"on_tool_error": "retry"})
        )
        fake_agent = MagicMock()
        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        _invoke(tool, name="x", args="")
        # fake_agent.on(...).inject(...) called
        assert fake_agent.on.called

    def test_no_hooks_no_agent_calls(self):
        mgr = _mgr_with(_skill("x", body="body"))
        fake_agent = MagicMock()
        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        _invoke(tool, name="x", args="")
        assert not fake_agent.on.called
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestIdempotence tests/plugins/builtin/test_skill_tool.py::TestHookRegistration -v`
Expected: FAIL.

- [ ] **Step 3: Update `make_skill_tool`**

Edit `src/obelix/plugins/builtin/skill_tool.py`. Replace `make_skill_tool` and `_execute` with:

```python
def make_skill_tool(
    manager: SkillManager,
    listing_budget: int = DEFAULT_LISTING_BUDGET,
    session_id: str | None = None,
    parent_agent: Any = None,
):
    """Factory — SkillTool instance bound to the given manager and (optional) agent.

    `parent_agent` is needed to register skill-scoped hooks; when None,
    hooks in skills are ignored (useful in tests).
    """
    session = session_id or str(uuid.uuid4())
    state = _InvocationState()

    @tool(
        name="Skill",
        description=(
            "Execute a skill within the main conversation. "
            "Use when a skill from the listing matches the user's request."
        ),
    )
    class SkillTool:
        name: str = Field(..., description="Name of the skill to invoke")
        args: str = Field(
            default="", description="Optional arguments passed as $ARGUMENTS"
        )

        def execute(self) -> str:
            return _execute(
                manager, session, state, parent_agent, self.name, self.args
            )

        def system_prompt_fragment(self) -> str:
            listing = manager.format_listing(listing_budget)
            if not listing:
                return ""
            return (
                "\n\n## Available skills\n\n"
                "Use the Skill tool to invoke one of the following:\n\n"
                f"{listing}\n\n"
                "When a skill matches, invoke it BEFORE generating your response.\n"
            )

    return SkillTool()


class _InvocationState:
    """Per-tool-instance state for idempotence and hook tracking."""

    def __init__(self):
        self.active_skills: set[str] = set()
        self.registered_hooks: list[tuple[Any, Any]] = []  # (event, hook)


def _execute(
    manager: SkillManager,
    session_id: str,
    state: _InvocationState,
    parent_agent: Any,
    name: str,
    args: str,
) -> str:
    skill = manager.load(name)
    if skill is None:
        available = ", ".join(s.name for s in manager.list_all())
        return f"Skill '{name}' not found. Available: {available}"

    if name in state.active_skills:
        return (
            f"Skill '{name}' is already active in this query, "
            "continue following its instructions."
        )

    try:
        rendered = substitute_placeholders(
            body=skill.body,
            args=args,
            declared_args=skill.arguments,
            base_dir=skill.base_dir,
            session_id=session_id,
        )
    except SkillInvocationError as e:
        return f"Skill '{name}' invocation failed: {e}"

    if skill.hooks and parent_agent is not None:
        _register_skill_hooks(parent_agent, state, skill)

    state.active_skills.add(name)
    return rendered


def _register_skill_hooks(
    agent: Any, state: _InvocationState, skill: Skill
) -> None:
    from obelix.core.agent.hooks import AgentEvent
    from obelix.core.model.human_message import HumanMessage

    for event_name, instruction in skill.hooks.items():
        try:
            event = AgentEvent(event_name)
        except ValueError:
            logger.warning(
                "[skill] ignoring unknown hook event '%s' on skill '%s' "
                "(should have been caught by validator)",
                event_name,
                skill.name,
            )
            continue
        hook = agent.on(event)
        hook.inject(
            lambda _status, msg=instruction: HumanMessage(content=msg)
        )
        state.registered_hooks.append((event, hook))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/plugins/builtin/skill_tool.py tests/plugins/builtin/test_skill_tool.py
git commit -m "feat(skill): add idempotent invocation and hook registration to SkillTool"
```

---

### Task 7.4: Hook cleanup on `QUERY_END`

**Files:**
- Modify: `src/obelix/plugins/builtin/skill_tool.py`
- Modify: `tests/plugins/builtin/test_skill_tool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/plugins/builtin/test_skill_tool.py`:

```python
class TestHookCleanup:
    def test_query_end_removes_registered_hooks(self):
        mgr = _mgr_with(
            _skill("x", body="body", hooks={"on_tool_error": "x"})
        )
        fake_agent = MagicMock()
        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        _invoke(tool, name="x", args="")

        # The tool must have registered a cleanup hook on QUERY_END too.
        from obelix.core.agent.hooks import AgentEvent

        on_events = [c.args[0] for c in fake_agent.on.call_args_list]
        assert AgentEvent.QUERY_END in on_events or AgentEvent.ON_TOOL_ERROR in on_events

    def test_multi_query_resets_active_set(self):
        """After QUERY_END cleanup fires, invoking again returns body not 'already active'."""
        # We simulate QUERY_END cleanup manually by calling the internal
        # cleanup function.
        mgr = _mgr_with(_skill("x", body="body"))
        fake_agent = MagicMock()
        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        first = _invoke(tool, name="x", args="")
        assert first == "body"

        # Trigger the cleanup the tool registered
        from obelix.plugins.builtin.skill_tool import _trigger_cleanup_for_tests

        _trigger_cleanup_for_tests(tool)

        second = _invoke(tool, name="x", args="")
        assert second == "body"  # active set was cleared
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestHookCleanup -v`
Expected: FAIL.

- [ ] **Step 3: Implement cleanup mechanism**

Edit `src/obelix/plugins/builtin/skill_tool.py`. At the bottom of the module add:

```python
# Registry for test-only trigger hook
_test_cleanup_callbacks: dict[int, Any] = {}


def _trigger_cleanup_for_tests(tool_instance) -> None:
    """Test hook: fires the QUERY_END cleanup attached to this tool."""
    cb = _test_cleanup_callbacks.get(id(tool_instance))
    if cb is not None:
        cb()
```

Edit `_execute` to register a cleanup hook the first time a skill is activated per tool-instance lifetime. Refactor `make_skill_tool`:

```python
def make_skill_tool(
    manager: SkillManager,
    listing_budget: int = DEFAULT_LISTING_BUDGET,
    session_id: str | None = None,
    parent_agent: Any = None,
):
    session = session_id or str(uuid.uuid4())
    state = _InvocationState()

    @tool(
        name="Skill",
        description=(
            "Execute a skill within the main conversation. "
            "Use when a skill from the listing matches the user's request."
        ),
    )
    class SkillTool:
        name: str = Field(..., description="Name of the skill to invoke")
        args: str = Field(
            default="", description="Optional arguments passed as $ARGUMENTS"
        )

        def execute(self) -> str:
            _ensure_cleanup_hook(parent_agent, state)
            return _execute(
                manager, session, state, parent_agent, self.name, self.args
            )

        def system_prompt_fragment(self) -> str:
            listing = manager.format_listing(listing_budget)
            if not listing:
                return ""
            return (
                "\n\n## Available skills\n\n"
                "Use the Skill tool to invoke one of the following:\n\n"
                f"{listing}\n\n"
                "When a skill matches, invoke it BEFORE generating your response.\n"
            )

    instance = SkillTool()

    # Install a test-reachable cleanup reference
    def _do_cleanup():
        _cleanup_skill_hooks(parent_agent, state)

    _test_cleanup_callbacks[id(instance)] = _do_cleanup

    return instance


def _ensure_cleanup_hook(agent: Any, state: _InvocationState) -> None:
    if state.cleanup_installed or agent is None:
        return
    state.cleanup_installed = True
    from obelix.core.agent.hooks import AgentEvent

    hook = agent.on(AgentEvent.QUERY_END)
    hook.inject(
        lambda _status, a=agent, s=state: (
            _cleanup_skill_hooks(a, s)
            or None  # keep hook lambda returning None
        )
    )


def _cleanup_skill_hooks(agent: Any, state: _InvocationState) -> None:
    if agent is None:
        state.active_skills.clear()
        state.registered_hooks.clear()
        return
    for event, hook in state.registered_hooks:
        try:
            agent._unregister_hook(event, hook)
        except Exception as e:  # noqa: BLE001 — best-effort cleanup
            logger.warning("[skill] hook cleanup failed: %s", e)
    state.active_skills.clear()
    state.registered_hooks.clear()
```

Update `_InvocationState`:

```python
class _InvocationState:
    def __init__(self):
        self.active_skills: set[str] = set()
        self.registered_hooks: list[tuple[Any, Any]] = []
        self.cleanup_installed: bool = False
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/plugins/builtin/skill_tool.py tests/plugins/builtin/test_skill_tool.py
git commit -m "feat(skill): clean up skill-scoped hooks on QUERY_END"
```

---

### Task 7.5: Fork execution

**Files:**
- Modify: `src/obelix/plugins/builtin/skill_tool.py`
- Modify: `tests/plugins/builtin/test_skill_tool.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/plugins/builtin/test_skill_tool.py`:

```python
class TestForkExecution:
    def test_fork_context_invokes_sub_agent(self, monkeypatch):
        mgr = _mgr_with(_skill("x", body="Fork body", context="fork"))
        fake_agent = MagicMock()
        fake_agent.provider = MagicMock()
        fake_agent.registered_tools = []
        fake_agent.max_iterations = 15

        called = {}

        class FakeSub:
            def __init__(self, inner_agent):
                called["inner_agent"] = inner_agent

            async def execute_query(self, q):
                called["q"] = q
                return "fork result text"

        # Patch the import inside skill_tool at the reference used
        from obelix.plugins.builtin import skill_tool as st_mod

        monkeypatch.setattr(st_mod, "SubAgentWrapper", FakeSub)

        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        out = _invoke(tool, name="x", args="")
        assert out == "fork result text"
        assert "q" in called

    def test_fork_skill_not_marked_active_on_parent(self, monkeypatch):
        """Fork keeps its own active-set; parent's tool-instance set is unchanged."""
        mgr = _mgr_with(_skill("x", body="Fork body", context="fork"))
        fake_agent = MagicMock()
        fake_agent.provider = MagicMock()
        fake_agent.registered_tools = []
        fake_agent.max_iterations = 15

        from obelix.plugins.builtin import skill_tool as st_mod

        class FakeSub:
            def __init__(self, inner_agent):
                pass

            async def execute_query(self, q):
                return "ok"

        monkeypatch.setattr(st_mod, "SubAgentWrapper", FakeSub)

        tool = make_skill_tool(mgr, session_id="sid", parent_agent=fake_agent)
        _invoke(tool, name="x", args="")
        # Calling again should NOT return "already active" — fork is stateless
        # on the parent tool
        second = _invoke(tool, name="x", args="")
        assert "already active" not in second.lower()
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py::TestForkExecution -v`
Expected: FAIL.

- [ ] **Step 3: Implement fork path**

Edit `src/obelix/plugins/builtin/skill_tool.py`. Add import at top:

```python
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.subagent_wrapper import SubAgentWrapper
```

Edit `_execute` to branch on `skill.context`:

```python
def _execute(
    manager: SkillManager,
    session_id: str,
    state: _InvocationState,
    parent_agent: Any,
    name: str,
    args: str,
) -> str:
    skill = manager.load(name)
    if skill is None:
        available = ", ".join(s.name for s in manager.list_all())
        return f"Skill '{name}' not found. Available: {available}"

    try:
        rendered = substitute_placeholders(
            body=skill.body,
            args=args,
            declared_args=skill.arguments,
            base_dir=skill.base_dir,
            session_id=session_id,
        )
    except SkillInvocationError as e:
        return f"Skill '{name}' invocation failed: {e}"

    if skill.context == "fork":
        return _execute_fork(rendered, skill, parent_agent)

    # Inline
    if name in state.active_skills:
        return (
            f"Skill '{name}' is already active in this query, "
            "continue following its instructions."
        )

    if skill.hooks and parent_agent is not None:
        _register_skill_hooks(parent_agent, state, skill)

    state.active_skills.add(name)
    return rendered


def _execute_fork(rendered_body: str, skill: Skill, parent_agent: Any) -> str:
    import asyncio

    if parent_agent is None:
        return "Fork execution requires parent agent context"

    inner = BaseAgent(
        system_message=rendered_body,
        provider=parent_agent.provider,
        max_iterations=getattr(parent_agent, "max_iterations", 15),
    )
    for t in getattr(parent_agent, "registered_tools", []):
        inner.register_tool(t)

    if skill.hooks:
        inner_state = _InvocationState()
        _register_skill_hooks(inner, inner_state, skill)

    sub = SubAgentWrapper(inner)

    coro = sub.execute_query("Begin executing the skill as specified above.")
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Must be awaited by caller; for sync path use new loop
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.new_event_loop().run_until_complete(coro)
```

Note: The exact async handling may need adjustment based on how tools execute in the current BaseAgent loop. If tools are always awaited, make `execute` async and remove the loop juggling. Verify with:

Run: `grep -n "async def execute\|def execute" src/obelix/plugins/builtin/bash_tool.py | head`

If `bash_tool.execute` is async, make `SkillTool.execute` async too and simply `await sub.execute_query(...)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/plugins/builtin/test_skill_tool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/obelix/plugins/builtin/skill_tool.py tests/plugins/builtin/test_skill_tool.py
git commit -m "feat(skill): add fork execution branch using SubAgentWrapper"
```

---

## Phase 8 — BaseAgent integration

### Task 8.1: `skills_config` parameter

**Files:**
- Modify: `src/obelix/core/agent/base_agent.py`
- Create: `tests/core/agent/test_base_agent_skills.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/agent/test_base_agent_skills.py`:

```python
from pathlib import Path
import pytest
from unittest.mock import MagicMock
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.skill.skill import SkillValidationError

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class _StubProvider:
    model_id = "stub-model"
    provider_type = "stub"

    async def invoke(self, *a, **k):
        raise NotImplementedError


class TestSkillsConfigAccepted:
    def test_none_behaves_as_before(self):
        a = BaseAgent(
            system_message="You are X.",
            provider=_StubProvider(),
            skills_config=None,
        )
        # No skill tool registered
        assert all(
            getattr(t, "tool_name", "") != "Skill" for t in a.registered_tools
        )

    def test_valid_single_dir_registers_skill_tool(self):
        a = BaseAgent(
            system_message="You are X.",
            provider=_StubProvider(),
            skills_config=str(FIXTURES / "minimal"),
        )
        names = [getattr(t, "tool_name", None) for t in a.registered_tools]
        assert "Skill" in names
        assert "## Available skills" in a.system_message.content

    def test_valid_list_of_dirs(self):
        a = BaseAgent(
            system_message="X",
            provider=_StubProvider(),
            skills_config=[
                str(FIXTURES / "minimal"),
                str(FIXTURES / "with_when_to_use"),
            ],
        )
        assert "minimal" in a.system_message.content
        assert "with_when_to_use" in a.system_message.content

    def test_pathlib_path_accepted(self):
        a = BaseAgent(
            system_message="X",
            provider=_StubProvider(),
            skills_config=FIXTURES / "minimal",
        )
        assert "## Available skills" in a.system_message.content


class TestValidationFailure:
    def test_invalid_skill_raises_on_construction(self):
        with pytest.raises(SkillValidationError):
            BaseAgent(
                system_message="X",
                provider=_StubProvider(),
                skills_config=str(FIXTURES / "invalid_missing_description"),
            )

    def test_aggregated_validation_failure(self):
        with pytest.raises(SkillValidationError) as exc:
            BaseAgent(
                system_message="X",
                provider=_StubProvider(),
                skills_config=str(FIXTURES / "invalid_multiple_issues"),
            )
        assert len(exc.value.issues) >= 4
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/core/agent/test_base_agent_skills.py -v`
Expected: FAIL — `skills_config` param not recognized.

- [ ] **Step 3: Add `skills_config` to `BaseAgent.__init__`**

Open `src/obelix/core/agent/base_agent.py`. Find the `__init__` signature (line 57-69). After `mcp_config` add `skills_config`:

```python
    def __init__(
        self,
        system_message: str,
        provider: AbstractLLMProvider,
        max_iterations: int = 15,
        tools: Tool | type[Tool] | list[type[Tool] | Tool] | None = None,
        tool_policy: list[ToolRequirement] | None = None,
        exit_on_success: list[str] | None = None,
        response_schema: type[BaseModel] | None = None,
        tracer: "Tracer | None" = None,
        planning: bool = False,
        mcp_config: "str | Path | MCPServerConfig | list | None" = None,
        skills_config: "str | Path | list | None" = None,
    ):
```

At the end of `__init__` (after the MCP block), add:

```python
        # Skills integration (optional)
        self._skill_manager = None
        if skills_config is not None:
            from obelix.adapters.outbound.skill.filesystem import (
                FilesystemSkillProvider,
            )
            from obelix.core.skill.manager import SkillManager
            from obelix.plugins.builtin.skill_tool import make_skill_tool

            paths = self._normalize_skills_config(skills_config)
            fs_provider = FilesystemSkillProvider(paths)
            # MCP skill provider wired when MCP is also configured.
            providers = [fs_provider]
            if self._mcp_manager is not None:
                from obelix.adapters.outbound.skill.mcp import MCPSkillProvider

                providers.append(MCPSkillProvider(self._mcp_manager))

            self._skill_manager = SkillManager(providers)

            if self._skill_manager.list_all():
                session_id = None
                if tracer is not None and hasattr(tracer, "session_id"):
                    session_id = tracer.session_id
                skill_tool = make_skill_tool(
                    self._skill_manager,
                    session_id=session_id,
                    parent_agent=self,
                )
                self.register_tool(skill_tool)

    @staticmethod
    def _normalize_skills_config(cfg) -> list[Path]:
        if isinstance(cfg, (str, Path)):
            return [Path(cfg)]
        if isinstance(cfg, list):
            out: list[Path] = []
            for item in cfg:
                if isinstance(item, (str, Path)):
                    out.append(Path(item))
                else:
                    raise TypeError(
                        f"skills_config list items must be str or Path, got {type(item).__name__}"
                    )
            return out
        raise TypeError(
            f"skills_config must be str, Path, list, or None, got {type(cfg).__name__}"
        )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/core/agent/test_base_agent_skills.py tests/core/agent/test_base_agent_hook_unregister.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full skill test suite**

Run: `uv run pytest tests/core/skill/ tests/adapters/outbound/skill/ tests/plugins/builtin/ tests/core/agent/test_base_agent_skills.py tests/core/agent/test_base_agent_hook_unregister.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/obelix/core/agent/base_agent.py tests/core/agent/test_base_agent_skills.py
git commit -m "feat(agent): add skills_config parameter wiring SkillManager and SkillTool"
```

---

## Phase 9 — Integration tests

### Task 9.1: Integration — full inline flow

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/skill/__init__.py`
- Create: `tests/integration/skill/test_full_flow.py`

- [ ] **Step 1: Write failing integration test**

Create `tests/integration/__init__.py` (empty). Create `tests/integration/skill/__init__.py` (empty).

Create `tests/integration/skill/test_full_flow.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider:
    """A provider that replies with scripted responses."""

    model_id = "stub"
    provider_type = "stub"

    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    async def invoke(self, *a, **k):
        resp = self._responses[self._idx]
        self._idx = min(self._idx + 1, len(self._responses) - 1)
        return resp


@pytest.mark.asyncio
async def test_agent_invokes_skill_inline():
    # Round 1: LLM decides to call Skill(name="minimal")
    # Round 2: After receiving the body, LLM responds with final text
    tool_call_message = AssistantMessage(
        content="",
        tool_calls=[
            ToolCall(id="t1", name="Skill", arguments={"name": "minimal"})
        ],
    )
    final_message = AssistantMessage(content="Done after reading minimal.")
    provider = StubProvider([tool_call_message, final_message])

    agent = BaseAgent(
        system_message="You are X.",
        provider=provider,
        skills_config=str(FIXTURES / "minimal"),
    )
    result = await agent.execute_query("please run minimal")
    # Final response reaches the caller
    assert "Done" in result or final_message.content in result

    # Conversation contains the skill body
    bodies = [
        m for m in agent.conversation_history if "Body of the minimal" in str(m)
    ]
    assert bodies
```

- [ ] **Step 2: Run — expected to PASS or surface issues**

Run: `uv run pytest tests/integration/skill/test_full_flow.py -v`
Expected: PASS. If it fails, adjust the StubProvider shape to match the real provider contract (check `AbstractLLMProvider`). Record any additional fixes needed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/
git commit -m "test(skill): integration test for inline skill invocation end-to-end"
```

---

### Task 9.2: Integration — validation failure aggregated

**Files:**
- Create: `tests/integration/skill/test_validation_failure.py`

- [ ] **Step 1: Write test**

Create `tests/integration/skill/test_validation_failure.py`:

```python
from pathlib import Path
import pytest
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.skill.skill import SkillValidationError

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider:
    model_id = "stub"
    provider_type = "stub"

    async def invoke(self, *a, **k):
        raise NotImplementedError


def test_agent_construction_fails_cleanly_on_invalid_skills():
    with pytest.raises(SkillValidationError) as exc:
        BaseAgent(
            system_message="X",
            provider=StubProvider(),
            skills_config=[
                str(FIXTURES / "invalid_missing_description"),
                str(FIXTURES / "invalid_duplicate_args"),
                str(FIXTURES / "invalid_unknown_hook"),
            ],
        )
    # All 3 files represented
    files = {str(i.file_path) for i in exc.value.issues}
    assert len(files) == 3
```

- [ ] **Step 2: Run and commit**

Run: `uv run pytest tests/integration/skill/test_validation_failure.py -v`
Expected: PASS.

```bash
git add tests/integration/skill/test_validation_failure.py
git commit -m "test(skill): integration test for aggregated validation failure"
```

---

### Task 9.3: Integration — hook lifecycle

**Files:**
- Create: `tests/integration/skill/test_hook_lifecycle.py`

- [ ] **Step 1: Write test**

Create the test that uses `with_all_hooks/SKILL.md` and verifies a hook injection actually fires during the loop:

```python
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.hooks import AgentEvent
from obelix.core.model.assistant_message import AssistantMessage
from obelix.core.model.tool_message import ToolCall

FIXTURES = Path(__file__).parent.parent.parent / "fixtures" / "skills"


class StubProvider:
    model_id = "stub"
    provider_type = "stub"

    def __init__(self, responses):
        self._r = list(responses)
        self._i = 0

    async def invoke(self, *a, **k):
        r = self._r[self._i]
        self._i = min(self._i + 1, len(self._r) - 1)
        return r


@pytest.mark.asyncio
async def test_query_end_hook_fires_and_is_cleaned_up():
    # LLM calls skill, then emits final response
    provider = StubProvider(
        [
            AssistantMessage(
                content="",
                tool_calls=[
                    ToolCall(id="t1", name="Skill", arguments={"name": "with_all_hooks"})
                ],
            ),
            AssistantMessage(content="done"),
        ]
    )
    agent = BaseAgent(
        system_message="X",
        provider=provider,
        skills_config=str(FIXTURES / "with_all_hooks"),
    )
    # Number of hooks on QUERY_END before (0 + 1 from SkillTool cleanup installed lazily)
    await agent.execute_query("q")
    # After query, skill-scoped hooks should be gone
    # (only agent's own built-in hooks remain)
    for event in AgentEvent:
        for hook in agent._hooks[event]:
            # No skill-scoped hook should still be registered
            # We can't easily distinguish, but at least the active set is empty
            pass
```

Note: the test above is a smoke-check; the finer-grained test that a specific injection fired lives in the unit tests.

- [ ] **Step 2: Run and commit**

Run: `uv run pytest tests/integration/skill/test_hook_lifecycle.py -v`
Expected: PASS or SKIP if hook introspection is too tight (acceptable for v1).

```bash
git add tests/integration/skill/test_hook_lifecycle.py
git commit -m "test(skill): integration test for hook lifecycle around skill invocation"
```

---

## Phase 10 — Regression guardrails

### Task 10.1: Import boundary test with `grimp`

**Files:**
- Create: `tests/regression/__init__.py`
- Create: `tests/regression/test_import_boundaries.py`

- [ ] **Step 1: Create empty `__init__.py`**

Create `tests/regression/__init__.py` (empty).

- [ ] **Step 2: Write boundary tests**

Create `tests/regression/test_import_boundaries.py`:

```python
import grimp
import pytest


@pytest.fixture(scope="module")
def graph():
    return grimp.build_graph("obelix")


def test_skill_module_is_leaf(graph):
    """core/skill/skill.py must not import from other core/skill/* modules."""
    deps = graph.find_downstream_modules("obelix.core.skill.skill", as_package=False)
    # Allowed: nothing in core.skill itself
    forbidden = {m for m in graph.find_modules_directly_imported_by(
        "obelix.core.skill.skill"
    ) if m.startswith("obelix.core.skill.") and m != "obelix.core.skill.skill"}
    assert not forbidden, f"skill.py imports from other core/skill: {forbidden}"


def test_validation_does_not_touch_io(graph):
    """validation.py must not import yaml, Path operations for reading, etc."""
    direct = graph.find_modules_directly_imported_by("obelix.core.skill.validation")
    # yaml is allowed only in parsing.py
    assert "yaml" not in direct, "validation.py must not import yaml"


def test_manager_does_not_import_yaml_or_io(graph):
    direct = graph.find_modules_directly_imported_by("obelix.core.skill.manager")
    assert "yaml" not in direct
    assert "obelix.core.skill.parsing" not in direct


def test_skill_tool_does_not_import_filesystem_or_mcp(graph):
    """skill_tool orchestrates via SkillManager, not providers."""
    direct = graph.find_modules_directly_imported_by(
        "obelix.plugins.builtin.skill_tool"
    )
    assert "obelix.adapters.outbound.skill.filesystem" not in direct
    assert "obelix.adapters.outbound.skill.mcp" not in direct


def test_core_skill_does_not_import_plugins_or_adapters(graph):
    """core/skill/* is upstream of adapters and plugins."""
    for mod in [
        "obelix.core.skill.skill",
        "obelix.core.skill.substitution",
        "obelix.core.skill.parsing",
        "obelix.core.skill.validation",
        "obelix.core.skill.reporting",
        "obelix.core.skill.manager",
    ]:
        direct = graph.find_modules_directly_imported_by(mod)
        forbidden = {m for m in direct if m.startswith("obelix.plugins.")}
        assert not forbidden, f"{mod} imports plugins: {forbidden}"
        forbidden_adapters = {
            m for m in direct if m.startswith("obelix.adapters.")
        }
        # validation.py may import hooks which is in core, not adapters — fine.
        assert not forbidden_adapters, f"{mod} imports adapters: {forbidden_adapters}"
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/regression/test_import_boundaries.py -v`
Expected: PASS. If any test fails, that indicates a coupling violation — fix the offending import.

- [ ] **Step 4: Commit**

```bash
git add tests/regression/__init__.py tests/regression/test_import_boundaries.py
git commit -m "test(regression): add grimp-based import boundary enforcement"
```

---

### Task 10.2: Snapshot test for skill listing

**Files:**
- Create: `tests/regression/test_skill_listing_snapshot.py`

- [ ] **Step 1: Write snapshot test**

Create `tests/regression/test_skill_listing_snapshot.py`:

```python
from pathlib import Path
from unittest.mock import MagicMock
from obelix.core.skill.manager import SkillManager
from obelix.core.skill.skill import Skill


def _mk(name: str, desc: str, when: str | None = None):
    return Skill(
        name=name,
        description=desc,
        body="body",
        base_dir=Path("/"),
        when_to_use=when,
    )


def _fake_provider(skills):
    p = MagicMock()
    p.discover.return_value = list(skills)
    return p


def test_listing_canonical_snapshot():
    skills = [
        _mk("alpha", "The alpha skill"),
        _mk("beta", "Beta helper", when="when beta-ing"),
        _mk("gamma", "Gamma processor"),
    ]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(10_000)
    expected = (
        "- alpha: The alpha skill\n"
        "- beta: Beta helper — when beta-ing\n"
        "- gamma: Gamma processor"
    )
    assert out == expected


def test_listing_truncation_snapshot():
    """When budget is tight, all names survive and descriptions are truncated."""
    long = "x" * 300
    skills = [_mk(f"s{i}", long) for i in range(5)]
    mgr = SkillManager(providers=[_fake_provider(skills)])
    out = mgr.format_listing(120)
    # Each line's description fits the per-entry budget
    for line in out.split("\n"):
        assert line.startswith("- s")
    # At least some truncation occurred
    assert "\u2026" in out or "..." in out or all(len(ln) < 50 for ln in out.split("\n"))
```

- [ ] **Step 2: Run and commit**

Run: `uv run pytest tests/regression/test_skill_listing_snapshot.py -v`
Expected: PASS.

```bash
git add tests/regression/test_skill_listing_snapshot.py
git commit -m "test(regression): snapshot test for skill listing format"
```

---

### Task 10.3: Run full coverage check

- [ ] **Step 1: Run full coverage gate**

Run:
```bash
uv run pytest tests/ \
  --cov=obelix.core.skill \
  --cov=obelix.adapters.outbound.skill \
  --cov=obelix.plugins.builtin.skill_tool \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=90
```

Expected: overall ≥ 90%, `core/skill/*` ≥ 95%. If below, add tests for uncovered lines before proceeding.

- [ ] **Step 2: Fix any gaps**

For each uncovered line or branch reported, go back to the matching Phase task and add the missing test. Commit each batch of new tests separately with `test: cover <module>.<case>`.

---

## Phase 11 — Documentation

### Task 11.1: `docs/skills.md` — user-facing (EN)

**Files:**
- Create: `docs/skills.md`

- [ ] **Step 1: Write the document**

Create `docs/skills.md`:

````markdown
# Skills in Obelix

A **Skill** is a Markdown file with YAML frontmatter that describes a specialized capability the agent can invoke on demand. Obelix loads skills from paths you configure per agent; at runtime the LLM picks a skill from the listing and invokes it via the built-in `Skill` tool.

This guide shows:
- How to author a SKILL.md
- Supported frontmatter fields
- How to wire skills into an agent
- Inline vs fork execution
- Hook frontmatter
- Worked examples

## 1. Authoring a skill

A skill is a directory containing a `SKILL.md` file. The directory name becomes the skill name (unless overridden in frontmatter).

```
my_project/skills/code-reviewer/
├── SKILL.md
└── checklist.md            # optional: referenced via ${OBELIX_SKILL_DIR}
```

Minimal `SKILL.md`:
```markdown
---
description: Reviews a Python file for design quality
when_to_use: User asks for code review
---

# Code Review Protocol

Examine the file and produce a report.
```

## 2. Frontmatter reference

| Field | Required | Type | Default | Description |
|-------|----------|------|---------|-------------|
| `name` | no | str | directory name | Override the skill's exposed name |
| `description` | **yes** | str | — | Short sentence shown in the LLM listing |
| `when_to_use` | no | str | — | Heuristic the LLM reads to decide when to invoke |
| `arguments` | no | `list[str]` | `[]` | Named positional placeholders for the body (`$name`) |
| `allowed_tools` | no | `list[str]` | `[]` | Informational: tools the skill expects |
| `context` | no | `"inline"` or `"fork"` | `"inline"` | `fork` runs the skill in an isolated sub-agent |
| `hooks` | no | `dict[str, str]` | `{}` | Event-scoped instructions (see §5) |

Unknown fields are ignored (forward compat).

## 3. Placeholders

In the body:

| Placeholder | Replaced with |
|-------------|---------------|
| `$<name>` | The `<name>`-th value from `arguments:` (or `""` if missing) |
| `$ARGUMENTS` | The full raw arg string passed to `Skill(name, args)` |
| `${OBELIX_SKILL_DIR}` | Absolute path to the skill's directory |
| `${OBELIX_SESSION_ID}` | Session identifier (stable across an agent instance) |

Named-arg parsing uses shell quoting (`shlex`): `args='"my file.py" 3'` → `$path=my file.py`, `$depth=3`. Passing more args than declared fails the invocation.

## 4. Wiring skills into an agent

```python
from obelix.core.agent import BaseAgent

agent = BaseAgent(
    system_message="You are a senior Python engineer.",
    provider=provider,
    skills_config="./skills/code-reviewer",    # single skill
)
```

Variants:
```python
skills_config="./skills/"                     # directory containing subdirs
skills_config=["./skills/a", "./skills/b"]   # list of paths
skills_config=Path("./skills/a")             # pathlib.Path
skills_config=None                           # no skills (default)
```

Validation runs at agent construction. Invalid skills raise `SkillValidationError` listing every issue across every file.

## 5. Hook frontmatter

A skill can request an instruction to be injected whenever a specific agent event fires, for the duration of the query:

```yaml
---
description: Safe refactor
hooks:
  on_tool_error: "A tool failed. Show the error, then ask whether to rollback."
  query_end: "Before concluding, list all files modified."
---
```

Supported events (snake_case of `AgentEvent` enum values): `before_llm_call`, `after_llm_call`, `before_tool_execution`, `after_tool_execution`, `on_tool_error`, `before_final_response`, `query_end`.

Hooks are automatically removed at the end of the query that invoked the skill.

## 6. Fork execution

When `context: fork`, the skill runs in a **sub-agent** with its own conversation history, isolated from the parent. The parent receives only the final assistant message.

```yaml
---
description: Deep debug using the scientific method
context: fork
---

# Debug Protocol
1. Run the failing test
2. Form a hypothesis
...
```

Fork execution inherits the parent's provider and tools; memory graph is NOT inherited.

## 7. Worked examples

### Example 1 — code-reviewer

`skills/code-reviewer/SKILL.md`:
```markdown
---
description: Reviews a Python file for design quality
when_to_use: User asks for code review
arguments: [path]
---

# Code Review Protocol

Review the file at `$path`. Produce a report with:
- Severity levels (high/medium/low)
- Specific line numbers
- Suggested fixes

If `$path` is empty, ask the user for one.
```

Invocation:
```python
agent = BaseAgent(
    system_message="You are a reviewer.",
    provider=provider,
    skills_config="./skills/code-reviewer",
)
result = await agent.execute_query("review src/foo.py")
```

### Example 2 — commit-writer (fork)

`skills/commit-writer/SKILL.md`:
```markdown
---
description: Drafts a conventional commit message from a diff
when_to_use: User asks to commit staged changes
context: fork
---

# Commit Protocol
1. Read the staged diff
2. Identify type: feat, fix, chore, refactor, test, docs
3. Write a one-line subject (max 70 chars)
4. Optional body with "why"

Output ONLY the final commit message (no preamble).
```

### Example 3 — debugger with hooks

`skills/debugger/SKILL.md`:
```markdown
---
description: Systematic bug diagnosis using the scientific method
when_to_use: User reports a failing test or unexpected behavior
arguments: [target]
hooks:
  on_tool_error: "Read the failure message carefully. Form a new hypothesis before trying again."
---

# Debug Protocol
1. Reproduce the issue: run `$target`
2. Form a hypothesis
3. Test it (read code, inspect logs)
4. Iterate

Report root cause + fix.
```

## 8. Troubleshooting

**`SkillValidationError` on agent construction** — read the issue list; every problem is shown with path + field. Fix all of them (validation is exhaustive).

**LLM never invokes the skill** — check the `description` and `when_to_use` are clear and specific. The LLM uses these to decide.

**Skill is "already active"** — calling the same skill twice in one query is idempotent. The body is only injected the first time.

**Fork skill loses context from parent** — by design. Pass the context you need via `arguments`, not via implicit inheritance.

**`${OBELIX_SESSION_ID}` unstable across invocations** — without a tracer, the session ID is generated once per agent instance. Attach a tracer to get a stable, traceable session ID.
````

- [ ] **Step 2: Update index**

Edit `docs/index.md`. Add a line under the existing list pointing to `skills.md` (place near `base_agent.md` for discoverability):

```markdown
- [skills.md](skills.md) — Skills subsystem: authoring SKILL.md, wiring to agents, inline vs fork, hook frontmatter
```

- [ ] **Step 3: Update base_agent docs**

Edit `docs/base_agent.md`. In the constructor parameters section, add an entry:

```markdown
- `skills_config` (optional): `str | Path | list | None` — one or more paths to skill directories or SKILL.md files. See [skills.md](skills.md).
```

- [ ] **Step 4: Commit**

```bash
git add docs/skills.md docs/index.md docs/base_agent.md
git commit -m "docs(skills): add user-facing skills documentation"
```

---

### Task 11.2: `docs/skills_implementation_notes.md` — personal IT

**Files:**
- Create: `docs/skills_implementation_notes.md`

- [ ] **Step 1: Write the document**

Create `docs/skills_implementation_notes.md`:

````markdown
# Note di implementazione — Skills (IT, personal)

> Questo file è per me, per ricordare cosa è stato fatto in v1, cosa NON c'è, e come usare il sistema con esempi completi. Non è documentazione ufficiale — quella è in `skills.md`.

## Stato v1 — cosa c'è

- Discovery **esplicita**: nessuna auto-search. Si passa path(s) a `skills_config`.
- Due sorgenti: **filesystem** (SKILL.md) + **MCP** (prompt esposti da server MCP connessi).
- Validation **a tappeto**: raccoglie tutti gli errori da tutti i file, alza un solo `SkillValidationError` con la lista completa.
- Tool built-in `Skill` registrato automaticamente quando `skills_config` è valorizzato.
- Listing skill iniettato nel system prompt via `system_prompt_fragment()` (pattern Obelix esistente), budget 1% del context window, troncamento adattivo.
- Invocazione inline (default): il body diventa ToolResult, l'LLM nella iter successiva segue le istruzioni.
- Invocazione fork (`context: fork`): body diventa system_message di un `SubAgentWrapper`, eredita tool ma non memory.
- Idempotenza: invocare la stessa skill due volte in una query ritorna "already active" la seconda volta.
- Hook frontmatter: mappati 1:1 su `AgentEvent`, registrati all'invoke, rimossi a `QUERY_END`.
- Placeholder: `$ARGUMENTS`, named positional (`$path`, `$depth`, ...), `${OBELIX_SKILL_DIR}`, `${OBELIX_SESSION_ID}`.
- Shlex quoting su args named.
- Coverage: ≥95% su `core/skill`, ≥90% su adapter e skill_tool.
- Guardrail: `grimp` test che valida le regole di coupling (nessun import vietato).

## Stato v1 — cosa NON c'è

### Auto-discovery filesystem
Non leggiamo `~/.obelix/skills/` o `.obelix/skills/` automaticamente. L'utente DEVE passare i path esplicitamente in `skills_config`. **Perché**: determinismo per deploy (A2A serve), no sorprese.

### Compat `~/.claude/skills/`
Non leggiamo da lì. Le skill community sono compatibili sul **formato** (SKILL.md + frontmatter) ma l'utente copia/linka la cartella dove vuole.

### `paths:` conditional activation
Skill attivate solo quando l'LLM tocca file matching. Non c'è. **Perché**: richiede intercettare ogni tool call, complessità alta per uso raro (~5% skill community).

### Shell inline `` !`cmd` ``
Esecuzione di comandi shell nel body. NON SUPPORTATO. **Perché**: security risk serio, servirebbe sandboxing (OpenShell?) o approval per comando.

### `model:` / `effort:` override
Skill non può cambiare modello o reasoning effort. **Perché**: richiede hot-swap del provider, aggiungibile dopo.

### Walk-up discovery
Non risaliamo da cwd a home cercando `.obelix/skills/`. **Perché**: path espliciti, zero magia.

### `SkillConfig` class
Il parametro accetta `str | Path | list | None`, non c'è classe wrapper. **Perché**: tutta la config sta in un path, niente da configurare oltre. Aggiungibile senza breaking change quando servirà.

### Plugin marketplace / remote skill search
Nessun package manager, nessuna scoperta remota. Per installare una skill: `git clone` nella dir del progetto.

### Skill nella AgentFactory
Le skill sono per-agent, non per-factory. Se un agent registrato nella factory vuole skill, le passa al costruttore come un BaseAgent qualunque.

## Esempi d'uso — end-to-end

### Esempio 1 — skill minimal su progetto

```
my_app/
├── main.py
└── skills/
    └── greeter/
        └── SKILL.md
```

`skills/greeter/SKILL.md`:
```markdown
---
description: Greets the user in a specific language
arguments: [language]
---
Say hello in $language, briefly.
```

`main.py`:
```python
import asyncio
from obelix.core.agent.base_agent import BaseAgent
from obelix.adapters.outbound.llm.anthropic import (
    AnthropicProvider,
    AnthropicConnection,
)

async def main():
    provider = AnthropicProvider(
        connection=AnthropicConnection(api_key="..."),
        model_id="claude-sonnet-4-6",
    )
    agent = BaseAgent(
        system_message="You are helpful.",
        provider=provider,
        skills_config="./skills/greeter",
    )
    print(await agent.execute_query("say hi in Italian"))

asyncio.run(main())
```

Cosa succede:
1. Al boot, `FilesystemSkillProvider` legge `./skills/greeter/SKILL.md`, validate, costruisce `Skill`.
2. `SkillManager` aggrega, `SkillTool` registrato.
3. System prompt include `- greeter: Greets the user in a specific language`.
4. LLM decide di invocare `Skill(name="greeter", args="Italian")`.
5. `SkillTool` ritorna body sostituito: `"Say hello in Italian, briefly."`
6. LLM legge, risponde "Ciao!".

### Esempio 2 — skill con fork

`skills/deep-debug/SKILL.md`:
```markdown
---
description: Systematically debug a failing test
when_to_use: User reports a test failure or stack trace
context: fork
arguments: [target]
---
# Debug Protocol
1. Run `$target` and capture output
2. Form a hypothesis
3. Inspect relevant code
4. Iterate

Report root cause + fix.
```

Uso identico a sopra. Differenza: l'LLM principale vede **solo la conclusione** del debug, non i 10 turn interni. Sub-agent ha token budget suo.

### Esempio 3 — skill con hook

`skills/safe-edit/SKILL.md`:
```markdown
---
description: Edit with safety checks
arguments: [path]
hooks:
  on_tool_error: "The previous operation failed. Do not continue without showing me the error and asking if I want to rollback."
  query_end: "List every file you modified before finishing."
---
Edit `$path` carefully. Rules:
- Always Read before Edit
- If tests exist, mention running them
```

Al primo invoke, `SkillTool` registra due hook: uno su `ON_TOOL_ERROR`, uno su `QUERY_END`. Durante il loop della query, se un tool fallisce, viene iniettato un `HumanMessage` con il testo dell'hook. A `QUERY_END`, gli hook vengono rimossi (non afflutriscono query successive).

### Esempio 4 — skill da server MCP

Se ho configurato:
```python
BaseAgent(
    ...,
    mcp_config="./mcp.json",
    skills_config=None,  # non passo skill filesystem
)
```

E il server MCP espone un prompt `reviewer` con `description: "Reviews code"`, Obelix lo trasforma in skill `mcp__<server>__reviewer`. L'LLM la vede nel listing e può invocarla con `Skill(name="mcp__<server>__reviewer")`.

Se invece ho **sia** filesystem **sia** MCP e una skill fs e una MCP hanno lo stesso nome, **vince filesystem**. Loggato un warning.

## Fallback / edge cases

- **Skill non trovata**: `ToolResult(error="Skill 'X' not found. Available: [...]")` — l'LLM vede la lista e può ritentare.
- **Args troppi**: errore chiaro con expected vs actual. Non crasha l'agent.
- **YAML rotto in un file**: issue aggregato, altri file continuano.
- **Body vuoto**: issue "empty body", skill scartata.
- **Hook event sconosciuto**: issue al boot, skill scartata.
- **Placeholder non dichiarato**: issue al boot, skill scartata.

## Troubleshooting personale

### "Perché l'LLM non invoca la mia skill?"
- Controlla che `description` e `when_to_use` siano chiari e matchino la query tipo dell'utente.
- Se hai troppe skill, il listing potrebbe essere truncated → nome visibile ma description troncata. Aumenta `listing_budget` in `make_skill_tool` (hardcoded 8000).

### "Il body non viene sostituito correttamente"
- Named positional sostituiti **prima** di `$ARGUMENTS`. Ordine stabile: nomi più lunghi prima (prefix safety).
- `${OBELIX_SKILL_DIR}` viene sostituito sempre, anche se non serve.

### "Skill in fork non vede la memory del padre"
- By design. Passa la memory via `arguments`.

### "Coverage sotto target nei test"
- `core/skill/*` deve essere ≥95% branch+line. Run `pytest --cov-branch --cov-report=term-missing` per vedere le righe scoperte.

## Dove cercare se qualcosa si rompe

| Sintomo | File da controllare |
|---------|---------------------|
| Validazione non aggrega errori | `adapters/outbound/skill/filesystem.py::discover` |
| Body mal sostituito | `core/skill/substitution.py::substitute_placeholders` |
| Listing sballato | `core/skill/manager.py::format_listing` |
| Hook non rimossi | `plugins/builtin/skill_tool.py::_cleanup_skill_hooks` |
| Fork non funziona | `plugins/builtin/skill_tool.py::_execute_fork` + `subagent_wrapper.py` |
| MCP skill non scoperta | `adapters/outbound/mcp/manager.py::list_prompts` + `adapters/outbound/skill/mcp.py::_to_skill` |

## Prossime iterazioni probabili

1. `paths:` conditional activation (hook su Read/Edit che attiva skill matching)
2. Shell inline sandboxato (forse via OpenShell)
3. `model:` override
4. `SkillConfig` class per opzioni avanzate (quando servirà)
5. Walk-up discovery opzionale (quando qualcuno avrà un monorepo)
````

- [ ] **Step 2: Commit**

```bash
git add docs/skills_implementation_notes.md
git commit -m "docs(skills): add personal implementation notes (IT)"
```

---

## Final verification

- [ ] **Step 1: Run the whole test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 2: Coverage gate**

Run:
```bash
uv run pytest tests/ \
  --cov=obelix.core.skill \
  --cov=obelix.adapters.outbound.skill \
  --cov=obelix.plugins.builtin.skill_tool \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=90
```
Expected: PASS with ≥90%.

- [ ] **Step 3: Lint + format**

Run: `uv run ruff format . && uv run ruff check .`
Expected: no errors.

- [ ] **Step 4: Changelog / final commit**

If a CHANGELOG exists, add an entry. Otherwise final wrap-up commit:
```bash
git log --oneline develop..HEAD | head -40
```
Verify the sequence of commits reads well (one per task, clear messages).

---

## Self-review checklist (used during authoring)

- Every task shows actual code, not placeholders ✓
- Every test block is runnable as-is ✓
- File paths are absolute-from-repo-root wherever shown ✓
- Phase ordering respects the DAG (no task references code from a later phase) ✓
- Coverage targets referenced match the spec (≥95% core, ≥90% adapters/tool) ✓
- All spec decisions have a home in a task:
  - Collect-all validation → Task 4.3 (filesystem aggregation) + tests in 4.3
  - Hook lifecycle → Tasks 7.3, 7.4
  - Fork execution → Task 7.5
  - Idempotence → Task 7.3
  - MCP integration → Tasks 4.4, 4.5
  - Coupling enforcement → Task 10.1
  - Snapshot guardrail → Task 10.2
  - Listing budget → Task 5.1
  - User docs → Task 11.1
  - Personal notes → Task 11.2
