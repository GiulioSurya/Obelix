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


# Matches an opening `---` on its own line (with optional BOM and trailing ws)
# and captures YAML content up to a closing `---` on its own line.
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
        # YAML lines are 0-indexed in problem_mark; +1 for human-readable
        line_out = (line + 1) if line is not None else None
        raise ParseError(f"invalid YAML: {e}", line=line_out) from e

    if not isinstance(loaded, dict):
        raise ParseError(
            f"frontmatter must be a YAML mapping (dict), got {type(loaded).__name__}"
        )

    return SkillCandidate(file_path=file_path, frontmatter=loaded, body=body)
