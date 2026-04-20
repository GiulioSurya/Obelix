"""Pure placeholder substitution for skill bodies.

No IO, no side effects. Stateless.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path


class SkillInvocationError(Exception):
    """Raised at runtime when a skill is invoked with invalid args."""


def _replace_named(body: str, name: str, value: str) -> str:
    """Replace $<name> only when followed by a non-word char or end-of-string.

    This prevents $pa from being partially replaced inside $path.
    """
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
      1. Named positional ($path, $depth, ...), longest name first (prefix safety).
      2. $ARGUMENTS flat fallback.
      3. Meta placeholders (${OBELIX_SKILL_DIR}, ${OBELIX_SESSION_ID}).
    """
    # Parse args with shell quoting
    try:
        parsed = shlex.split(args) if args else []
    except ValueError as e:
        raise SkillInvocationError(f"failed to parse skill arguments: {e}") from e
    # Flat-mode (no declared args) accepts any args verbatim via $ARGUMENTS —
    # shlex-count is irrelevant. Count check applies only when named positional
    # arguments are declared; excess args would be typo'd skill definitions,
    # caught at load time, not here.
    if declared_args and len(parsed) > len(declared_args):
        raise SkillInvocationError(
            f"skill expects {len(declared_args)} argument(s), got {len(parsed)}"
        )

    # Named positional, longest name first
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
