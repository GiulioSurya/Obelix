"""Counts occurrences of TEMP-PATCH-SPEC-1 in src/. Fails if the count
diverges from the expected value, signaling that a future cleanup
has either dropped patch sites unintentionally or added new ones
without updating the expected count.

When spec 2 (CLI streaming SSE) lands, this test gets EXPECTED_COUNT=0
and is then deleted along with all the patch sites.

See: docs/superpowers/specs/2026-05-07-a2a-server-drainer-design.md § 9
     docs/superpowers/2026-05-07-a2a-async-agents-roadmap.md
"""

from __future__ import annotations

from pathlib import Path

# Update this number ONLY when consciously adding/removing a TEMP-PATCH-SPEC-1
# marker. A mismatch is a signal that the cleanup audit is out of date.
EXPECTED_COUNT = 4


def _find_repo_root() -> Path:
    """Walk up from this file until pyproject.toml is found."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("could not locate repo root (no pyproject.toml found)")


def _collect_marker_lines(src_dir: Path) -> list[str]:
    """Walk src/ and return every line that contains the TEMP-PATCH-SPEC-1
    marker, formatted as ``path:lineno:line``. Pure Python, no shell tools,
    so the test is portable on Windows where ``grep`` may be absent.
    """
    matches: list[str] = []
    for py in sorted(src_dir.rglob("*.py")):
        try:
            text = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "TEMP-PATCH-SPEC-1" in line:
                matches.append(f"{py}:{lineno}:{line}")
    return matches


def test_temp_patch_marker_count():
    repo_root = _find_repo_root()
    src_dir = repo_root / "src"
    assert src_dir.is_dir(), f"src/ not found at {src_dir}"

    lines = _collect_marker_lines(src_dir)
    assert len(lines) == EXPECTED_COUNT, (
        f"TEMP-PATCH-SPEC-1 count mismatch: expected {EXPECTED_COUNT}, "
        f"found {len(lines)}.\nMatches:\n" + "\n".join(lines)
    )
