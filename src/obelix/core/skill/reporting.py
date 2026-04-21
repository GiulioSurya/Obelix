"""Human-readable formatting of SkillIssue lists.

Takes validation output and renders it. Knows nothing about how issues
were produced.
"""

from __future__ import annotations

from collections import defaultdict

from obelix.core.skill.skill import SkillIssue


def format_validation_error(issues: list[SkillIssue]) -> str:
    """Format a list of issues for human readers.

    Groups by file (with `None` -> `[mcp]`), sorts file paths alphabetically,
    keeps issue order within each group.
    """
    if not issues:
        raise ValueError("cannot format empty issue list")

    by_source: dict[str, list[SkillIssue]] = defaultdict(list)
    for issue in issues:
        key = issue.file_path.as_posix() if issue.file_path is not None else "[mcp]"
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
