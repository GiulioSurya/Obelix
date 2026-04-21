---
description: Writes a conventional commit message from code review findings
when_to_use: Commit agent needs to produce a commit message after code review is complete
context: fork
---

# Commit Message Protocol

You have received code review findings via shared memory context.

## Your task

Write a single conventional commit message that accurately describes the staged changes.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**: `feat` | `fix` | `refactor` | `test` | `docs` | `chore` | `perf`
**Scope**: the module or area affected (e.g. `auth`, `api`, `db`) — omit if changes span many areas
**Subject**: imperative mood, ≤ 72 chars, no trailing period
**Body**: 1–3 lines explaining WHY (not what) — omit if subject is self-explanatory
**Footer**: `Fixes #N`, `BREAKING CHANGE: ...` — omit if not applicable

## Rules

- Use the review findings to understand what changed and why
- If the diff touches multiple unrelated areas, pick the dominant type
- Prefer specific scopes over generic ones (`auth.login` > `auth` > `backend`)
- Never start the subject with "fix: fixed" or "feat: added" — use "fix:" + imperative verb

## Output

Output ONLY the commit message. No preamble, no explanation, no markdown fences.
