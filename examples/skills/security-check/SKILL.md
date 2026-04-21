---
description: Quick security audit of staged changes
when_to_use: Reviewer agent needs to check for security vulnerabilities in staged changes
hooks:
  on_tool_error: "If reading a file fails, note the failure inline and continue reviewing the remaining files."
---

# Security Check Protocol

Run a focused security scan on the staged diff.

## Steps

1. Use `git_diff` with mode `staged` to read all changes.
2. For each changed file, check for the issues below.

## What to flag

- **Credentials**: hardcoded passwords, API keys, tokens, secrets — even in comments or test code
- **Injection**: SQL built by string concatenation, shell commands from user input, template injection
- **Path traversal**: user-controlled paths used in file operations without sanitization
- **Auth bypass**: missing auth checks, broken access control, IDOR patterns
- **Sensitive logging**: PII, credentials, or secrets logged at any level

## Output format

If findings exist:
```
## Security Check

### Findings
- [file:line] ISSUE_TYPE — brief description

### Risk level: HIGH / MEDIUM / LOW
```

If no findings:
```
## Security Check

No security issues found in staged changes.
```

Keep it short. Only report actual findings, not speculation.
