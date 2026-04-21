# Skills

## What is a skill?

A **skill** is a markdown file with YAML frontmatter that packages a specialized capability as a prompt. Skills are lazy-loaded: at agent construction only each skill's `name` and `description` enter the system prompt (as a compact listing), while the full body is loaded only when the LLM decides to invoke the skill via the built-in `Skill` tool. This lets you ship dozens of capabilities without blowing up the context window.

Each skill lives on disk as a directory containing a `SKILL.md` file (skills sourced from MCP servers are also supported — see below).

---

## Authoring a skill

A minimal `SKILL.md`:

```markdown
---
description: Reviews a Python file for design quality
when_to_use: User asks for a code review
---
# Code review protocol

Read the target file, flag design smells, suggest refactors.
```

File layout:

```
skills/
  code-reviewer/
    SKILL.md
```

The skill's `name` defaults to the containing directory (`code-reviewer`) unless overridden in frontmatter.

---

## Frontmatter reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `str` | no | Skill identifier. Defaults to the parent directory name. |
| `description` | `str` | **yes** | One-line summary shown in the system prompt listing. Must be non-empty. |
| `when_to_use` | `str` | no | Hint to the LLM about trigger conditions. Strongly recommended — it's what makes the model pick the right skill. |
| `arguments` | `list[str]` | no | Names of named positional arguments (see Placeholders). The name `ARGUMENTS` is reserved. |
| `allowed_tools` | `list[str]` | no | Informational in v1 — the agent does not enforce this list. |
| `context` | `"inline"` \| `"fork"` | no | Execution mode. Defaults to `inline`. See [Fork execution](#fork-execution). |
| `hooks` | `dict[str, str]` | no | Map of `AgentEvent` → instruction to inject. See [Hook frontmatter](#hook-frontmatter). |

Unknown frontmatter keys are **ignored** (forward-compat). All fields are validated at agent construction; on failure `SkillValidationError` is raised with every issue aggregated.

---

## Placeholders

The skill body supports four placeholder forms, substituted just before the body is returned to the LLM:

| Syntax | Meaning |
|--------|---------|
| `$ARGUMENTS` | The raw args string passed by the LLM, verbatim. |
| `$<name>` | A named positional argument declared in `arguments:`. Parsed with `shlex.split`. |
| `${OBELIX_SKILL_DIR}` | Absolute path of the skill's containing directory (filesystem skills only — empty for MCP). |
| `${OBELIX_SESSION_ID}` | UUID identifying the agent instance; stable across all invocations of the same agent. |
| `\$foo` | Escape — literal `$foo` is preserved. |

Rules:

- Named args are substituted longest-name-first, so `$pa` inside `$path` is not partially replaced.
- Every `$<name>` in the body must be declared in `arguments:` (or be `$ARGUMENTS`) — otherwise validation fails with `placeholder '$foo' referenced but not declared`.
- Passing more positional args than declared raises `SkillInvocationError` at invocation time.

Example — `arguments: [path, depth]`, body `Target $path at depth $depth.` invoked as `./src/obelix 2` renders to `Target ./src/obelix at depth 2.`

---

## Wiring into an agent

Pass `skills_config` to `BaseAgent`:

```python
from obelix.core.agent.base_agent import BaseAgent

agent = BaseAgent(
    system_message="You are X.",
    provider=provider,
    skills_config="./skills/code-reviewer",
)
```

Accepted shapes for `skills_config`:

| Value | Meaning |
|-------|---------|
| `None` (default) | Skills disabled. No `Skill` tool registered. |
| `str` or `Path` pointing to a skill dir | Load that single skill. |
| `str` or `Path` pointing to a directory of skill dirs | Load every skill dir inside. |
| `list[str \| Path]` | Mix of skill dirs and skill-root dirs. |

Empty strings are rejected (`ValueError`) — pass `None` to disable. When at least one skill is registered, a built-in `Skill` tool is added to the agent and its listing is injected into the system prompt via `system_prompt_fragment()` (budget ≈ 1% of a 200k-token context, i.e. ~8000 chars).

If `mcp_config` is also set on the agent, MCP prompts are automatically discovered as skills under the name `mcp__<server>__<prompt_name>` and appear in the same listing.

---

## Hook frontmatter

Skills can inject instructions into the parent agent at specific lifecycle events. The map key must exactly match an `AgentEvent` value:

- `before_llm_call`
- `after_llm_call`
- `before_tool_execution`
- `after_tool_execution`
- `on_tool_error`
- `before_final_response`
- `query_end`

```yaml
hooks:
  on_tool_error: "The last tool failed. Re-read its schema, then retry with fixed args."
  before_final_response: "Before answering, double-check the output format."
```

Each value is a **string instruction** appended as a `HumanMessage` to the conversation when the event fires. Hooks are **scoped to the query**: they are registered on first invocation of the skill and automatically unregistered at `QUERY_END` — the next query starts clean. For `context: fork` skills, hooks are applied to the ephemeral inner agent only.

Unknown event names are rejected at construction with an explicit list of valid events (catches typos like `on_tool_erorr`).

---

## Fork execution

Setting `context: fork` runs the skill in an **isolated sub-agent**:

- Fresh conversation history — the rendered skill body becomes the sub-agent's system message.
- Inherits the parent's provider, registered tools, and `max_iterations`.
- Does **NOT** inherit memory or conversation history (`SubAgentWrapper(stateless=True)`).
- Only the sub-agent's last `AssistantMessage` content flows back as the `Skill` tool result.

Use fork when the skill benefits from an empty context (e.g., a disciplined checklist walk) or when you explicitly do not want its intermediate reasoning to pollute the parent's history. Pass any parent state the skill needs via `arguments:` — there is no other channel.

---

## Worked examples

### 1. `code-reviewer` — inline with named argument

`skills/code-reviewer/SKILL.md`:

```markdown
---
description: Reviews a Python file for design quality
when_to_use: User asks for a code review of a specific file
arguments: [path]
---
# Code review protocol

Target file: $path

1. Read the file fully before commenting.
2. Flag design smells (god classes, feature envy, primitive obsession).
3. Propose concrete refactors with code snippets.
4. End with a one-line verdict.
```

```python
from obelix.core.agent.base_agent import BaseAgent

agent = BaseAgent(
    system_message="You are a senior Python reviewer.",
    provider=provider,
    skills_config="./skills/code-reviewer",
)
agent.execute_query("Please review src/obelix/core/agent/base_agent.py")
```

### 2. `commit-writer` — fork, no arguments

`skills/commit-writer/SKILL.md`:

```markdown
---
description: Writes a conventional-commit message from staged changes
when_to_use: User asks to craft a commit message
context: fork
---
You are a conventional-commit writer.

1. Call the shell tool to run `git diff --staged`.
2. Summarize the change in one imperative sentence.
3. Output only the commit message, no preamble.
```

```python
agent = BaseAgent(
    system_message="You are a git helper.",
    provider=provider,
    tools=[BashTool],
    skills_config="./skills/commit-writer",
)
# The parent history only ever sees: tool_call -> final commit message.
agent.execute_query("Write me a commit message.")
```

### 3. `debugger` — inline with `on_tool_error` hook

`skills/debugger/SKILL.md`:

```markdown
---
description: Systematic debugging protocol
when_to_use: User reports a bug or failing test
hooks:
  on_tool_error: "A tool just failed. Before retrying, re-read its schema and explain in one line why it failed."
---
# Debugging protocol

Reproduce the failure, form a hypothesis, isolate, fix, add a regression test.
```

```python
agent = BaseAgent(
    system_message="You are a debugging assistant.",
    provider=provider,
    tools=[BashTool],
    skills_config="./skills",      # directory-of-dirs form
)
agent.execute_query("My tests fail with ImportError on obelix.core.skill.")
```

---

## Troubleshooting

**`SkillValidationError` at agent construction.** Every issue is aggregated in a single exception. Read the message — each entry pinpoints the file, field, and problem. Common causes: missing `description`, typo in a hook event name, a `$foo` placeholder without a matching `arguments:` declaration, empty body.

**LLM never invokes the skill.** Strengthen `when_to_use` — it's the primary trigger signal. Make it concrete: `User asks for X` beats `Useful for various tasks`. Also check the skill actually appears in the listing (the built-in `Skill` tool injects it into the system prompt).

**"Skill 'X' is already active" on second invocation.** Expected. Skills are **idempotent per query**: a second call in the same query returns a short marker instead of re-injecting the body. The idempotence state is reset at `QUERY_END`.

**Fork skill doesn't see parent context.** By design — `context: fork` runs in isolation. If the skill needs parent state, declare `arguments:` and have the parent pass them explicitly.
