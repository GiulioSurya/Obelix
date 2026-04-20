# Skills System — Design Spec

**Date**: 2026-04-20
**Status**: Draft
**Inspiration**: Claude Code skills system (compat on format, not on discovery)

---

## Goal

Introduce a **Skills** subsystem in Obelix, analogous to Claude Code's skills. A Skill is a markdown file with YAML frontmatter that describes a specialized capability (code review, commit writing, debugging protocol, etc.). Skills are lazy-loaded: only the name + description enter the LLM context at boot; the full body is loaded only when the LLM invokes the Skill tool.

Two user-visible outcomes:

1. Obelix agents can invoke skills in the middle of a query.
2. Skills authored for the community skill format (same SKILL.md + frontmatter convention) work in Obelix, provided the user copies them into the agent's configured skill paths.

## Scope

### In scope (v1)

- Explicit skill loading via `skills_config` on `BaseAgent` (no filesystem auto-discovery)
- Skill format: directory with `SKILL.md`, YAML frontmatter + markdown body
- Supported frontmatter fields: `name`, `description`, `when_to_use`, `allowed_tools`, `arguments`, `context`, `hooks`
- Built-in `SkillTool` registered on agents that have skills
- Inline skill execution (body returned as ToolResult, LLM follows instructions in next iteration)
- Fork execution via existing `SubAgentWrapper` (skills with `context: fork`)
- Hook frontmatter mapped to existing `AgentEvent` enum
- MCP-sourced skills (when `mcp_config` is present) via `MCPManager.list_prompts()`
- Placeholder substitution: `$ARGUMENTS`, named positional (`$path`, `$depth`, ...), `${OBELIX_SKILL_DIR}`, `${OBELIX_SESSION_ID}`
- Comprehensive boot-time validation (all errors collected, not fail-fast)
- Budget-bounded skill listing in system prompt (1% of context window)

### Out of scope (v1)

- Filesystem auto-discovery (no walk-up, no `~/.obelix/skills/` auto-read, no `~/.claude/skills/` compat)
- `paths:` conditional activation (skill activated when tool touches matching file)
- Shell injection in body (`` !`cmd` ``) — security risk, needs sandboxing
- `model:` / `effort:` override
- Plugin marketplace / remote skill search
- `SkillConfig` class (parameter accepts path/list only; can add class later without breaking change)
- Skill-to-skill cross-agent sharing via `AgentFactory` (skills are per-agent)

## Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Discovery | Explicit paths only, no auto-search | Determinism, easier deployment (A2A server), no surprises |
| 2 | Config shape | `str \| Path \| list \| None` | Mirrors `mcp_config`, no extra class needed in v1 |
| 3 | Scope | Per-agent (not per-factory) | Skills are workload-specific, not infra-wide |
| 4 | Validation | Collect-all, boot-time | A well-engineered framework validates exhaustively; fail-fast hides compound errors |
| 5 | Tool invocation | LLM uses built-in `Skill` tool | Matches CC pattern; leverages existing `register_tool` + `system_prompt_fragment` |
| 6 | Idempotence | Second invoke of same skill returns "already active" | Prevents duplicate hook registration and body re-injection |
| 7 | Arguments | `$ARGUMENTS` flat + named positional via shlex | Flexibility without the heavy machinery of full DSL |
| 8 | Fork | Uses existing `SubAgentWrapper`, inherits tools | Reuses infra; isolates token budget |
| 9 | Fork memory | NOT inherited | Isolation contract |
| 10 | Fork return | Last AssistantMessage only | Point of fork is summarization |
| 11 | Hooks events | snake_case YAML keys | Consistent with Python enum |
| 12 | Hook lifecycle | Per-query (cleared on `QUERY_END`) | No leaked state across queries |
| 13 | Hook composition | Skill hooks run alongside programmatic hooks | Non-interference |
| 14 | Event naming | Snake case (matches `AgentEvent` enum) | Idiomatic Python |
| 15 | Skill path precedence | filesystem > mcp | Filesystem is explicit user choice; MCP can change remotely |
| 16 | Listing budget | 1% of context window, adaptive truncation | Same heuristic as CC (`prompt.ts:70`) |
| 17 | Deliverables | Code + `docs/skills.md` (EN, user) + `docs/skills_implementation_notes.md` (IT, personal) | Official docs + personal reference |

---

## Architecture

### Design principles

Three rules the design is optimized for:

1. **Single responsibility**: every module has one reason to change. No module mixes "parse YAML" with "write to disk" with "format error messages". Each concern is its own file.
2. **Acyclic, shallow dependencies**: the module graph is a DAG. No cycles. Every module depends on as few other modules as possible. Data flows bottom-up from pure leaves (dataclasses, pure functions) to orchestrators (SkillTool, BaseAgent integration).
3. **Depend on abstractions, not implementations**: `SkillManager` talks to `AbstractSkillProvider`, not `FilesystemSkillProvider`. `SkillTool` talks to `SkillManager`, not providers. Adding a future provider (remote skills, bundled skills) is adding a file, not refactoring.

### Dependency graph (must stay acyclic)

```
                     skill.py (dataclass, pure)
                         ▲
         ┌───────────────┼───────────────┐
         │               │               │
    parsing.py    validation.py    substitution.py
    (YAML→raw)    (raw→validated)  (pure function)
         ▲               ▲               ▲
         └───────────────┤               │
                         │               │
                skill_provider.py        │
                (ABC: discover)          │
                         ▲               │
         ┌───────────────┴───────┐       │
         │                       │       │
    filesystem.py            mcp.py      │
    (IO + parse + validate)  (IO via MCPManager)
         ▲                       ▲       │
         └───────────┬───────────┘       │
                     │                   │
                manager.py               │
                (aggregate, dedup,       │
                 format listing)         │
                     ▲                   │
                     └─────┬─────────────┘
                           │
                      skill_tool.py
                      (orchestrate invocation)
                           ▲
                           │
                      base_agent.py
                      (wire-up only)
```

### File structure

```
ports/outbound/
  skill_provider.py           # AbstractSkillProvider (ABC)

adapters/outbound/skill/
  __init__.py                 # Public exports
  filesystem.py               # FilesystemSkillProvider: reads SKILL.md, delegates to parsing + validation
  mcp.py                      # MCPSkillProvider: wraps MCPManager.list_prompts()

core/skill/
  __init__.py                 # Public API exports
  skill.py                    # Skill (validated), SkillCandidate (raw), SkillIssue
  parsing.py                  # parse_skill_file(): file → SkillCandidate | ParseError
  validation.py               # Validator protocol + validator chain + run_validators()
  reporting.py                # SkillValidationError.format() — human-readable issue aggregation
  substitution.py             # substitute_placeholders() — pure function
  manager.py                  # SkillManager — aggregates providers, dedups, formats listing

plugins/builtin/
  skill_tool.py               # SkillTool — orchestrates invocation (inline or fork)

core/agent/
  base_agent.py               # MODIFIED: +skills_config param, minimal wiring
```

### Responsibility matrix

Each module has exactly one reason to change. Dependencies column is *direct* imports only.

| Module | Single responsibility | Depends on |
|--------|-----------------------|------------|
| `skill.py` | Data shape (`Skill`, `SkillCandidate`, `SkillIssue` dataclasses) | — (stdlib only) |
| `parsing.py` | Split raw file into frontmatter + body, produce `SkillCandidate` | `skill.py`, `yaml`, `pydantic` |
| `validation.py` | Run validator chain on `SkillCandidate` → `Skill` or `list[SkillIssue]` | `skill.py`, `obelix.core.agent.hooks` (enum lookup) |
| `reporting.py` | Format `list[SkillIssue]` into human-readable error message | `skill.py` |
| `substitution.py` | Replace `$ARG`, `${OBELIX_SKILL_DIR}`, etc. in body | — (stdlib `shlex`) |
| `skill_provider.py` | Contract `discover() -> list[Skill]` | `skill.py` |
| `filesystem.py` | Walk paths, read files, delegate to parsing+validation | `skill.py`, `parsing.py`, `validation.py`, `skill_provider.py` |
| `mcp.py` | Wrap `MCPManager.list_prompts()` output as `Skill` objects | `skill.py`, `validation.py`, `skill_provider.py`, MCP manager |
| `manager.py` | Aggregate providers, dedup, build listing within budget | `skill.py`, `skill_provider.py` |
| `skill_tool.py` | Orchestrate skill invocation (inline/fork), hook registration, `system_prompt_fragment` | `skill.py`, `manager.py`, `substitution.py`, `hooks.py`, `SubAgentWrapper` |
| `base_agent.py` | Wire `skills_config` → `SkillManager` → register `SkillTool` | `manager.py`, `skill_tool.py`, `filesystem.py`, `mcp.py` (imports only to construct) |

### Coupling rules (enforceable in review)

| Rule | Rationale |
|------|-----------|
| `skill.py` must never import from other `core/skill/*` modules | Pure data layer, leaf of DAG |
| `validation.py` must not do IO | Pure validators; tests never touch disk |
| `parsing.py` must not validate semantic rules (e.g. "this hook event exists") | Parsing and validation are separate concerns |
| `reporting.py` must not know how issues are produced | Reporter consumes `SkillIssue` list, nothing else |
| `manager.py` must not know about YAML, files, or MCP | Talks only to `AbstractSkillProvider` |
| `skill_tool.py` must not import from `filesystem.py` or `mcp.py` | Orchestrator is provider-agnostic |
| `base_agent.py` delta is <30 LOC | Wiring only; any logic must live in `core/skill/` |
| No circular import between skill/ submodules | Enforced by the DAG above |

---

## User experience

### Authoring a skill

```
my_project/skills/code-reviewer/
├── SKILL.md
└── checklist.md          # optional, referenceable via ${OBELIX_SKILL_DIR}
```

`SKILL.md`:
```markdown
---
name: code-reviewer                # optional; defaults to directory name
description: Reviews a Python file for design quality
when_to_use: User asks for code review
arguments: [path]
allowed_tools: [Read, Grep]        # informational in v1
hooks:
  on_tool_error: "If a tool fails, stop and ask the user before retrying."
---

# Code Review Protocol

Review the file at $path.
Use ${OBELIX_SKILL_DIR}/checklist.md as your reference.

Produce a report with severity levels (high/medium/low).
```

### Wiring into an agent

```python
from obelix.core.agent import BaseAgent

agent = BaseAgent(
    system_message="You are a senior Python engineer.",
    provider=provider,
    skills_config="./skills/code-reviewer",      # single skill dir
)

# Or a directory of skills (each subdir is a skill)
agent = BaseAgent(..., skills_config="./skills/")

# Or a list of paths
agent = BaseAgent(..., skills_config=[
    "./skills/code-reviewer",
    "/opt/shared-skills/commit-writer",
])
```

### Runtime

```python
async with agent:                                 # required if mcp_config present
    result = await agent.execute_query("review src/foo.py")
```

Behind the scenes:
1. LLM sees skill listing in system prompt (injected via `SkillTool.system_prompt_fragment()`)
2. LLM calls `Skill(name="code-reviewer", args="src/foo.py")`
3. `SkillTool` loads body, substitutes `$path=src/foo.py`, registers hook for `ON_TOOL_ERROR`
4. Body returned as `ToolResult` → LLM next iteration follows the protocol
5. At `QUERY_END`, skill-scoped hooks are removed

---

## Data model

### `Skill` dataclass

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dataclasses import field

@dataclass(frozen=True)
class Skill:
    name: str                                 # unique identifier
    description: str                          # required, shown in listing
    body: str                                 # markdown after frontmatter
    base_dir: Path | None                     # directory for ${OBELIX_SKILL_DIR}
    when_to_use: str | None = None
    allowed_tools: tuple[str, ...] = ()       # informational in v1
    arguments: tuple[str, ...] = ()           # named positional arg slots
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = field(default_factory=dict)   # event_name -> instruction
    source: Literal["filesystem", "mcp"] = "filesystem"
    file_path: Path | None = None             # origin, for error messages
```

### `SkillFrontmatter` Pydantic model (validation-only)

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal

class SkillFrontmatter(BaseModel):
    name: str | None = None                   # defaults to dir name
    description: str = Field(..., min_length=1)
    when_to_use: str | None = None
    allowed_tools: list[str] = []
    arguments: list[str] = []
    context: Literal["inline", "fork"] = "inline"
    hooks: dict[str, str] = {}

    @field_validator("arguments")
    @classmethod
    def no_duplicate_args(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("duplicate argument names")
        if "ARGUMENTS" in v:
            raise ValueError("'ARGUMENTS' is reserved, use a different name")
        return v
```

### Supported hook events

Mapped 1:1 to `obelix.core.agent.hooks.AgentEvent`:

| YAML key | Enum value |
|---|---|
| `before_llm_call` | `BEFORE_LLM_CALL` |
| `after_llm_call` | `AFTER_LLM_CALL` |
| `before_tool_execution` | `BEFORE_TOOL_EXECUTION` |
| `after_tool_execution` | `AFTER_TOOL_EXECUTION` |
| `on_tool_error` | `ON_TOOL_ERROR` |
| `before_final_response` | `BEFORE_FINAL_RESPONSE` |
| `query_end` | `QUERY_END` |

YAML keys mirror the `AgentEvent.value` of each enum member (which is already snake_case).

---

## Validation

### Principle

A well-built framework validates exhaustively and reports every problem in one shot. Three invariants:

- **Collect-all**: never short-circuit on the first error. Every check runs; every issue is appended to a bag.
- **Pure**: validation does no IO and produces no side effects. It reads a `SkillCandidate` and yields `list[SkillIssue]`.
- **Composable**: each check is its own class implementing `Validator`. Adding a rule is adding a file, not editing orchestrator code.

### Three-stage pipeline

```
file content (str)
      │
      ▼  parsing.py — syntactic concerns only
SkillCandidate (raw frontmatter + body, pre-semantic)
      │
      ▼  validation.py — semantic concerns, validator chain
Skill (validated)  OR  list[SkillIssue]
      │
      ▼  reporting.py — aggregate issues from ALL skills
SkillValidationError (raised once, contains every issue found)
```

Each stage is a separate file with a distinct responsibility. Parsing does not know what fields mean. Validation does not touch disk. Reporting does not know how issues were produced.

### Validator protocol

```python
# validation.py

from typing import Protocol
from obelix.core.skill.skill import SkillCandidate, SkillIssue

class Validator(Protocol):
    """One validator = one rule. Pure. Stateless."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...


class FrontmatterSchemaValidator:
    """Required fields, types, Literal values (Pydantic under the hood)."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...

class HookEventValidator:
    """Every hook key maps to a valid AgentEvent."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...

class ArgumentUniquenessValidator:
    """No duplicates in `arguments`; reserved name 'ARGUMENTS' forbidden."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...

class PlaceholderConsistencyValidator:
    """Every `$X` in body (except `$ARGUMENTS`, `${OBELIX_*}`) is in declared `arguments`."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...

class ContextValueValidator:
    """`context` is 'inline' or 'fork'."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...

class BodyNonEmptyValidator:
    """Body (after frontmatter) is non-empty after stripping whitespace."""
    def check(self, candidate: SkillCandidate) -> list[SkillIssue]: ...


DEFAULT_VALIDATORS: tuple[Validator, ...] = (
    FrontmatterSchemaValidator(),
    HookEventValidator(),
    ArgumentUniquenessValidator(),
    PlaceholderConsistencyValidator(),
    ContextValueValidator(),
    BodyNonEmptyValidator(),
)


def run_validators(
    candidate: SkillCandidate,
    validators: tuple[Validator, ...] = DEFAULT_VALIDATORS,
) -> list[SkillIssue]:
    """Pure. Runs every validator against the candidate. No short-circuit."""
    issues: list[SkillIssue] = []
    for v in validators:
        issues.extend(v.check(candidate))
    return issues
```

### `SkillIssue` data shape

```python
@dataclass(frozen=True)
class SkillIssue:
    file_path: Path | None       # None if candidate is MCP-sourced
    field: str                   # 'frontmatter.description', 'hooks.on_random', 'body'
    message: str                 # human-readable reason
    line: int | None = None      # if derivable from YAML parser
```

### Orchestration (filesystem provider)

```python
# adapters/outbound/skill/filesystem.py

def discover(self) -> list[Skill]:
    all_issues: list[SkillIssue] = []
    validated: list[Skill] = []

    for path in self._candidate_files():
        raw = path.read_text(encoding="utf-8")
        try:
            candidate = parse_skill_file(raw, path)
        except ParseError as e:
            all_issues.append(SkillIssue(path, field="parse", message=str(e), line=e.line))
            continue

        issues = run_validators(candidate)
        if issues:
            all_issues.extend(issues)
            continue

        validated.append(Skill.from_candidate(candidate))

    if all_issues:
        raise SkillValidationError(all_issues)   # formatted via reporting.py
    return validated
```

Key point: **even when a skill has issues, the loop continues to the next file**. All issues from all skills are aggregated before raising.

### `SkillValidationError` — formatted output

Rendered by `reporting.py`:

```
SkillValidationError: 3 skill(s) failed validation (5 issue(s) total)

./skills/a/SKILL.md:
  [frontmatter.description] missing required field
  [body] placeholder '$depth' referenced but not declared in 'arguments'

./skills/b/SKILL.md:
  [parse] invalid YAML at line 3: mapping values are not allowed here
  [hooks.on_random] unknown event. Valid: before_llm_call, after_llm_call, before_tool_execution, after_tool_execution, on_tool_error, before_final_response, query_end

./skills/c/SKILL.md:
  [arguments] duplicate name 'path'
```

### Validation checks catalog

| Validator | Target | On failure |
|-----------|--------|------------|
| `FrontmatterSchemaValidator` | Required fields, types, literals | One issue per Pydantic error, field-scoped |
| `HookEventValidator` | `hooks.*` keys | "unknown event 'X'. Valid: ..." |
| `ArgumentUniquenessValidator` | `arguments` list | "duplicate name 'X'" / "'ARGUMENTS' is reserved" |
| `PlaceholderConsistencyValidator` | Body scan | "placeholder '$X' referenced but not declared" |
| `ContextValueValidator` | `context` field | "invalid value 'X', must be 'inline' or 'fork'" |
| `BodyNonEmptyValidator` | Body string | "empty body" |

### Cross-skill duplicates (warning, not error)

Handled in `SkillManager.__init__`, after providers discover. If two skills share a name:
- Apply precedence: filesystem > mcp
- Log a warning, keep the winner, discard the loser
- No `SkillValidationError` — this is a runtime coexistence concern, not a skill-level defect

### MCP skills

MCP-sourced skills build a `SkillCandidate` from MCP prompt metadata (no YAML parsing). Then run through the **same validator chain**. A malformed MCP prompt is **logged and skipped**, not raised — Obelix cannot repair remote servers, but it refuses to present broken skills to the LLM.

---

## Runtime flow

### Inline execution (default)

```
LLM invokes Skill(name="X", args="...")
   ↓
SkillTool.execute:
   1. skill = manager.load(name)
   2. if skill is None: return ToolResult(error="skill 'X' not found, available: [...]")
   3. if skill already active in this query: return ToolResult("already active, continue")
   4. Substitute placeholders in skill.body:
        - $ARGUMENTS = args
        - $<arg_name> = shlex.split(args)[i]  (missing → empty string)
        - ${OBELIX_SKILL_DIR} = str(skill.base_dir)
        - ${OBELIX_SESSION_ID} = tracer.session_id if tracer else uuid
   5. If skill.hooks: register skill-scoped hooks on agent
   6. Mark skill active for this query
   7. Return ToolResult(content=rendered_body)

Next iteration: LLM reads body, follows instructions.
On QUERY_END: skill-scoped hooks are removed, active set cleared.
```

### Fork execution (`context: fork`)

```
SkillTool.execute with skill.context == "fork":
   1. Substitute placeholders in skill.body (system message for fork)
   2. Build SubAgentWrapper:
        - system_message = rendered_body
        - provider = parent.provider
        - tools = list(parent.registered_tools)   # includes SkillTool so fork can compose skills
        - mcp_config = None   # MCP tools already in parent.registered_tools
        - memory_graph = None   # isolated
   3. Register skill.hooks on SUB-AGENT (not parent)
   4. result = await sub_agent.execute_query("Begin executing the skill as specified above.")
   5. Return ToolResult(content=result)   # only last AssistantMessage
```

### Placeholder substitution algorithm

```python
def substitute(body: str, args: str, declared_args: tuple[str, ...], base_dir: Path, session_id: str) -> str:
    # Named positional args (shlex-split args)
    parsed = shlex.split(args) if args else []
    if len(parsed) > len(declared_args):
        raise SkillInvocationError(f"skill expects {len(declared_args)} args, got {len(parsed)}")
    for i, name in enumerate(declared_args):
        value = parsed[i] if i < len(parsed) else ""
        body = body.replace(f"${name}", value)
    # Flat fallback
    body = body.replace("$ARGUMENTS", args)
    # Meta
    body = body.replace("${OBELIX_SKILL_DIR}", str(base_dir) if base_dir else "")
    body = body.replace("${OBELIX_SESSION_ID}", session_id)
    return body
```

Note: replacement is order-sensitive. Named positional replaced first (so `$path` does not get partially replaced by `$ARGUMENTS` substitution).

---

## Hook frontmatter

### Example

```yaml
hooks:
  on_tool_error: "A tool failed. Show the error, then ask whether to rollback."
  query_end: "Before concluding, list all files modified."
```

### Registration

At first invocation of a skill with hooks:
```python
for event_name, instruction in skill.hooks.items():
    event = _map_yaml_to_enum(event_name)    # e.g. "on_tool_error" -> AgentEvent.ON_TOOL_ERROR
    hook_id = agent.on(event).inject(lambda s, msg=instruction: HumanMessage(content=msg))
    self._active_skill_hooks.append((event, hook_id))   # for later removal
```

### Lifecycle

- Scope = one `execute_query` on the parent agent.
- At `QUERY_END` of that query, SkillTool removes all hooks it registered.
- The hook registry in `BaseAgent` needs to support **removal** (see Implementation Notes below).
- For fork: hooks register on the sub-agent; the sub-agent is ephemeral, so hooks die with it.

### Composition

- Programmatic hooks registered by user code are **never** touched.
- If both exist on the same event, they fire in registration order (programmatic first, then skill's).
- Skill hooks carry an internal tag `_source = ("skill", skill_name)` so they are identifiable.

### Implementation note: hook removal

Current `hooks.py` stores hooks in `dict[AgentEvent, list[Hook]]`. Removal is not currently exposed. v1 adds:

```python
# base_agent.py
def _unregister_hook(self, event: AgentEvent, hook: Hook) -> None:
    self._hooks[event] = [h for h in self._hooks[event] if h is not hook]
```

Kept `_`-private; only `SkillTool` uses it.

---

## Skill listing in system prompt

### Generation

`SkillTool.system_prompt_fragment()` returns the listing. Called by `BaseAgent.register_tool()` automatically.

```
## Available skills

Use the Skill tool to invoke one of the following:

- code-reviewer: Reviews a Python file for design quality — User asks for code review
- commit-writer: Creates conventional commits — User asks to commit
- debugger: Debugs failing tests using scientific method — User reports a bug

When a skill matches, invoke it BEFORE generating your response.
```

### Budget

- Target: 1% of provider context window, expressed in characters (roughly `ctx_tokens × 4 × 0.01`).
- Default fallback: 8000 chars (1% of 200k).
- Algorithm (port of `prompt.ts:formatCommandsWithinBudget`):
  1. Compute full listing. If under budget → use it.
  2. Else: truncate descriptions uniformly to fit, preserving all skill names.
  3. Else (extreme): names-only listing.

### Listing item format

```
- <name>: <description> — <when_to_use>
```

`when_to_use` included only if present. Description truncated at 250 chars per entry (CC convention).

---

## MCP integration

### Wiring

```python
# In SkillManager.__init__:
providers: list[AbstractSkillProvider] = []
if filesystem_paths:
    providers.append(FilesystemSkillProvider(filesystem_paths))
if mcp_manager:
    providers.append(MCPSkillProvider(mcp_manager))

all_skills = []
for p in providers:
    all_skills.extend(p.discover())
# dedup by name, filesystem wins over mcp
```

### MCP prompts as skills

`MCPManager.list_prompts()` (to add) returns `list[mcp.types.Prompt]`. For each prompt:
- `name` → skill name, prefixed `mcp__<server>__<name>` to avoid collision
- `description` → skill description
- Prompt arguments → `arguments` list
- Prompt template → body (markdown)

If the MCP prompt does not fit the skill shape (e.g. complex argument types), it is skipped with a debug log. No failure.

### Security

- MCP skills do NOT execute shell inline (no feature in v1 anyway).
- MCP skills cannot invoke filesystem skills (just an invocation semantics, nothing special).
- MCP skill body is treated as untrusted input: the agent's tool permission policy still applies.

---

## Security considerations

| Surface | Risk | Mitigation |
|---|---|---|
| Arbitrary markdown body | Prompt injection against the agent | Same risk as any user-provided content; agent tool policies still apply |
| Placeholder substitution | Injection via args | Simple string replace, no templating engine; args are opaque strings |
| MCP skills | Remote server supplies the body | Only run if user explicitly set `mcp_config` |
| Fork execution | Sub-agent inherits tools | Tool policies are inherited; no privilege escalation |
| `allowed_tools:` frontmatter | Informational only in v1 | Documented as non-enforcing |
| File path traversal | Skill referencing `../../../etc/passwd` via `${OBELIX_SKILL_DIR}/...` | `base_dir` is resolved at validation; no escape; LLM could still guess but `Read` tool has its own sandbox |

No shell injection — the `` !`cmd` `` syntax of CC is NOT supported in v1.

---

## Error catalog

All errors under a common base `SkillError`.

| Error | Raised when | Contains |
|---|---|---|
| `SkillValidationError` | One or more skills fail boot validation | List of `SkillIssue(path, line, field, message)` |
| `SkillNotFoundError` | LLM invokes unknown skill at runtime | Skill name + available names |
| `SkillInvocationError` | Invalid args at runtime (e.g. too many positional) | Skill name, args, expected shape |
| `SkillCollisionError` | Two filesystem skills have identical resolved `realpath` | Both paths, chosen one |

Runtime errors (`SkillNotFoundError`, `SkillInvocationError`) are converted to `ToolResult(error=...)` by the SkillTool, **never** raised to the agent loop — the LLM sees them as a normal tool failure and decides how to recover.

---

## Deliverables

### Code

- `src/obelix/core/skill/` (~400 LOC + tests)
- `src/obelix/ports/outbound/skill_provider.py` (~30 LOC)
- `src/obelix/adapters/outbound/skill/` (~250 LOC + tests)
- `src/obelix/plugins/builtin/skill_tool.py` (~150 LOC + tests)
- `src/obelix/core/agent/base_agent.py` modifications (~20 LOC)

### Tests

- Unit: Skill dataclass validation, frontmatter schema, substitution edge cases, collect-all validation, hook registration/removal, listing budget algorithm.
- Integration: full agent loop with a skill invocation (inline); full agent loop with fork skill; MCP skill discovery via fake MCP server; hook firing during skill execution.

### Documentation

- `docs/skills.md` — user-facing, EN. How to write a SKILL.md, frontmatter reference, invocation examples, fork vs inline, hook frontmatter, troubleshooting. Includes 3+ complete runnable examples (code-reviewer, commit-writer, debugger).
- `docs/skills_implementation_notes.md` — personal, IT. What's implemented in v1 vs not, why, concrete usage scenarios with full end-to-end examples, known limitations.
- Update `docs/index.md` to link the new user-facing doc.
- Update `docs/base_agent.md` with the `skills_config` parameter.

---

## Testing strategy

Tests are a first-class deliverable, not an afterthought. The validator-chain and pure-function architecture is deliberately built for deep unit coverage.

### Coverage targets

| Layer | Target |
|-------|--------|
| `core/skill/*` (pure logic) | **≥ 95% line + branch** |
| `adapters/outbound/skill/*` (IO) | **≥ 90% line** |
| `plugins/builtin/skill_tool.py` (orchestration) | **≥ 90% line + branch** |
| `base_agent.py` delta (wiring) | **100% of new lines** |

Measured with `pytest --cov=obelix.core.skill --cov=obelix.adapters.outbound.skill --cov=obelix.plugins.builtin.skill_tool --cov-branch`. CI must fail if below target.

### Test isolation rules

- **Unit tests do NO disk IO** (except tmp_path fixtures when strictly needed). `parsing.py`, `validation.py`, `substitution.py`, `reporting.py`, `manager.py`, `skill_tool.py` tested against in-memory fixtures.
- **Unit tests do NO network IO**. MCP integration uses an in-process fake implementing the `MCPManager` surface.
- **Unit tests do NOT run an LLM**. `SkillTool.execute()` tested with a stub parent agent; fork execution tested with a stub `SubAgentWrapper`.
- **Integration tests** live in `tests/integration/skill/`, may touch `tmp_path`, may spin a fake MCP server, never hit a real provider.

### Fixture catalog — `tests/fixtures/skills/`

A corpus of real SKILL.md files covering valid and invalid cases, loaded via a `fixture_skill(name)` helper. Each fixture is its own directory so filesystem paths behave realistically.

**Valid fixtures** (smoke + coverage):
- `minimal/` — only `description`, body
- `with_when_to_use/`
- `with_named_args/` — `arguments: [path, depth]`
- `with_flat_args_only/` — uses `$ARGUMENTS`
- `with_mixed_args/` — has `arguments: [path]` and body uses both `$path` and `$ARGUMENTS`
- `with_all_hooks/` — one hook per supported event
- `with_single_hook/`
- `fork_context/` — `context: fork`
- `with_subdir_files/` — references `${OBELIX_SKILL_DIR}/checklist.md`

**Invalid fixtures** (validation coverage):
- `missing_description/`
- `invalid_yaml_syntax/`
- `empty_body/`
- `unknown_hook_event/`
- `duplicate_arguments/`
- `reserved_argument_name/` — uses `ARGUMENTS` in `arguments:`
- `undeclared_placeholder/` — body uses `$foo` but `arguments: [path]`
- `invalid_context_value/` — `context: parallel`
- `frontmatter_missing/`
- `multiple_issues/` — triggers 3 issues at once (to verify collect-all)

### Unit test matrix — by module

#### `parsing.py`
- Valid YAML + body → `SkillCandidate` with correct fields
- YAML with line-numbered errors → `ParseError.line` is populated
- No frontmatter at all → `ParseError("missing frontmatter")`
- Frontmatter with no body → `SkillCandidate(body="")`
- Unicode in body and frontmatter values
- Windows CRLF line endings
- BOM prefix on file
- Frontmatter delimiter variations (`---` vs `----`)

#### `validation.py` — one test class per Validator
Each validator tested independently against `SkillCandidate` literals.

**`FrontmatterSchemaValidator`**:
- Happy: all fields present → zero issues
- Missing `description` → one issue, `field == "frontmatter.description"`
- `description` is empty string → one issue
- Wrong type (`description: 42`) → one issue
- Unknown frontmatter key present → zero issues (forward-compat, keys silently ignored)

**`HookEventValidator`**:
- All supported events → zero issues
- Single unknown event → one issue listing all valid events
- Multiple unknown events → multiple issues (never short-circuit)
- Empty hooks dict → zero issues

**`ArgumentUniquenessValidator`**:
- Unique names → zero
- Duplicates → one issue per duplicate pair
- `ARGUMENTS` present → one issue
- `arguments: []` → zero

**`PlaceholderConsistencyValidator`**:
- Body uses `$path`, `arguments: [path]` → zero issues
- Body uses `$path` + `$depth`, `arguments: [path]` → one issue for `$depth`
- Body uses `$ARGUMENTS` with `arguments: []` → zero issues
- Body uses `${OBELIX_SKILL_DIR}` → zero issues (meta, not an arg)
- Body uses `${OBELIX_SESSION_ID}` → zero issues
- Body with no placeholders → zero issues
- Edge: body uses `$foo_bar` → treated as single placeholder name

**`ContextValueValidator`**:
- `inline` → zero
- `fork` → zero
- Missing → zero (defaults to `inline`)
- Garbage → one issue

**`BodyNonEmptyValidator`**:
- Normal body → zero
- Empty string → one issue
- Whitespace only → one issue

**`run_validators`**:
- Zero validators → zero issues always
- Custom validator returning 2 issues → both in result
- Collect-all invariant: a validator that raises is treated as a bug (propagates) — validators must return issues, not raise

#### `substitution.py` — pure function
- `$ARGUMENTS` only, no declared args → flat replacement
- Single named arg, provided → replaced
- Single named arg, missing (empty args) → replaced with `""`
- Multiple named args, partial provided → unprovided ones empty
- Named args with quoted value: `args='"my file" 3'` → `$path="my file"`, `$depth=3`
- Args count exceeds declared → `SkillInvocationError`
- `${OBELIX_SKILL_DIR}` replaced with path
- `${OBELIX_SESSION_ID}` replaced with session id
- Body has a placeholder that is also a prefix of another: `$pa` and `$path` coexist → order of replacement documented, no accidental prefix replacement (test with both orderings)
- Body has no placeholders → returned verbatim
- Unicode in args → preserved
- Property-based: any body + any args shlex-parseable → substitution is deterministic and idempotent on already-substituted text (second call is a no-op on meta placeholders)

#### `reporting.py`
- Empty issues list → does not raise
- Single issue → correct format (path, field, message)
- Multiple issues same file → grouped under one path header
- Multiple files → each path its own section, sorted alphabetically (deterministic output for snapshot tests)
- Issue without `line` → no line rendered
- Issue with `file_path=None` (MCP) → renders as `[mcp]` section

#### `manager.py` — `SkillManager`
- Zero providers → empty listing, `load` returns None
- Single provider with 3 skills → all exposed
- Two providers, disjoint names → union
- Two providers, overlapping name → first provider's skill wins (precedence documented), warning logged
- `format_listing(budget)`: under budget → full descriptions
- `format_listing(budget)`: slightly over → descriptions truncated uniformly, names intact
- `format_listing(budget)`: extreme under → names-only
- `format_listing(budget=0)` → empty string, no crash
- `load(name)` for known/unknown
- Stability: `list_all()` returns skills in deterministic order (same across calls)

#### `skill.py` — dataclass semantics
- `Skill` is frozen (mutation raises)
- `Skill.from_candidate(candidate)` preserves all fields
- Equality by value
- Hashable (for set/dict usage in dedup)

#### `filesystem.py` — `FilesystemSkillProvider`
- Single file `.md` path → one skill
- Directory with `SKILL.md` → one skill
- Directory without `SKILL.md` but with subdirs containing `SKILL.md` → N skills
- Directory with no skills → empty list
- Nonexistent path → one `SkillIssue`, aggregated in `SkillValidationError`
- Symlink handling (POSIX only): symlinked dir resolved via realpath
- Mix of valid and invalid files in one discovery → valid ones returned, invalid ones aggregated into a single `SkillValidationError` raised at end
- Multi-file validation: if 3 files each have 2 issues, raised error contains 6 issues

#### `mcp.py` — `MCPSkillProvider`
- Fake `MCPManager` exposing a well-formed prompt → one skill discovered, name prefixed `mcp__<server>__<name>`
- Fake manager with malformed prompt → skill skipped, debug logged, discovery continues
- Fake manager not connected → empty discovery, no crash

#### `skill_tool.py` — `SkillTool`
Tested against a stub agent exposing the minimal `BaseAgent` surface (`on()`, `_unregister_hook()`, `conversation_history`, `registered_tools`).

- `system_prompt_fragment()` with 0 skills → empty string
- `system_prompt_fragment()` with 3 skills, default budget → contains all names + descriptions
- `system_prompt_fragment()` with huge descriptions, small budget → truncated
- Invoke unknown skill → `ToolResult(error=...)` with list of available
- Invoke inline skill → body returned with placeholders substituted, `Skill` marked active
- Invoke same skill twice in one query → second returns "already active"
- Invoke skill with hooks → hooks registered on agent; fire on emit; removed on `QUERY_END` emit
- Invoke skill with `context: fork` → calls `SubAgentWrapper` with body as system_message, returns only last AssistantMessage
- Fork skill hooks → registered on **sub-agent**, not parent
- Invoke skill with too many args → `ToolResult(error=...)` with expected vs actual count
- Skill body uses `${OBELIX_SESSION_ID}` with no tracer → UUID stable across invocations of same agent instance
- Skill body uses `${OBELIX_SESSION_ID}` with tracer → uses tracer session id

#### `base_agent.py` delta
- `BaseAgent(skills_config=None)` → no skills manager, no SkillTool registered, zero behavior change
- `BaseAgent(skills_config="./skills/valid")` → `SkillManager` constructed, `SkillTool` registered, listing in system_message
- `BaseAgent(skills_config="./skills/broken")` → `SkillValidationError` at `__init__`
- `BaseAgent(skills_config=["./a", "./b"])` → both paths loaded
- `BaseAgent(skills_config="./skills/", mcp_config=...)` → both sources merged in listing (after MCP connect)

### Combinatorial / property tests

Combinations grow fast with skills; these cover the matrix rather than individual points.

| Combination | Property tested |
|-------------|-----------------|
| `{inline, fork}` × `{no hooks, 1 hook, 8 hooks}` | Invocation and hook lifecycle in all permutations |
| `{0, 1, 3} args declared` × `{0, 1, 3, N} args supplied` × `{quoted, unquoted}` | Substitution invariants, including error on overflow |
| `{0, 1, 10, 100} skills` × `{small, medium, huge} budget` | Listing truncation algorithm never exceeds budget, always preserves names, deterministic |
| `{FS only, MCP only, both, both with collision}` | Provider aggregation and precedence rules |
| `{single issue, multi-issue same file, multi-file}` | Collect-all correctness: issue count matches exactly |
| Idempotence of substitution on already-substituted body | Second call with same args produces same result |

Property-based candidates (`hypothesis`): substitution (arbitrary body + args), listing formatter (arbitrary skills + budget), validator chain (arbitrary candidate).

### Integration tests — `tests/integration/skill/`

Full agent loop, real file IO in `tmp_path`, stub LLM provider replying deterministically.

1. **Happy path inline**: agent with 1 skill → LLM stub emits `Skill` tool call → skill body appears in conversation → loop continues → final response returned
2. **Fork execution**: parent + fork skill → sub-agent spawns → sub-agent history isolated → parent receives single ToolResult
3. **Hook firing during skill**: skill with `on_tool_error` → another tool fails mid-skill → injected message appears in conversation → hook removed at `QUERY_END`
4. **MCP skill round-trip**: fake MCP server exposes a prompt → agent with `mcp_config` + empty `skills_config` → skill discovered → LLM invokes it → body from MCP returned
5. **Skill + MCP collision**: same name in filesystem and MCP → filesystem wins, warning captured
6. **Validation failure boots clean**: agent construction with broken skill path → `SkillValidationError` with every issue → agent object never materializes (no partial state)
7. **Multi-query reuse**: same agent runs two queries, each invoking the same skill with different args → no hook leak between queries, no listing drift

### Regression guardrails

- **Snapshot test** for `system_prompt_fragment()` output with a fixed skill set + budget → change in truncation algorithm requires explicit snapshot update (review gate).
- **Snapshot test** for `SkillValidationError` rendering with a canonical multi-issue corpus.
- **Import boundary test**: a static check (`grimp` or equivalent) asserts the coupling rules from the Architecture section — e.g. `manager.py` does not import `yaml`, `skill.py` does not import from `core/skill/*`.

---

## Resolved design points (for traceability)

1. **Session ID for `${OBELIX_SESSION_ID}` when no tracer**: per agent instance (stable across query chain), fallback UUID.
2. **Fork execution and Tool registration**: SkillTool is included in fork's tool list → sub-agents can compose other skills.
3. **Skill listing refresh on late MCP connect**: v1 computes the listing once at `register_tool` time. When `mcp_config` is present, MCP is connected synchronously before the first `execute_query` so all skills are discoverable.

---

## Migration and compatibility

- Zero breaking change: existing agents without `skills_config` work exactly as before.
- Existing `register_tool` + `system_prompt_fragment` auto-injection pattern is reused; no refactor.
- Existing `MCPManager` requires one addition: `list_prompts()` method. Non-breaking (additive).
- Existing `hooks.py` requires one addition: `_unregister_hook` private method. Non-breaking.