# BashTool Guide

BashTool gives agents the ability to execute shell commands. It is the most powerful — and most dangerous — tool in Obelix. This guide covers how it works, how to configure it safely, and what to watch out for.

---

## Why a Dedicated Guide

An LLM with shell access can do almost anything: read files, install packages, modify configurations, delete data, exfiltrate secrets. BashTool is designed with this risk in mind:

- **Two execution modes** — server-side (local) or client-side (deferred) — so you choose who runs the commands
- **Permission policies** — the CLI client can require confirmation before every execution
- **Mandatory description field** — the LLM must explain what each command does before executing
- **Timeout and output limits** — prevent runaway processes and context window bloat
- **System prompt injection** — the LLM knows the exact shell and platform, reducing broken commands

But none of these are a security boundary. BashTool does **not** sandbox commands. If the executor has filesystem access, so does the LLM. Treat it accordingly.

---

## Two Execution Modes

### Deferred Mode (default)

The LLM generates the command, but **does not execute it**. The agent loop stops and delegates execution to the client (A2A client, CLI runner, or your custom code).

```python
from obelix.plugins.builtin import BashTool

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
    tools=[BashTool()],  # no executor = deferred
)
```

Flow:
```
LLM generates: bash(command="git status", description="Check repo status")
    ↓
Agent loop stops → yields StreamEvent(deferred_tool_calls=[...])
    ↓
Client receives the command and decides whether to execute
    ↓
Client sends result back → agent resumes
```

**When to use**: production environments, multi-tenant setups, any case where you don't trust the LLM to run arbitrary commands unsupervised.

### Local Mode

The command executes **on the server** via `LocalShellExecutor`. The agent loop continues without stopping.

```python
from obelix.plugins.builtin import BashTool
from obelix.adapters.outbound.shell import LocalShellExecutor

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
    tools=[BashTool(executor=LocalShellExecutor())],
)
```

Flow:
```
LLM generates: bash(command="git status", description="Check repo status")
    ↓
LocalShellExecutor runs the command as a subprocess
    ↓
Result returned to LLM → loop continues
```

**When to use**: development, single-user local agents, sandboxed environments (Docker, VMs), CI/CD pipelines.

### Comparison

| | Deferred | Local |
|---|---|---|
| Who executes | Client | Server |
| Human review possible | Yes (always) | No (runs immediately) |
| `is_deferred` | `True` | `False` |
| `system_prompt_fragment()` | Returns `None` | Returns shell environment block |
| Requires `--extra serve` | Yes (for A2A) | No |
| Risk level | Low (client controls) | **High** (LLM controls) |

---

## Security Considerations

### What BashTool CAN Do (with local executor)

- Read any file the process user can access
- Write, move, or delete files
- Install or remove packages
- Make network requests (`curl`, `wget`, `ssh`)
- Start or kill processes
- Access environment variables (including secrets like API keys)
- Modify system configuration

### What BashTool CANNOT Do

- Escalate beyond the process user's permissions (no automatic `sudo`)
- Break out of OS-level sandboxes (Docker, VMs, cgroups)
- Bypass filesystem permissions

### Mitigation Strategies

**1. Use deferred mode in production**

The safest approach. The LLM proposes commands, a human (or automated policy) approves them.

**2. Run in a container or VM**

If you need local mode, run the agent inside a sandboxed environment:

```python
from obelix.ports.outbound.shell_executor import AbstractShellExecutor

class DockerExecutor(AbstractShellExecutor):
    """Execute commands inside a disposable container."""

    async def execute(self, command, timeout=120, working_directory=None):
        # docker exec ... or docker run --rm ...
        ...

agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[BashTool(executor=DockerExecutor(image="python:3.13-slim"))],
)
```

**3. Use hooks to block dangerous commands**

```python
from obelix.core.agent.hooks import AgentEvent, HookDecision

BLOCKED_PATTERNS = ["rm -rf", "mkfs", "dd if=", "> /dev/sd", "chmod 777", "curl | sh"]

agent.on(AgentEvent.BEFORE_TOOL_EXECUTION).when(
    lambda s: s.tool_call.name == "bash" and any(
        p in s.tool_call.arguments.get("command", "")
        for p in BLOCKED_PATTERNS
    )
).handle(decision=HookDecision.FAIL)
```

**4. Limit iterations and timeout**

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[BashTool(executor=LocalShellExecutor())],
    max_iterations=5,  # fewer chances to cause damage
)
```

BashTool enforces `timeout` per command (1-600 seconds, default 120). Processes that exceed it are killed.

**5. Restrict the system message**

Be explicit about what the agent should and should not do:

```python
system_message = (
    "You are a read-only system inspector. "
    "You can run commands to inspect files, check status, and gather information. "
    "NEVER modify, delete, or create files. "
    "NEVER install packages or change system configuration. "
    "NEVER run commands that require sudo or elevated privileges."
)
```

---

## Permission Policies (CLI Client)

When using the CLI client (`CLIClient`) with A2A, the `BashHandler` supports three permission policies:

### ALWAYS_ASK (default)

Every command is shown to the user with a `[Y/n]` prompt. The user must explicitly approve each execution.

```python
from obelix.adapters.inbound.a2a.client.handlers import default_dispatcher, PermissionPolicy

dispatcher = default_dispatcher(bash_permission=PermissionPolicy.ALWAYS_ASK)
client = CLIClient(dispatcher=dispatcher)
```

Terminal output:
```
╭─ bash ─────────────────────────────────────────╮
│  Check the current git status                  │
│                                                │
│    $ git status                                │
│                                                │
│    timeout: 120s                               │
╰────────────────────────────────────────────────╯
Execute? [Y/n] _
```

### AUTO_APPROVE

Commands execute immediately without confirmation. Use only in trusted/sandboxed environments.

```python
dispatcher = default_dispatcher(bash_permission=PermissionPolicy.AUTO_APPROVE)
```

### ALWAYS_DENY

All commands are rejected. The agent receives `exit_code: -1` with `stderr: "Execution denied by user"`. Useful for testing agent behavior when commands fail.

```python
dispatcher = default_dispatcher(bash_permission=PermissionPolicy.ALWAYS_DENY)
```

### Custom Permission Logic

For fine-grained control, subclass `BashHandler`:

```python
from obelix.adapters.inbound.a2a.client.handlers import BashHandler, PermissionPolicy

class SafeBashHandler(BashHandler):
    """Auto-approve read-only commands, ask for everything else."""

    SAFE_PREFIXES = ("ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "pwd", "git status", "git log", "git diff")

    def __init__(self):
        super().__init__(permission=PermissionPolicy.ALWAYS_ASK)

    async def prompt_response(self, args, session):
        command = args.get("command", "")
        if any(command.strip().startswith(p) for p in self.SAFE_PREFIXES):
            # Auto-approve read-only commands
            executor = await self._get_executor()
            result = await executor.execute(
                command=command,
                timeout=args.get("timeout", 120),
                working_directory=args.get("working_directory"),
            )
            import json
            return json.dumps(result)

        # Ask for everything else
        return await super().prompt_response(args, session)
```

Register it on the dispatcher:

```python
from obelix.adapters.inbound.a2a.client.handlers import HandlerDispatcher, RequestInputHandler

dispatcher = HandlerDispatcher([
    RequestInputHandler(),
    SafeBashHandler(),
])
client = CLIClient(dispatcher=dispatcher)
```

---

## System Prompt Fragment

In local mode, `BashTool` injects a `## Shell Environment` block into the agent's system message. This happens automatically when the tool is registered — no manual setup needed.

Example of what gets injected (Windows with Git Bash):

```
## Shell Environment
- Platform: Windows
- OS Version: Windows-11-10.0.26200-SP0
- Shell: bash (use Unix shell syntax, not Windows -- e.g., /dev/null not NUL, forward slashes in paths)
- Working directory: /c/Users/GLoverde/Projects
- Drive mounts: C: -> /c/, D: -> /d/
- Use Unix-style paths with forward slashes (e.g. /c/Users/... not C:\Users\...)
```

Example (Linux):

```
## Shell Environment
- Platform: Linux
- Shell: bash
- Working directory: /home/user/projects
```

This prevents the LLM from generating Windows `dir` commands on a Unix shell, or using `C:\` paths in Git Bash.

In deferred mode (no executor), `system_prompt_fragment()` returns `None` — the server doesn't know the client's environment.

---

## LocalShellExecutor Details

### Shell Detection

On construction, `LocalShellExecutor` detects the best available shell:

| Platform | Priority |
|----------|----------|
| Windows | Git Bash > WSL bash > cmd.exe |
| Unix | bash > zsh > sh |

Git Bash is preferred on Windows because it uses MSYS2 path translation (`C:/` → `/c/`). WSL bash uses `/mnt/c/` paths which are incompatible.

Override with an explicit shell:

```python
executor = LocalShellExecutor(shell="/bin/zsh")
executor = LocalShellExecutor(shell="cmd.exe")  # Windows-only, loses Unix features
```

### Environment Probe

At construction, the executor runs a one-shot probe (`~100ms`) to discover:
- Current working directory (`pwd`)
- Home directory (`$HOME`)
- Drive mounts (Windows only: `mount | grep "^[A-Z]:"`)

This data populates `shell_info` and drives `system_prompt_fragment()`.

### MSYSTEM on Windows

Git Bash requires `MSYSTEM=MINGW64` for correct path translation. The executor handles this automatically:

- Not set → sets `MSYSTEM=MINGW64`
- Already `MINGW64` → no change
- Set to something else (e.g. `MSYS`) → raises `ShellEnvironmentError` with instructions

### Output Truncation

stdout and stderr are truncated at **50,000 characters** to prevent bloating the LLM context window. Truncated output gets a footer:

```
... [output truncated at 50000 chars — 123456 total]
```

### Timeout

Commands that exceed the timeout are killed (`process.kill()`). The result:

```json
{"stdout": "", "stderr": "Command timed out after 120s and was killed.", "exit_code": -1}
```

---

## Input and Output Schema

### Input (what the LLM provides)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `command` | str | Yes | Shell command to execute |
| `description` | str | Yes | Human-readable explanation (for audit/review) |
| `timeout` | int | No | Max seconds (1-600, default 120) |
| `working_directory` | str | No | Working directory for the command |

The `description` field is enforced by the schema — the LLM cannot call BashTool without explaining what the command does. This is critical for the deferred mode review flow.

### Output (what the executor returns)

| Field | Type | Description |
|-------|------|-------------|
| `stdout` | str | Standard output (truncated at 50K chars) |
| `stderr` | str | Standard error (truncated at 50K chars) |
| `exit_code` | int | 0 = success, non-zero = failure, -1 = timeout/denied |

---

## Examples

### Minimal A2A Server with BashTool

```python
# examples/bash_server.py
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.plugins.builtin import BashTool

class BashAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a helpful assistant with shell access.",
            **kwargs,
        )
        self.register_tool(BashTool())  # deferred mode

factory = AgentFactory()
factory.register(name="bash_agent", cls=BashAgent)
factory.a2a_serve("bash_agent", port=8002)
```

### CLI Client with Permission Control

```python
# examples/cli_client.py
from obelix.adapters.inbound.a2a.client import CLIClient
from obelix.adapters.inbound.a2a.client.handlers import default_dispatcher, PermissionPolicy

dispatcher = default_dispatcher(bash_permission=PermissionPolicy.ALWAYS_ASK)
client = CLIClient(dispatcher=dispatcher)

import asyncio
asyncio.run(client.run(["http://localhost:8002"]))
```

### Local Agent (no A2A, no confirmation)

```python
from obelix.adapters.outbound.shell import LocalShellExecutor
from obelix.core.agent import BaseAgent
from obelix.plugins.builtin import BashTool

agent = BaseAgent(
    system_message=(
        "You are a system inspector. Read files and check status. "
        "NEVER modify or delete anything."
    ),
    provider=provider,
    tools=[BashTool(executor=LocalShellExecutor())],
    max_iterations=5,
)

response = agent.execute_query("What Python version is installed?")
print(response.content)
```

---

## Checklist: Before Deploying BashTool

- [ ] **Decide the mode**: deferred (human review) or local (automated)?
- [ ] **If local**: is the agent running in a sandbox (container, VM, restricted user)?
- [ ] **If local**: have you restricted the system message to prevent destructive commands?
- [ ] **If local**: consider adding BEFORE_TOOL_EXECUTION hooks to block dangerous patterns
- [ ] **If deferred**: have you configured the appropriate `PermissionPolicy` on the client?
- [ ] **Timeout**: is the default 120s appropriate, or should you lower it?
- [ ] **Iterations**: have you set `max_iterations` to a reasonable limit?
- [ ] **Secrets**: are there environment variables (API keys, tokens) the LLM could read via `env` or `printenv`?
- [ ] **Network**: can the shell make outbound network requests? Should it?

---

## See Also

- [Tools Guide](tools.md) — tool system overview, OutputSchema, system_prompt_fragment
- [BaseAgent Guide](base_agent.md) — hooks, streaming, deferred tools
- [Agent Factory](agent_factory.md) — a2a_serve(), tracer integration
