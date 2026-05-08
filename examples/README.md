# Examples

Five A2A agent servers plus an interactive CLI client to talk to them. Each server runs on its own port so you can start them in parallel. The fifth (orchestrator) is itself an A2A client of the other four.

## Setup

```bash
uv sync --extra litellm --extra serve
export API_KEY=sk-...
```

## Servers

| File | Port | What it runs |
|---|---|---|
| [`dev_workflow_server.py`](dev_workflow_server.py) | **8001** | 3-agent pipeline — Reviewer + Commit + Summary, orchestrated by a Coordinator. Uses skills (`code-review`, `security-check`, `commit-writer`). |
| [`deploy_demo/`](deploy_demo/README.md) | **8002** | Sandboxed `BashTool` agent deployed inside an OpenShell sandbox with kernel-level policy. See the [deploy_demo README](deploy_demo/README.md). |
| [`bash_server.py`](bash_server.py) | **8003** | Single agent with `BashTool` (local executor or deferred client-side). |
| [`mcp_playwright.py`](mcp_playwright.py) | **8004** | Browser agent powered by the Playwright MCP server (via `npx`). |
| [`orchestrator_server.py`](orchestrator_server.py) | **8005** | Orchestrator agent — itself an A2A client of the four servers above. Delegates work via `dispatch_agent` and surfaces async completions to the user. |

Start any of them:

```bash
uv run python examples/dev_workflow_server.py    # :8001
uv run python examples/deploy_demo/deploy.py     # :8002 (requires OpenShell gateway)
uv run python examples/bash_server.py            # :8003
uv run python examples/mcp_playwright.py         # :8004
uv run python examples/orchestrator_server.py    # :8005 (after starting at least one of the above)
```

## CLI client

Connects to one or more A2A servers for an interactive chat:

```bash
uv run python examples/cli_client.py http://localhost:8001
uv run python examples/cli_client.py http://localhost:8001 http://localhost:8002 http://localhost:8003
```

## Agents

### `dev_workflow_server.py` (:8001)

A Coordinator orchestrates three stateless sub-agents in order:

- **ReviewerAgent** — reads the staged diff via a `git_diff` tool, then runs the `code-review` skill (fork) and the `security-check` skill (inline).
- **CommitAgent** — receives the review findings via shared memory and produces a conventional commit message with the `commit-writer` skill (fork).
- **SummaryAgent** — assembles the final markdown report from both review and commit.

Stage some changes with `git add`, connect the CLI client to `:8001`, and ask: _"review my staged changes"_.

### `deploy_demo/` (:8002)

- **SandboxedBashAgent** — a `BashTool` agent running **inside** an OpenShell sandbox. Commands execute as subprocesses but the OpenShell Policy Engine enforces filesystem, network and process restrictions at the kernel level; LLM calls go through a gateway-managed inference proxy so the sandbox never sees the API key. See [`deploy_demo/README.md`](deploy_demo/README.md) for policy, setup and customization.

### `bash_server.py` (:8003)

- **BashAgent** — a single agent with shell access via `BashTool`. Toggle `LOCAL_EXECUTOR` in the file to switch between local execution on the server and deferred execution where the A2A client runs the command and returns the result.

### `mcp_playwright.py` (:8004)

- **BrowserAgent** — a web-browsing agent that connects to the Playwright MCP server over stdio (`npx @playwright/mcp@latest`). It discovers browser tools (navigate, click, type, screenshot, …) at startup and uses them to fulfill user queries.

### `orchestrator_server.py` (:8005)

- **OrchestratorAgent** — a pure orchestrator with no local tools. At startup it fetches the AgentCards of the four other servers (those that are running — unreachable ones are warned-and-skipped, not fatal) and the system prompt is auto-enriched with their `name`/`description`/`skills`. The five outbound A2A tools (`dispatch_agent`, `respond_to_remote`, `task_list`, `task_get`, `task_stop`) are auto-injected into the agent.

When the user asks something through the CLI client, the orchestrator's LLM picks the right remote and calls `dispatch_agent(name, query)`. The call returns immediately with a `task_id`; the orchestrator ends its turn. Push notifications from the remote arrive on the orchestrator's `/webhook` (mounted on the same uvicorn as port 8005), are queued per-context, and surface as `<remote_task_update>` user-role messages at the start of the next turn — that's when the LLM reads the result and reports back to the user.

```bash
# Two-agent smoke test (minimum):
API_KEY=sk-... uv run python examples/dev_workflow_server.py    # :8001
API_KEY=sk-... uv run python examples/orchestrator_server.py    # :8005
uv run python examples/cli_client.py http://localhost:8005
> review my staged changes
```

The orchestrator dispatches to dev_workflow on :8001, ends its turn, and you'll see the review report come back as a notification on the next turn (typically a few seconds later when the dev_workflow pipeline finishes). To exercise multi-remote, start `bash_server.py` (with `LOCAL_EXECUTOR = True` for a clean smoke) and `mcp_playwright.py` and ask the orchestrator something that requires multiple skills — e.g. _"check the staged diff and run `git status` to confirm what's pending"_.
