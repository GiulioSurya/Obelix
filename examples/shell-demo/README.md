# Shell Demo — BashTool + OpenShell Sandbox

A2A server with BashTool that executes commands inside an OpenShell sandbox
with policy-governed filesystem, network, and process controls.

## Prerequisites

- Docker Desktop running
- OpenShell gateway started on the host (one-time):
  ```bash
  openshell gateway start
  ```
- `.env` file in this directory with your API key:
  ```
  API_KEY=sk-...
  ```

## How to Run

```bash
cd examples/shell-demo
docker compose up --build
```

The server starts on `http://localhost:8002`. Connect with the CLI client:

```bash
uv run python examples/cli_client.py --url http://localhost:8002
```

## Policy

`policy.yaml` covers all 4 OpenShell security domains and is mounted read-only
into the container. Changes to the file are watched and hot-reloaded
automatically (no restart needed).

| Domain | What it does |
|--------|-------------|
| **Filesystem** | read-only on system dirs, read-write only on `/sandbox` and `/tmp` |
| **Process** | runs as unprivileged `sandbox` user (no root) |
| **Network** | allows only Anthropic API + tracer backend, everything else blocked |
| **Inference** | configure at runtime via `openshell inference set` |

## What to Test

1. **Filesystem**: ask the agent to `cat /etc/passwd` (allowed, read-only) vs `echo x > /etc/test` (blocked)
2. **Process**: ask `whoami` — should be `sandbox`, not `root`
3. **Network**: ask to `curl https://api.anthropic.com` (allowed) vs `curl https://google.com` (blocked)
4. **Inference**: configure with `openshell inference set --provider anthropic --model ...`
