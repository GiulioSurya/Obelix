# Shell Demo — BashTool + OpenShell Sandbox

An A2A agent that executes shell commands inside an OpenShell sandbox, with kernel-level controls on filesystem, network, and processes enforced by `policy.yaml`.

## Platform Requirements

| Component | Where it runs | OS requirement |
|-----------|--------------|----------------|
| **Server** (Docker + OpenShell) | `docker compose up` | Linux or macOS (or WSL2 on Windows) — OpenShell has no Windows wheel |
| **CLI client** | `uv run python examples/cli_client.py` | Anywhere (Windows, macOS, Linux) |

## Prerequisites

- Docker Desktop running
- OpenShell gateway started (once, on the Linux/macOS/WSL host):
  ```bash
  openshell gateway start
  ```
- `.env` file in this directory (copy `.env.example`, fill in your key)

## Start the Server

From the project root on Linux/macOS/WSL:

```bash
cd examples/shell-demo
docker compose up --build
```

Server starts on `http://localhost:8002`.

## Connect with the CLI

From any terminal, project root:

```bash
uv run python examples/cli_client.py http://localhost:8002
```

The CLI reads `examples/shell-demo/.env` automatically for config.

### How results arrive: webhook vs polling

When you send a message, the server processes it asynchronously. The CLI needs to know when the result is ready. Two mechanisms:

1. **Webhook (push)** — the CLI starts a small HTTP server on your machine. When the task finishes, the Docker container POSTs the result to that server. This is instant but requires the container to reach your machine over the network.

2. **Polling (fallback)** — if the webhook fails (container can't reach your host), the CLI automatically polls the server every 2 seconds asking "is my task done?". You'll see a yellow warning when this happens. Results still arrive, just slightly slower.

### Webhook setup (optional, for instant results)

Uncomment these two lines in `.env`:

```
OBELIX_WEBHOOK_HOST=host.docker.internal
OBELIX_WEBHOOK_PORT=8010
```

`host.docker.internal` is how Docker containers refer to the host machine. Port `8010` is a fixed port so you can open it in the firewall once.

On Windows, open the port (PowerShell as admin, one-time):

```powershell
New-NetFirewallRule -DisplayName "Obelix Webhook" -Direction Inbound -Protocol TCP -LocalPort 8010 -Action Allow
```

Without these settings, polling kicks in automatically — no setup needed.

## Policy

`policy.yaml` is mounted read-only into the container and hot-reloaded every 60s — no restart needed after edits.

| Domain | Rule |
|--------|------|
| **Filesystem** | read-only on system dirs, read-write on `/sandbox` and `/tmp` |
| **Process** | unprivileged `sandbox` user, no root |
| **Network** | only Anthropic API + tracer, everything else blocked |

## What to Test

1. `cat /etc/passwd` — allowed (read-only) vs `echo x > /etc/test` — blocked
2. `whoami` — should return `sandbox`, not `root`
3. `curl https://api.anthropic.com` — allowed vs `curl https://google.com` — blocked
4. Edit `policy.yaml`, wait 60s, verify the new rule takes effect