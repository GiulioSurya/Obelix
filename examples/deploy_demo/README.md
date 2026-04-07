# Deploy Demo — Sandboxed BashTool Agent

This example deploys an Obelix agent with shell access inside an OpenShell sandbox. The agent can execute bash commands, but all commands are restricted by a kernel-level security policy (filesystem, network, processes).

## Architecture

```
Windows / Host                          OpenShell Sandbox (K8s pod)
+------------------+                    +---------------------------+
| cli_client.py    | -- A2A / HTTP ---> | serve.py (A2A server)     |
| localhost:8002   |    SSH tunnel      | SandboxedBashAgent        |
+------------------+                    | BashTool + LocalShellExec |
                                        +---------------------------+
                                        | OpenShell Policy Engine   |
                                        | (landlock, netns, proxy)  |
                                        +---------------------------+
```

## Prerequisites

1. **Docker** running
2. **OpenShell gateway** started:
   ```bash
   openshell gateway start
   ```
3. **Dependencies** installed:
   ```bash
   uv sync --extra litellm --extra serve --extra openshell
   ```
4. **Anthropic API key** in `examples/shell-demo/.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Run

```bash
uv run python examples/deploy_demo/deploy.py
```

This will:
1. Register the Anthropic provider in the OpenShell gateway
2. Build a Docker image with the project code and targeted dependencies
3. Create a sandbox with the security policy (`policy.yaml`)
4. Start the A2A server inside the sandbox
5. Forward port 8002 to localhost

The deploy blocks until Ctrl+C. On exit, the sandbox is destroyed automatically.

## Connect

From another terminal:
```bash
uv run python examples/cli_client.py http://localhost:8002
```

## Files

| File | Description |
|---|---|
| `deploy.py` | Entrypoint — configures and launches the deployer |
| `serve.py` | Runs **inside** the sandbox — sets up the agent and A2A server |
| `policy.yaml` | OpenShell security policy (filesystem, network, process rules) |

## Policy Overview

The `policy.yaml` defines four security areas:

### Filesystem
- **Read-only**: `/usr`, `/bin`, `/etc`, `/lib`, `/proc`, `/app` (workdir)
- **Read-write**: `/sandbox`, `/tmp`
- Everything else is denied by landlock

### Process
- Runs as unprivileged user `sandbox` (no root)

### Network (default deny-all, whitelist only)
- `api.anthropic.com:443` — LLM API calls
- `host.docker.internal:8100` — Obelix tracer (optional)
- `host.docker.internal:8010` — A2A client webhook (push notifications)
- `*.dns.google:53/443` — DNS resolution
- Everything else is blocked (no PyPI, no GitHub, no arbitrary HTTP)

### Network policy types
- **L7 (protocol: rest)** — for HTTPS endpoints. Enables HTTP method/path inspection.
- **L4 (no protocol)** — for plain HTTP endpoints. TCP passthrough only.

## Testing the Policy

Ask the agent to run commands that test each security boundary:

```
# Filesystem (should work)
"Write 'hello' to /tmp/test.txt and read it back"

# Filesystem (should fail)
"Try to write a file to /etc/test"

# Process isolation
"Run whoami and id"

# Network allowed
"Curl https://api.anthropic.com/v1/messages and show the status code"

# Network denied
"Curl https://www.google.com with a 5 second timeout"

# Landlock enforcement
"Try to create a directory at /opt/test"
```

## Customization

### Using a different LLM provider

Edit `serve.py` to use a different provider, and update `deploy.py`:

```python
factory.a2a_openshell_deploy(
    "sandbox_bash",
    providers=["openai"],       # gateway credentials
    extras=["openai"],          # Docker build dependencies (not needed — openai is a base dep)
    ...
)
```

### Adding network access

Edit `policy.yaml` to add endpoints. Network policies are hot-reloadable:

```bash
openshell policy set <sandbox-name> --policy policy.yaml --wait
```

### Monitoring

```bash
# Sandbox supervisor logs
openshell logs <sandbox-name> --tail

# A2A server logs (inside the sandbox)
ssh -o "ProxyCommand=openshell ssh-proxy --gateway-name openshell --name <sandbox>" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    sandbox@openshell-<sandbox> "cat /tmp/a2a-server.log"
```
