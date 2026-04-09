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
4. **LLM provider registered** in the gateway (one-time setup):

   **Option A** — auto-detect API key from environment + add model config:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   openshell provider create --name anthropic --type anthropic --from-existing
   openshell provider update anthropic --config LITELLM_MODEL=anthropic/claude-haiku-4-5-20251001
   ```
   `--from-existing` reads exported env vars and tool config (e.g. `~/.config/claude/`), NOT `.env` files.
   `--from-existing` and `--credential` are mutually exclusive, so the model is added via `update`.
   The model uses `--config` (not `--credential`) because it's a configuration value, not a secret.
   `--credential` values are injected as resolver references, `--config` values are injected as plain env vars.

   **Option B** — pass the API key explicitly + model config (no export needed):
   ```bash
   openshell provider create --name anthropic --type anthropic --credential ANTHROPIC_API_KEY=sk-ant-...
   openshell provider update anthropic --config LITELLM_MODEL=anthropic/claude-haiku-4-5-20251001
   ```

   Both options store the API key as a secret and the model as config in the gateway.
   The sandbox receives them as environment variables at runtime — no `.env` files needed.

## Run

```bash
uv run python examples/deploy_demo/deploy.py
```

This will:
1. Verify that the required providers exist in the OpenShell gateway
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

To switch from Anthropic to another provider (e.g. OpenAI):

**Step 1** — Register the new provider in the gateway:
```bash
export OPENAI_API_KEY=sk-...
openshell provider create --name openai --type openai --from-existing
openshell provider update openai --config LITELLM_MODEL=openai/gpt-4o
```

`--from-existing` picks up credentials from your local environment (e.g. `OPENAI_API_KEY`).
The model is added separately via `update` using [LiteLLM's naming convention](https://docs.litellm.ai/docs/providers).

**Step 2** — Update `deploy.py` to reference the new provider:
```python
factory.a2a_openshell_deploy(
    "sandbox_bash",
    providers=["openai"],
    ...
)
```

**Step 3** — Update `policy.yaml` to allow the new API endpoint:
```yaml
network_policies:
  openai_api:
    name: openai-api
    endpoints:
      - host: api.openai.com
        port: 443
        protocol: rest
        enforcement: enforce
        access: read-write
    binaries:
      - path: /usr/bin/python*
      - path: /usr/local/bin/python*
```

No changes needed in `serve.py` — it reads `LITELLM_MODEL` from the environment
and LiteLLM routes to the correct provider automatically.

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
