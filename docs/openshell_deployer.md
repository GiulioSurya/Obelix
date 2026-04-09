# OpenShell Deployer

The OpenShell Deployer runs an Obelix agent **inside an OpenShell sandbox**, exposing it as an A2A server. From the outside it is indistinguishable from a server started with `a2a_serve()` — same protocol, same Agent Card, same CLI client.

The difference is that the agent runs in an isolated environment with kernel-level security policies (filesystem, network, processes).

---

## Comparison with a2a_serve()

| | a2a_serve() | a2a_openshell_deploy() |
|---|---|---|
| Where it runs | Local process | OpenShell sandbox (K8s pod) |
| Security | OS permissions only | Kernel-level policy (landlock, network namespaces) |
| BashTool | Any executor | LocalShellExecutor (protected by policy) |
| API keys | In your .env / environment | Injected by the gateway inference proxy at network level |
| Requirements | `uv sync --extra serve` | Docker + OpenShell gateway |
| Client access | `localhost:<port>` | `localhost:<port>` (via SSH tunnel) |

---

## Prerequisites

- **Docker** running (Docker Desktop on Windows/Mac, Docker Engine on Linux)
- **OpenShell gateway** started: `openshell gateway start`
- **Dependencies**: `uv sync --extra serve --extra openshell`

---

## Quick Start

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()

factory.a2a_openshell_deploy(
    "my_agent",
    port=8002,
    entrypoint="myapp.serve",       # Python module that calls a2a_serve()
    policy="policy.yaml",           # OpenShell security policy
    providers=["anthropic"],        # LLM credentials (auto-detected from local env)
    extras=["litellm"],             # additional pyproject extras for the Docker build
)
```

Then connect normally:
```bash
uv run python examples/cli_client.py http://localhost:8002
```

The deployer blocks waiting for Ctrl+C. On exit, the sandbox is destroyed automatically.

---

## How It Works

```
1. Validate        Check SDK, CLI, gateway reachability, provider names
2. Providers       Verify that required providers exist in the gateway
                   (must be created beforehand via openshell provider create)
3. Image build     Docker build with project code + targeted dependencies
                   (or use a custom Dockerfile / pre-built image)
4. Sandbox create  openshell sandbox create --from <image> --provider ... --policy ...
                   (creates K8s pod with security policy)
5. Server start    Exec /app/.venv/bin/python -m <entrypoint> inside the sandbox
                   (starts uvicorn + FastAPI + A2A protocol)
6. Port forward    openshell forward start -d <port> <sandbox>
                   (SSH tunnel makes the server reachable on localhost)
```

---

## Docker Image

The deployer supports three modes:

### 1. Auto-generated (default)

When neither `dockerfile` nor `image` is provided, the deployer generates a minimal Dockerfile:

```python
factory.a2a_openshell_deploy("my_agent", port=8002,
    entrypoint="myapp.serve",
    policy="policy.yaml",
    providers=["anthropic"],
)
```

The generated Dockerfile:
- Base image: `python:3.13-slim`
- Installs `uv`, `iproute2` (required by OpenShell for network namespaces)
- Creates a `sandbox` user (required by the policy `run_as_user`)
- Copies the project and runs `uv sync` with targeted extras
- Uses `/app/.venv/bin/python` as runtime (no network access needed at runtime)

### 2. Custom Dockerfile

For images with special requirements (system packages, extra tools):

```python
factory.a2a_openshell_deploy("my_agent", port=8002,
    dockerfile="path/to/Dockerfile",
)
```

**Important**: your custom Dockerfile must include:
- `useradd -m sandbox` (or whatever user the policy's `run_as_user` requires)
- `iproute2` package (the OpenShell supervisor needs `ip` for network namespaces)
- Use `/app/.venv/bin/python` in CMD, not `uv run` (the sandbox may not have network access to PyPI)

### 3. Pre-built image

For fast deploys or images from a registry:

```python
factory.a2a_openshell_deploy("my_agent", port=8002,
    image="my-registry.com/my-agent:latest",
)
```

---

## Provider Extras Mapping

The deployer maps provider names to `pyproject.toml` optional-dependency groups for the Docker build. Only the required extras are installed:

| Provider | Extra | Notes |
|---|---|---|
| `anthropic` | *(base dep)* | Always installed |
| `openai` | *(base dep)* | Always installed |
| `litellm` | `litellm` | |
| `oci` | `oci` | |
| `ibm_watson` | `ibm` | |
| `ollama` | `ollama` | |
| `vllm` | `vllm` | |

`serve` and `openshell` extras are always included.

If the automatic mapping is not sufficient (e.g., you use `litellm` to wrap Anthropic), pass `extras` explicitly:

```python
factory.a2a_openshell_deploy(
    "my_agent",
    providers=["anthropic"],   # for the gateway (API key injection)
    extras=["litellm"],        # for the Docker build (Python dependencies)
)
```

Unknown provider names are rejected at validation time with a clear error message.

---

## Parameters

```python
factory.a2a_openshell_deploy(
    agent: str,                          # agent name (registered or defined in entrypoint)
    *,
    port: int = 8002,                    # A2A server port
    entrypoint: str | None = None,       # Python module to run inside the sandbox
    policy: str | None = None,           # path to OpenShell policy YAML
    providers: list[str] | None = None,  # provider names for gateway credential injection
    extras: list[str] | None = None,     # additional pyproject extras for Docker build
    dockerfile: str | None = None,       # path to custom Dockerfile
    image: str | None = None,            # pre-built image reference
    gateway: str | None = None,          # gateway endpoint (default: auto-detect)
    endpoint: str | None = None,         # override Agent Card URL
    version: str = "0.1.0",             # Agent Card version
    description: str | None = None,      # Agent Card description
)
```

---

## Policy

The policy controls what the agent can do inside the sandbox. It has **static** fields (immutable after creation) and **dynamic** fields (hot-reloadable at runtime).

### Static fields

- **`filesystem_policy`** — read-only and read-write paths. `include_workdir: true` adds the container's WORKDIR as read-only.
- **`process`** — `run_as_user` / `run_as_group`. The user must exist in the image.
- **`landlock`** — kernel-level filesystem enforcement. Use `compatibility: best_effort`.

### Dynamic fields (hot-reloadable)

- **`network_policies`** — endpoint whitelist with binary matching.

### Network policy tips

- **HTTPS endpoints**: set `protocol: rest` for L7 inspection (HTTP method/path filtering)
- **Plain HTTP endpoints**: omit `protocol` for L4 passthrough (TCP only). Using `protocol: rest` on plain HTTP causes the proxy to reject FORWARD requests
- **`tls: terminate`** is deprecated — TLS termination is now automatic. Remove it from policies
- **DNS**: always include a DNS rule, or hostname resolution will fail

### Hot-reload

```bash
openshell policy set <sandbox-name> --policy new-policy.yaml --wait
```

Or from code:
```python
await deployer.update_policy("new-policy.yaml")
```

---

## BashTool Inside the Sandbox

When the agent runs inside the sandbox, BashTool uses `LocalShellExecutor` — commands execute as normal subprocesses, but the OpenShell Policy Engine intercepts them at the kernel level:

```
LLM generates: bash(command="rm -rf /etc")
    -> LocalShellExecutor -> subprocess.run("rm -rf /etc")
        -> Policy Engine (kernel): /etc is read-only -> EPERM
            -> exit_code: 1, stderr: "Permission denied"
```

The agent does not know it is in a sandbox. It tries, fails, handles the error.

---

## LLM Credentials and Inference

API keys never reach the sandbox. OpenShell uses a **managed inference proxy** (`https://inference.local`) that injects credentials at the network level.

### How it works

1. You register a provider with your API key in the gateway (one-time setup).
2. You configure `inference.local` to route through that provider.
3. The agent inside the sandbox calls `https://inference.local` instead of the LLM API directly.
4. The gateway intercepts the request, injects the real API key, and forwards it to the provider.

The sandbox process never sees the real API key — it uses an arbitrary placeholder.

### Setup (one-time)

```bash
# Register the API key (--from-existing reads from exported env vars or tool config)
export ANTHROPIC_API_KEY=sk-ant-...
openshell provider create --name anthropic --type anthropic --from-existing

# Point the inference proxy at this provider and model
openshell inference set --provider anthropic --model claude-haiku-4-5-20251001
```

### Agent code (inside the sandbox)

Use LiteLLM with `base_url` pointing to the inference proxy:

```python
from obelix.adapters.outbound.litellm import LiteLLMProvider

provider = LiteLLMProvider(
    model_id="anthropic/claude-haiku-4-5-20251001",
    api_key="placeholder",  # real key injected by gateway at network level
    base_url="https://inference.local",  # gateway-managed inference proxy
)
```

### Why not environment variables?

OpenShell injects provider credentials into sandbox env vars as opaque resolver
references (e.g. `openshell:resolve:env:ANTHROPIC_API_KEY`), not as actual secret
values. These references are resolved by the gateway's network proxy, not by the
sandbox process. Reading them with `os.getenv()` returns the resolver string, not
the real key.

This is by design — it prevents credentials from leaking through process environment,
logs, or `/proc`. The managed inference proxy (`inference.local`) is the intended way
to consume LLM credentials inside a sandbox.

### Switching providers

To change the LLM provider, update the gateway inference config and the model in your
agent code:

```bash
export OPENAI_API_KEY=sk-...
openshell provider create --name openai --type openai --from-existing
openshell inference set --provider openai --model gpt-4o
```

All sandboxes on the gateway pick up the new inference route automatically.
No network policy changes needed — `inference.local` traffic stays internal.

### deploy.py providers parameter

The `providers=["anthropic"]` parameter in `a2a_openshell_deploy()` tells the deployer
to verify that the provider exists in the gateway before creating the sandbox. It does
not create or inject credentials — that is handled by the inference proxy.

---

## Cleanup

The deployer manages the full sandbox lifecycle. On Ctrl+C, SIGTERM, or context manager exit:

1. Port forwarding is stopped via `openshell forward stop`
2. The sandbox is deleted via SDK
3. Generated files (`.Dockerfile.openshell`, `.dockerignore`) are cleaned up

```python
# Blocking mode (deploy.py style)
factory.a2a_openshell_deploy("agent", port=8002, ...)
# blocks until Ctrl+C, then cleans up

# Context manager
async with OpenShellDeployer(...) as info:
    print(f"Agent at {info.endpoint}")
# sandbox destroyed automatically
```

---

## Troubleshooting

### Sandbox stuck in "Provisioning" / CrashLoopBackOff

Check the previous container logs:
```bash
openshell doctor exec -- kubectl -n openshell logs <sandbox-name> --previous
```

Common causes:
- **"sandbox user not found"** — add `RUN useradd -m sandbox` to your Dockerfile
- **"Network namespace creation failed"** — install `iproute2` in your Dockerfile
- **"No such file or directory (os error 2)"** — same as above, `ip` command is missing

### Server not responding (curl returns empty reply)

Read the server log inside the sandbox:
```bash
ssh -o "ProxyCommand=openshell ssh-proxy --gateway-name openshell --name <sandbox>" \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    sandbox@openshell-<sandbox> "cat /tmp/a2a-server.log"
```

Common causes:
- **"No module named 'litellm'"** — add `extras=["litellm"]` to `a2a_openshell_deploy()`
- **"Permission denied: '/app/logs/...'"** — use `log_dir="/tmp/logs"` in `setup_logging()`
- **"Failed to download ... tunnel error"** — the sandbox has no PyPI access. Use `/app/.venv/bin/python` instead of `uv run`
- **"missing required argument: 'executor'"** — pass `BashTool(executor=LocalShellExecutor())`

### Push notifications not working (polling fallback)

Check the sandbox proxy logs:
```bash
openshell logs <sandbox> | grep webhook
```

If you see `action=deny ... reason=endpoint has L7 rules; use CONNECT`:
- The webhook endpoint uses plain HTTP, but the policy has `protocol: rest` (L7)
- Fix: remove `protocol` from the webhook network policy rule (use L4 passthrough)

### Forward port already in use

```bash
openshell forward list          # see active forwards
openshell forward stop <port> <sandbox>   # stop stale forward
```

---

## Limitations

- **Linux/macOS only**: the OpenShell SDK does not have Windows wheels. Use WSL2 on Windows.
- **Docker required**: for image build and gateway.
- **Build time**: first build downloads all dependencies (~60-90s). Subsequent builds use Docker layer cache.
