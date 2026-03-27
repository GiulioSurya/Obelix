# OpenShell Setup Guide

How to deploy Obelix agents with sandboxed shell execution using NVIDIA OpenShell.

---

## What is OpenShell

[OpenShell](https://github.com/NVIDIA/OpenShell) is an NVIDIA runtime for sandboxing AI agent workloads. It provides kernel-level isolation for commands that agents execute, without requiring changes to your agent code.

Architecture:

- **Gateway** -- control plane that manages sandbox lifecycle (gRPC, mTLS). Runs as a single Docker container with an embedded K3s cluster.
- **Sandbox** -- isolated containers where commands execute. Each sandbox has its own filesystem, network stack, and process tree.
- **Policy Engine** -- YAML-defined rules that restrict filesystem access, network connections, and process execution at the kernel level (Landlock LSM + L7 HTTP proxy).

The `openshell` PyPI package includes both the Python SDK (`SandboxClient`) and the CLI (`openshell`).

Official documentation: <https://docs.nvidia.com/openshell/latest/>

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Docker | Docker Engine on Linux, Docker Desktop on macOS. Windows requires WSL2. |
| Python 3.13+ | Same as Obelix. |
| `uv sync --extra openshell` | Installs the `openshell` SDK. **Linux and macOS only** -- there is no Windows wheel. |
| Gateway running | `openshell gateway start` (one-time setup, see Scenario 1). |

On Windows, the gateway runs inside Docker/WSL, but the Python SDK must run on Linux or macOS. If your development machine is Windows, run the agent inside a Docker container (see Scenario 1).

---

## Your Code

The Python code is the same regardless of how you deploy the gateway. You write it once; the deployment scenario only changes environment variables and infrastructure.

```python
from obelix.adapters.outbound.shell import OpenShellExecutor
from obelix.plugins.builtin import BashTool
from obelix.core.agent import BaseAgent

executor = OpenShellExecutor(policy="./policy.yaml")
agent = BaseAgent(system_message="...", provider=my_provider)
agent.register_tool(BashTool(executor=executor))
```

The executor connects to the gateway, creates (or reuses) a sandbox, applies the policy, and routes every `BashTool` command through the sandbox. The LLM does not know it is sandboxed -- it receives stdout/stderr/exit_code as usual.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENSHELL_GATEWAY` | Active cluster from `~/.config/openshell/` | Gateway endpoint (`host:port`, gRPC format -- no `https://` prefix). |
| `OPENSHELL_TLS_CERT_DIR` | Auto-detected from active cluster | Path to directory containing `ca.crt`, `tls.crt`, `tls.key` for mTLS. Required when connecting to a remote or non-default gateway. |

Both can also be passed directly to the constructor:

```python
executor = OpenShellExecutor(
    policy="./policy.yaml",
    gateway="10.0.0.5:8080",
    tls_cert_dir="/etc/openshell/certs",
)
```

### Named vs Auto-Created Sandboxes

By default, `OpenShellExecutor` creates a temporary sandbox on first use and destroys it on `close()`. To reuse a pre-existing sandbox:

```python
executor = OpenShellExecutor(
    sandbox_name="my-sandbox",
    policy="./policy.yaml",
)
# close() does NOT destroy the sandbox when sandbox_name is provided
```

The context manager pattern handles cleanup automatically:

```python
async with OpenShellExecutor(policy="./policy.yaml") as executor:
    agent = BaseAgent(..., tools=[BashTool(executor=executor)])
    response = await agent.execute_query("list files in /sandbox")
# sandbox destroyed on exit
```

---

## Writing a Policy

Policies define what the sandbox can and cannot do. They are YAML files with three main sections.

### Minimal Example

```yaml
version: 1

# STATIC -- locked at sandbox creation, requires sandbox recreate to change

filesystem_policy:
  include_workdir: true
  read_only:
    - /usr
    - /lib
    - /bin
    - /sbin
    - /etc
    - /proc
    - /dev/urandom
  read_write:
    - /sandbox
    - /tmp
    - /dev/null

landlock:
  compatibility: best_effort

process:
  run_as_user: sandbox
  run_as_group: sandbox

# DYNAMIC -- hot-reloadable via `openshell policy set` without recreating the sandbox

network_policies:
  llm_api:
    name: llm-provider
    endpoints:
      - host: api.anthropic.com
        port: 443
        protocol: rest
        tls: terminate
        enforcement: enforce
        access: read-write
    binaries:
      - path: /usr/bin/python*
      - path: /usr/local/bin/python*

  dns:
    name: dns-resolution
    endpoints:
      - host: "*.dns.google"
        port: 53
      - host: "*.dns.google"
        port: 443
    binaries:
      - path: /usr/bin/python*
      - path: /usr/local/bin/python*
```

### Key Points

- **Static fields** (`filesystem_policy`, `process`, `landlock`) are locked at sandbox creation. Changing them requires destroying and recreating the sandbox.
- **Dynamic fields** (`network_policies`) are hot-reloadable. `OpenShellExecutor` watches the policy file for changes and reloads automatically (60-second polling interval).
- **`tls: terminate`** enables L7 HTTPS inspection through the proxy. Without it, the proxy only sees L4 (TCP host:port).
- **`binaries` must match actual paths** inside the sandbox. If your policy says `/usr/bin/python*` but the sandbox has Python at `/sandbox/.uv/python/cpython-3.13/bin/python3`, the connection is blocked with `403 Forbidden`. Verify with: `openshell sandbox exec <name> -- which python3`.
- **Network default is deny-all**. Only explicitly whitelisted endpoints are reachable.

Full policy reference: <https://docs.nvidia.com/openshell/latest/policies/>

---

## Scenario 1: Local Development (Docker Compose)

This is the simplest way to get started. The gateway runs on your machine; the Obelix agent runs in a Docker container that connects to it.

### Step 1: Start the Gateway

Run this once. The gateway persists across reboots as a Docker container.

```bash
openshell gateway start
```

This creates certificates in `~/.config/openshell/gateways/<name>/mtls/` and starts the gateway container.

### Step 2: Create the Sandbox

```bash
openshell sandbox create --name obelix-demo --policy policy.yaml
```

### Step 3: Docker Compose

```yaml
# docker-compose.yaml
services:
  bash-server:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8002:8002"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - openshell-config:/root/.config/openshell
    extra_hosts:
      - "host.docker.internal:host-gateway"
    env_file:
      - .env

volumes:
  openshell-config:
```

The `openshell-config` volume shares the gateway certificates with the container. The `extra_hosts` entry lets the container reach the gateway on the host.

### Step 4: Dockerfile

```dockerfile
FROM python:3.13-slim

# System deps + Docker CLI (for openshell gateway)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && \
    curl -fsSL https://get.docker.com | sh && \
    rm -rf /var/lib/apt/lists/*

# uv + openshell CLI
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv tool install openshell
ENV PATH="/root/.local/bin:$PATH"

# Project
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
RUN uv sync --no-dev --extra serve --extra openshell --link-mode=copy

COPY examples/ examples/
COPY policy.yaml .

EXPOSE 8002
CMD ["uv", "run", "python", "examples/shell-demo/bash_server_sandboxed.py"]
```

### Step 5: Run

```bash
# .env file should contain your LLM API key:
# API_KEY=sk-ant-...
# SANDBOX_NAME=obelix-demo

docker compose up --build
```

The agent is now reachable at `http://localhost:8002`. Connect with the CLI client:

```bash
uv run python examples/cli_client.py --url http://localhost:8002
```

A working example lives in `examples/shell-demo/` in the Obelix repository.

---

## Scenario 2: CI/CD (GitHub Actions)

Use OpenShell to sandbox agent shell commands during integration tests.

```yaml
# .github/workflows/test-sandboxed.yml
name: Test with OpenShell Sandbox

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4

      - name: Install openshell CLI
        run: uv tool install openshell

      - name: Start OpenShell gateway
        run: openshell gateway start

      - name: Create sandbox
        run: openshell sandbox create --name ci-sandbox --policy policy.yaml

      - name: Install dependencies
        run: uv sync --extra openshell --group dev

      - name: Run tests
        env:
          SANDBOX_NAME: ci-sandbox
          API_KEY: ${{ secrets.API_KEY }}
        run: uv run pytest tests/ -v -k "openshell"
```

The gateway starts as a Docker container in the runner. Tests use `OpenShellExecutor(sandbox_name="ci-sandbox")` to route commands through the sandbox.

---

## Scenario 3: Production (Kubernetes)

In production, the gateway runs as a Kubernetes Deployment, and agent pods connect to it via a ClusterIP Service.

### Gateway Deployment

```yaml
# k8s/openshell-gateway.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: openshell-system
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openshell-gateway
  namespace: openshell-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: openshell-gateway
  template:
    metadata:
      labels:
        app: openshell-gateway
    spec:
      containers:
        - name: gateway
          image: nvcr.io/nvidia/openshell-gateway:latest
          ports:
            - containerPort: 8080
              name: grpc
          volumeMounts:
            - name: certs
              mountPath: /etc/openshell/certs
      volumes:
        - name: certs
          secret:
            secretName: openshell-mtls
---
apiVersion: v1
kind: Service
metadata:
  name: openshell-gateway
  namespace: openshell-system
spec:
  selector:
    app: openshell-gateway
  ports:
    - port: 8080
      targetPort: grpc
```

### Agent Deployment

```yaml
# k8s/agent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: obelix-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: obelix-agent
  template:
    metadata:
      labels:
        app: obelix-agent
    spec:
      containers:
        - name: agent
          image: your-registry/obelix-agent:latest
          ports:
            - containerPort: 8002
          env:
            - name: OPENSHELL_GATEWAY
              value: "openshell-gateway.openshell-system.svc.cluster.local:8080"
            - name: OPENSHELL_TLS_CERT_DIR
              value: "/etc/openshell/certs"
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: llm-credentials
                  key: api-key
          volumeMounts:
            - name: policy
              mountPath: /etc/obelix/policy.yaml
              subPath: policy.yaml
            - name: certs
              mountPath: /etc/openshell/certs
              readOnly: true
      volumes:
        - name: policy
          configMap:
            name: obelix-policy
        - name: certs
          secret:
            secretName: openshell-mtls
```

### Managing Policy Updates

Since `network_policies` are hot-reloadable, you can update them without restarting the agent:

```bash
# Update the ConfigMap
kubectl create configmap obelix-policy \
  --from-file=policy.yaml=./policy.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# The ConfigMap volume propagates within ~60s.
# OpenShellExecutor's policy watcher detects the file change and reloads.
```

Static policy fields (`filesystem_policy`, `process`) require destroying and recreating the sandbox.

---

## Working with Claude Code

The [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) repository provides Claude Code skills that are useful when working with OpenShell deployments. If you have those skills installed in `.claude/skills/`, the following are relevant:

| Skill | What it does |
|---|---|
| `openshell-cli` | Guides through CLI usage: sandbox create/connect, policy set/get, provider config, port forwarding. |
| `generate-sandbox-policy` | Generates policy YAML from plain-language requirements and optional API docs. |
| `debug-openshell-cluster` | Diagnoses gateway start failures, cluster health issues, and infrastructure problems. |
| `debug-inference` | Diagnoses inference endpoint failures, provider base URL issues, and connectivity from within sandboxes. |

These skills are defined in the NVIDIA/OpenShell repo and must be installed separately. They are not part of Obelix.

---

## Troubleshooting

### ImportError: The 'openshell' package is required

The `openshell` PyPI package has no Windows wheel. It works on Linux and macOS only.

```bash
# Install on Linux/macOS
uv sync --extra openshell

# On Windows, run the agent inside a Docker container (see Scenario 1)
```

### 403 Forbidden on Whitelisted Endpoints

The L7 proxy blocks requests when the `binaries` paths in the policy do not match the actual binary path inside the sandbox.

```bash
# Find the real Python path inside the sandbox
openshell sandbox exec obelix-demo -- which python3

# Update the binaries section in your policy to match
# Then reload:
openshell policy set obelix-demo --policy policy.yaml --wait
```

Common mismatches:
- Policy has `/usr/bin/python*` but sandbox Python is at `/usr/local/bin/python3.13`
- Policy has `/usr/bin/python*` but the agent uses uv-managed Python at `/sandbox/.uv/python/cpython-3.13.2-linux-x86_64/bin/python3`

### Gateway Unreachable

```
OpenShell initialization error: failed to connect to gateway
```

Checklist:
1. Is the gateway running? `openshell gateway status` or `docker ps | grep openshell`
2. Is the endpoint correct? `OPENSHELL_GATEWAY` should be `host:port` (gRPC format), not `https://host:port`
3. From a Docker container, use `host.docker.internal:8080` (not `localhost:8080`)
4. In Kubernetes, use the full service DNS: `openshell-gateway.openshell-system.svc.cluster.local:8080`

### mTLS Errors

```
TLS handshake failed / certificate verify failed
```

- Certificates are generated by `openshell gateway start` and stored in `~/.config/openshell/gateways/<name>/mtls/`
- When connecting from a Docker container, mount the cert directory or copy certs into the image
- Verify the cert directory contains all three files: `ca.crt`, `tls.crt`, `tls.key`
- If using `SandboxClient.from_active_cluster()`, ensure the active cluster config matches the running gateway

### Policy Changes Not Taking Effect

- **Static fields** (`filesystem_policy`, `process`, `landlock`): changes require destroying and recreating the sandbox. `openshell sandbox delete <name>` then `openshell sandbox create --name <name> --policy policy.yaml`.
- **Dynamic fields** (`network_policies`): hot-reloadable. Apply with `openshell policy set <name> --policy policy.yaml --wait`. `OpenShellExecutor` also watches the file and reloads automatically every 60 seconds.
- After a reload, verify with: `openshell policy get <name>`

### Multiline Commands Rejected

The OpenShell gRPC SDK rejects commands containing newline characters:

```
command argument 2 contains newline or carriage return characters
```

Use one-liners instead: `python3 -c "print('hello')"`, command chaining with `&&`, or write a script to a file first and then execute it.

---

## See Also

- [BashTool Guide](bash_tool.md) -- execution modes, permission policies, security considerations
- [OpenShell Integration](openshell_integration.md) -- internal design notes, executor architecture, testing results
- [NVIDIA OpenShell Documentation](https://docs.nvidia.com/openshell/latest/) -- official reference
- [NVIDIA OpenShell GitHub](https://github.com/NVIDIA/OpenShell) -- source code and Claude Code skills
