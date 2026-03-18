# Obelix

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

**Agent Development Kit** for building, composing, and deploying LLM agents.
Create agents with tools and hooks, orchestrate them with shared memory, deploy as [A2A](https://a2a-protocol.org/) services, and interact through a Rich CLI client.

```
     ___  _          _ _
    / _ \| |__   ___| (_)_  __
   | | | | '_ \ / _ \ | \ \/ /
   | |_| | |_) |  __/ | |>  <
    \___/|_.__/ \___|_|_/_/\_\
    A2A Agent CLI

  OK bash_agent (http://localhost:8002)
  OK coordinator (http://localhost:8001)

                         Connected Agents
  ┏━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃   # ┃ Agent       ┃ Description     ┃ Skills        ┃ URL                     ┃
  ┡━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ 1 * │ bash_agent  │ Shell command   │ bash          │ http://localhost:8002   │
  │     │             │ execution       │               │                         │
  │   2 │ coordinator │ Math coordi-    │ math_agent,   │ http://localhost:8001   │
  │     │             │ nator with      │ report_agent  │                         │
  │     │             │ sub-agents      │               │                         │
  └─────┴─────────────┴─────────────────┴───────────────┴─────────────────────────┘

  ╭──────────────────────── Commands ─────────────────────────╮
  │   /agents              List connected agents              │
  │   /switch <n>          Switch to agent n                  │
  │   /clear               Clear conversation context         │
  │   /quit                Exit                               │
  │                                                           │
  │   @<path>              Attach a file (image, PDF, ...)    │
  │   @"path with spaces"  Attach a file with spaces in path  │
  ╰───────────────────────────────────────────────────────────╯

  [bash_agent] >
```

---

## What Obelix Does

- **Build agents** with tools, hooks, streaming, and planning mode
- **Compose** multi-agent systems with the Agent Factory and shared memory graphs
- **Deploy** agents as HTTP services via the [A2A protocol](https://a2a-protocol.org/)
- **Interact** via the built-in Rich CLI client with multimodal attachments and permission-controlled tool execution
- **Connect** to 100+ LLM providers: Anthropic, OpenAI, OCI, IBM, Ollama, vLLM, and LiteLLM

---

## Installation

Requires **Python 3.13+** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/GiulioSurya/Obelix.git
cd Obelix
uv sync                            # core only
uv sync --extra litellm            # + LiteLLM (100+ providers)
uv sync --extra serve              # + A2A server (FastAPI, Uvicorn)
uv sync --all-extras --group dev   # everything + dev tools
```

Available extras: `anthropic`, `openai`, `oci`, `ibm`, `ollama`, `vllm`, `litellm`, `mcp`, `serve`, `all-llm`, `all`.

pip is also supported: `pip install ".[litellm,serve]"`.

---

## Quick Tour

### 1. Create an Agent

```python
from obelix.adapters.outbound.litellm import LiteLLMProvider
from obelix.core.agent import BaseAgent

provider = LiteLLMProvider(model_id="anthropic/claude-haiku-4-5-20251001")

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
)
response = agent.execute_query("Hello!")
print(response.content)
```

> Full constructor reference: [BaseAgent Guide](docs/base_agent.md)

### 2. Add a Tool

```python
from obelix.core.tool.tool_decorator import tool
from pydantic import Field

@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    async def execute(self) -> dict:
        return {"result": self.a + self.b}

agent = BaseAgent(
    system_message="You are a math assistant.",
    provider=provider,
    tools=[CalculatorTool],
)
```

> Tools, OutputSchema, deferred tools, system_prompt_fragment: [Tools Guide](docs/tools.md)

### 3. Compose with Sub-Agents

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
factory.register("math", MathAgent, subagent_description="Does calculations")
factory.register("report", ReportAgent, subagent_description="Writes reports")
factory.register("coordinator", CoordinatorAgent)

coordinator = factory.create("coordinator", subagents=["math", "report"])
response = coordinator.execute_query("Calculate 123 * 456 and summarize")
```

> Shared memory, PropagationPolicy, tracer: [Agent Factory Guide](docs/agent_factory.md)

### 4. Deploy as A2A Server

```python
factory.a2a_serve("coordinator", port=8000, description="My agent")
```

The server exposes `GET /.well-known/agent.json` (Agent Card) and `POST /` (JSON-RPC 2.0).

> Streaming, input-required, deferred tools over A2A: [A2A Server Guide](docs/a2a_server.md)

### 5. Connect with the CLI Client

```bash
uv run python examples/cli_client.py http://localhost:8000
```

The CLI auto-discovers agents via their Agent Card, handles deferred tool execution (bash commands with `[Y/n]` confirmation, user input) and supports multi-agent switching.

> Permission policies, custom handlers: [BashTool Guide](docs/bash_tool.md)

---

## A2A Protocol

Obelix implements the [A2A protocol](https://a2a-protocol.org/) (Agent-to-Agent, Linux Foundation) for inter-agent communication over HTTP. Any Obelix agent can be deployed as a standalone A2A service and consumed by any A2A-compatible client.

```
                         A2A (JSON-RPC 2.0 over HTTP)
 ┌──────────────┐            ┌─────────────────────────────────────────┐
 │  CLI Client   │───────────│  A2A Server                             │
 │  (or any A2A  │  TextPart │  ┌─────────────────┐   ┌────────────┐  │
 │   client)     │  FilePart │  │ ObelixAgent      │──▶│  Provider  │  │
 │               │  DataPart │  │ Executor         │   │  (LiteLLM, │  │
 │  @image.png ──│──────────▶│  │                  │   │   OCI ...) │  │
 │               │           │  │  tools + hooks   │   └────────────┘  │
 │  bash [Y/n] ◀─│◀──────────│  │  deferred tools  │                   │
 │               │           │  └─────────────────┘                    │
 └──────────────┘            └─────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal** | Attach images, PDFs, files with `@path` from the CLI |
| **Deferred tools** | Agent delegates tool execution to the client (`bash [Y/n]`, user input) |
| **Streaming** | Token-by-token SSE streaming |
| **Multi-agent** | Connect to multiple servers, switch with `/switch <n>` |
| **Structured transport** | `DataPart` preserves types end-to-end (no string serialization) |

```
[agent] > describe this @screenshot.png
[agent] > analyze @report.pdf and @data.csv
[agent] > compare @"C:\path with spaces\photo.jpg" and @photo2.jpg
```

---

## Running the Examples

The `examples/` folder contains ready-to-run demos. All require an LLM API key:

```bash
uv sync --extra litellm --extra serve
```

### Start two A2A servers (in separate terminals)

```bash
# Terminal 1 — bash agent (shell command execution)
API_KEY=sk-... uv run python examples/bash_server.py
# Listening on http://localhost:8002

# Terminal 2 — coordinator with math + report sub-agents
API_KEY=sk-... uv run python examples/factory_server.py
# Listening on http://localhost:8001
```

### Connect with the CLI client

```bash
# Terminal 3
uv run python examples/cli_client.py http://localhost:8002 http://localhost:8001
```

The client resolves both Agent Cards, shows the table above, and you can chat with either agent. Use `/switch 2` to talk to the coordinator, `/switch 1` to go back to bash_agent.

| Example | What it does | Port |
|---------|-------------|------|
| `examples/bash_server.py` | Single agent with BashTool (deferred or local mode) | 8002 |
| `examples/factory_server.py` | Coordinator + math_agent + report_agent with shared memory | 8001 |
| `examples/cli_client.py` | Rich CLI client — multimodal, deferred tools, multi-agent | — |

---

## Supported Providers

| Provider | Extra | Import |
|----------|-------|--------|
| **LiteLLM** (100+) | `litellm` | `from obelix.adapters.outbound.litellm import LiteLLMProvider` |
| Anthropic | `anthropic` | `from obelix.adapters.outbound.anthropic.provider import AnthropicProvider` |
| OpenAI | `openai` | `from obelix.adapters.outbound.openai.provider import OpenAIProvider` |
| Oracle Cloud (OCI) | `oci` | `from obelix.adapters.outbound.oci.provider import OCILLm` |
| IBM Watson | `ibm` | `from obelix.adapters.outbound.ibm.provider import IBMProvider` |
| Ollama | `ollama` | `from obelix.adapters.outbound.ollama.provider import OllamaProvider` |
| vLLM | `vllm` | `from obelix.adapters.outbound.vllm.provider import VLLMProvider` |

LiteLLM routes to Azure, Bedrock, Vertex AI, Groq, Mistral, Together AI, DeepSeek, and many more through a single interface.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [BaseAgent](docs/base_agent.md) | Agent creation, execution, streaming, hooks, planning, deferred tools |
| [Tools](docs/tools.md) | `@tool` decorator, OutputSchema, system_prompt_fragment, built-in tools |
| [Agent Factory](docs/agent_factory.md) | Registration, composition, shared memory, tracer, A2A serve |
| [BashTool](docs/bash_tool.md) | Shell execution modes, security, permission policies, LocalShellExecutor |
| [A2A Server](docs/a2a_server.md) | HTTP deployment, JSON-RPC, streaming SSE, input-required flow |
| [Hooks](docs/hooks.md) | Lifecycle events, conditions, decisions, effects |

---

## Contributing

```bash
git clone https://github.com/GiulioSurya/Obelix.git && cd Obelix
make setup  # or: uv sync --all-extras --group dev && uv run pre-commit install
```

Pre-commit hooks run `ruff check` + `ruff format`. Pre-push runs `pytest`.

Architecture: hexagonal (`core/` | `ports/` | `adapters/`), Pydantic models, structural typing for tools, Python 3.13+ type hints.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).