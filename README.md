# Obelix

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

A multi-provider LLM agent framework with tool support, hooks system, and seamless integration with major AI providers.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Oracle Cloud (OCI), IBM Watson, Ollama, vLLM
- **Tool System**: Declarative tool creation with automatic validation using Pydantic
- **Sub-Agent Orchestration**: Compose hierarchical agent workflows with the Agent Factory
- **Parallel Tool Calls**: Execute multiple tools concurrently for improved performance
- **Hooks System**: Intercept and modify agent behavior at runtime (validation, error recovery, context injection)
- **Async-First Architecture**: All providers use async `invoke()` - native async clients (Anthropic, OpenAI, Ollama, OCI) or `asyncio.to_thread()` for sync SDKs (IBM, vLLM)
- **Thread-Safe Execution**: Parallel agent and tool calls don't share mutable state
- **Loguru Logging**: Structured logging with rotation and color output
- **Built on Pydantic**: Type safety, validation, and automatic schema generation throughout

---

## Requirements

- **Python 3.13+**
- One or more LLM provider credentials

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/GiulioSurya/Obelix.git
cd Obelix

# Sync dependencies (uses uv)
uv sync

# Or install in editable mode with all development tools
uv sync --all-extras --group dev
```

### Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies only
pip install .

# Install with specific provider(s)
pip install ".[openai]"             # OpenAI GPT models
pip install ".[anthropic]"          # Anthropic Claude
pip install ".[oci]"                # Oracle Cloud Infrastructure
pip install ".[ibm]"                # IBM Watson X
pip install ".[ollama]"             # Ollama (local models)
pip install ".[vllm]"               # vLLM (self-hosted inference)

# Install with MCP support
pip install ".[mcp]"

# Install multiple providers
pip install ".[anthropic,oci]"

# Install all LLM providers
pip install ".[all-llm]"

# Install everything (all providers + MCP)
pip install ".[all]"

# Install with development tools
pip install ".[dev]"
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `anthropic` | Anthropic Claude provider (native SDK) |
| `openai` | OpenAI GPT models |
| `oci` | Oracle Cloud Infrastructure Generative AI |
| `ibm` | IBM Watson X AI |
| `ollama` | Ollama local models |
| `vllm` | vLLM self-hosted inference |
| `mcp` | Model Context Protocol support |
| `all-llm` | All LLM providers |
| `all` | All providers + MCP |
| `dev` | Development tools (pytest, ruff, coverage) |

---

## Quick Start

### Minimal Agent

```python
from obelix.core.agent import BaseAgent
from obelix.infrastructure.logging import setup_logging

setup_logging()

agent = BaseAgent(system_message="You are a helpful assistant.")
response = agent.execute_query("Hello, how can you help me?")
print(response.content)
```

### Creating a Tool

Tools are created using the `@tool` decorator with Pydantic fields for automatic validation:

```python
from obelix.core.tool.tool_decorator import tool
from obelix.core.model.tool_message import ToolCall, ToolResult
from pydantic import Field


@tool(name="calculator", description="Performs basic arithmetic")
class CalculatorTool:
    operation: str = Field(..., description="add, subtract, multiply, divide")
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")

    async def execute(self) -> dict:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None,
        }
        if self.operation not in operations:
            raise ValueError(f"Unknown operation: {self.operation}")
        result = operations[self.operation](self.a, self.b)
        return {"result": result, "operation": self.operation}


# Register the tool with an agent
agent = BaseAgent(
    system_message="You are a helpful math assistant.",
    tools=[CalculatorTool],
)

response = agent.execute_query("What is 15 * 7?")
print(response.content)
```

### Creating an Agent

```python
from obelix.core.agent import BaseAgent


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are a specialized assistant for data analysis."
        )
        self.register_tool(CalculatorTool())


agent = MyAgent()
response = agent.execute_query("Analyze this data")
print(response.content)
```

### Sub-Agent Orchestration

Agents can coordinate with other agents via the Agent Factory:

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory


class SQLAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You are a SQL expert.")


class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You coordinate database tasks.")


factory = AgentFactory()
factory.register(
    "sql_analyzer",
    SQLAnalyzerAgent,
    subagent_description="Analyzes SQL errors and suggests fixes",
)
factory.register("coordinator", CoordinatorAgent)

coordinator = factory.create("coordinator", subagents=["sql_analyzer"])
response = coordinator.execute_query("Why is this query failing?")
print(response.content)
```

---

## Project Structure

```
Obelix/
├── src/obelix/                      # Main package
│   ├── core/                        # Core domain logic
│   │   ├── agent/                   # Agent framework
│   │   │   ├── base_agent.py        # BaseAgent execution engine
│   │   │   ├── agent_factory.py     # Factory for agent creation/composition
│   │   │   ├── subagent_wrapper.py  # Bridge between agents and tools
│   │   │   ├── hooks.py             # Hook system and lifecycle events
│   │   │   ├── shared_memory.py     # Shared memory for agent coordination
│   │   │   └── event_contracts.py   # Event validation specifications
│   │   ├── model/                   # Message types (all Pydantic models)
│   │   │   ├── human_message.py
│   │   │   ├── assistant_message.py
│   │   │   ├── system_message.py
│   │   │   ├── tool_message.py
│   │   │   └── standard_message.py
│   │   └── tool/                    # Tool infrastructure
│   │       ├── tool_base.py         # Tool Protocol
│   │       └── tool_decorator.py    # @tool decorator
│   ├── ports/                       # Inbound/Outbound abstract interfaces
│   │   └── outbound/
│   │       ├── llm_provider.py      # AbstractLLMProvider ABC
│   │       ├── llm_connection.py    # AbstractLLMConnection ABC
│   │       └── embedding_provider.py
│   ├── adapters/                    # Concrete implementations
│   │   └── outbound/
│   │       ├── anthropic/           # Anthropic Claude
│   │       ├── openai/              # OpenAI GPT
│   │       ├── oci/                 # Oracle Cloud Generative AI
│   │       ├── ibm/                 # IBM Watson X
│   │       ├── ollama/              # Ollama local models
│   │       ├── vllm/                # vLLM self-hosted
│   │       └── embedding/           # Embedding providers
│   ├── infrastructure/              # Cross-cutting concerns
│   │   ├── logging.py               # Loguru configuration
│   │   ├── providers.py             # Providers enum
│   │   ├── k8s.py                   # Kubernetes configuration
│   │   └── utility/
│   └── plugins/                     # Optional extensions
│       ├── builtin/                 # AskUserQuestionTool
│       └── mcp/                     # Model Context Protocol
├── tests/                           # Comprehensive test suite (200+ tests)
├── docs/                            # User documentation
├── pyproject.toml                   # Project configuration
├── uv.lock                          # uv lockfile
└── README.md                        # This file
```

---

## Supported LLM Providers

| Provider | Status | Import |
|----------|--------|--------|
| OpenAI | Supported | `from obelix.adapters.outbound.openai.provider import OpenAIProvider` |
| Anthropic | Supported | `from obelix.adapters.outbound.anthropic.provider import AnthropicProvider` |
| Oracle Cloud (OCI) | Supported | `from obelix.adapters.outbound.oci.provider import OCILLm` |
| IBM Watson | Supported | `from obelix.adapters.outbound.ibm.provider import IBMProvider` |
| Ollama | Supported | `from obelix.adapters.outbound.ollama.provider import OllamaProvider` |
| vLLM | Supported | `from obelix.adapters.outbound.vllm.provider import VLLMProvider` |

---

## Using Providers

### Basic Provider Usage

```python
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider
from obelix.core.agent import BaseAgent

# 1. Create connection (reads ANTHROPIC_API_KEY from env)
connection = AnthropicConnection()

# 2. Create provider
provider = AnthropicProvider(
    connection=connection,
    model_id="claude-sonnet-4-20250514",
    max_tokens=4000,
    temperature=0.2,
)

# 3. Pass to agent
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
)

response = agent.execute_query("Hello!")
print(response.content)
```


## Tooling

### uv (Package Manager)

The project uses **uv** for dependency management. Common commands:

```bash
# Sync dependencies
uv sync

# Install with specific extras
uv sync --extra oci
uv sync --all-extras

# Install development tools
uv sync --group dev

# Run commands in virtual environment
uv run pytest
uv run ruff check .
uv run ruff format .
```

### ruff (Linter & Formatter)

All code must pass ruff checks:

```bash
# Check for issues
uv run ruff check .

# Auto-fix issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without modifying
uv run ruff format --check .
```

### pytest (Testing)

Run tests with pytest:

```bash
# Run all tests
uv run pytest

# Verbose output
uv run pytest -v

# With coverage
uv run pytest --cov=obelix
```

---

## Documentation

Full documentation is available in the `docs/` directory:

- **[User Guides](docs/)** - How to use Obelix
  - [Creating and Using Agents](docs/base_agent.md)
  - [Agent Factory & Orchestration](docs/agent_factory.md)
  - [Hooks System](docs/hooks.md)

For detailed API reference and examples, see [docs/index.md](docs/index.md).

---

## How to Contribute

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GiulioSurya/Obelix.git
   cd Obelix
   ```

2. **Set up the development environment**:
   ```bash
   uv sync --all-extras --group dev
   ```

3. **Make your changes**:
   - Follow the hexagonal architecture patterns in `src/obelix/`
   - Use Pydantic for all data models
   - Write type hints using Python 3.13+ builtins (`dict`, `list`, `X | None`)
   - Add/update tests in `tests/`

4. **Run linting and formatting**:
   ```bash
   uv run ruff check . --fix
   uv run ruff format .
   ```

5. **Run tests**:
   ```bash
   uv run pytest
   ```

6. **Commit and create a pull request**:
   - Use clear commit messages
   - Reference any related issues
   - Ensure all tests pass and linting passes

### Architecture Principles

- **Hexagonal Architecture**: Core logic in `core/`, abstract interfaces in `ports/`, implementations in `adapters/`
- **No Backward Compatibility Re-exports**: All imports point to final locations
- **Self-Contained Providers**: Each provider converts messages and tools inline, no shared registry
- **Structural Typing for Tools**: Tools are Protocols, not base classes (flexibility without inheritance)
- **Type Safety**: All code must pass type checking with Python 3.13+ syntax

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Obelix Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
