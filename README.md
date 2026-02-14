# Obelix

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

A multi-provider LLM agent framework with tool support, hooks system, and seamless integration with major AI providers.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Oracle Cloud (OCI), IBM Watson, Ollama, vLLM
- **Tool System**: Declarative tool creation with automatic validation
- **Sub-Agent Orchestration**: Compose hierarchical agent workflows with the Agent Factory
- **Parallel Tool Calls**: Execute multiple tools concurrently for improved performance
- **Hooks System**: Intercept and modify agent behavior at runtime
- **Async-First Architecture**: All providers use async `invoke()` - native async clients (Anthropic, OpenAI, Ollama, OCI) or `asyncio.to_thread()` for sync SDKs (IBM, vLLM). Event loop never blocks.
- **Thread-Safe Execution**: Parallel agent and tool calls don't share mutable state
- **Loguru Logging**: Structured logging with rotation and color output

---

## Requirements

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- One or more LLM provider credentials

## Installing uv

uv is a fast Python package manager. Install it with:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip (any OS)
pip install uv
```

## Installation

```bash
# Clone the repository
git clone https://github.com/GiulioSurya/Obelix.git
cd Obelix
```

### Option 1: uv sync (Recommended)

```bash
# Install core dependencies only
uv sync

# Install with specific provider(s)
uv sync --extra openai             # OpenAI GPT models (also works with OpenAI-compatible APIs)
uv sync --extra anthropic          # Anthropic Claude
uv sync --extra oci                # Oracle Cloud Infrastructure
uv sync --extra ibm                # IBM Watson X
uv sync --extra ollama             # Ollama (local models)
uv sync --extra vllm               # vLLM (self-hosted inference)

# Install with MCP support
uv sync --extra mcp

# Install multiple providers
uv sync --extra anthropic --extra oci

# Install all LLM providers
uv sync --extra all-llm

# Install everything (all providers + MCP)
uv sync --extra all

# Install with development tools
uv sync --extra dev
```

### Option 2: uv pip install (Editable mode)

If you prefer `pip`-style installation:

```bash
# Create and activate a virtual environment first
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode
uv pip install -e .

# Install with extras
uv pip install -e ".[anthropic]"
uv pip install -e ".[anthropic,oci]"
uv pip install -e ".[all]"
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `openai` | OpenAI GPT models + OpenAI-compatible APIs (Anthropic, Azure, etc.) |
| `anthropic` | Anthropic Claude provider (native SDK) |
| `oci` | Oracle Cloud Infrastructure Generative AI |
| `ibm` | IBM Watson X AI |
| `ollama` | Ollama local models |
| `vllm` | vLLM self-hosted inference |
| `mcp` | Model Context Protocol support |
| `all-llm` | All LLM providers |
| `all` | All providers + MCP |
| `dev` | Development tools (pytest, coverage) |


## Powered by Pydantic

Obelix leverages [Pydantic](https://github.com/pydantic/pydantic) extensively for type safety, validation, and automatic schema generation throughout the framework.


### Automatic Schema Generation for Tools

The `@tool` decorator extracts Pydantic `Field` definitions and automatically generates JSON Schema for MCP compatibility:

```python
@tool(name="calculator", description="Performs arithmetic")
class CalculatorTool(ToolBase):
    operation: str = Field(..., description="add, subtract, multiply, divide")
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")

# Schema is auto-generated via Pydantic's model_json_schema()
# No manual JSON Schema writing required!
```

### Self-Healing Validation Loop

When an LLM generates invalid tool arguments, Pydantic catches the error which is automatically added to the conversation history, allowing the agent to self-correct without any extra configuration:

```
LLM generates tool call with wrong arguments
       |
Pydantic validates arguments -> ValidationError
       |
Error caught and wrapped in ToolResult(status=ERROR)
       |
ToolResult added to conversation history (built-in behavior)
       |
LLM sees the error and generates corrected arguments
       |
Tool executes successfully
```

This works out of the box - no hooks required. The error message from Pydantic is descriptive enough for the LLM to understand what went wrong and fix it.

-----

## Quick Start

```python
from obelix.domain.agent import BaseAgent
from obelix.infrastructure.logging import setup_logging

setup_logging()

agent = BaseAgent(system_message="You are a helpful assistant.")
response = agent.execute_query("Hello, how can you help me?")
print(response.content)
```

---

## Creating Tools

Tools allow agents to perform concrete actions like fetching data, creating charts, or calling external APIs.

### Basic Tool Structure

Tools are created by extending `ToolBase` and using the `@tool` decorator:

```python
from obelix.domain.tool.tool_decorator import tool
from obelix.domain.tool.tool_base import ToolBase
from pydantic import Field

@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(ToolBase):
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")

    async def execute(self) -> dict:
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }

        if self.operation not in operations:
            raise ValueError(f"Unknown operation: {self.operation}")

        result = operations[self.operation](self.a, self.b)
        return {"result": result, "operation": self.operation}
```

### Tool with Dependencies

Tools can have dependencies injected via constructor:

```python
@tool(name="weather_fetcher", description="Fetches current weather for a city")
class WeatherTool(ToolBase):
    city: str = Field(..., description="City name to get weather for")

    def __init__(self, api_client):
        self.api_client = api_client

    async def execute(self) -> dict:
        data = self.api_client.get_weather(self.city)
        return {
            "city": self.city,
            "temperature": data["temp"],
            "conditions": data["conditions"]
        }
```

### Key Points

- Use the `@tool` decorator with `name` and `description` (both required)
- Define input parameters using Pydantic `Field`
- Implement `async def execute(self)` method
- The decorator handles validation, error wrapping, and result formatting

### Interactive Questions Tool

The `AskUserQuestionTool` allows agents to gather user input through structured, interactive questions with multiple-choice options:

```python
from obelix.plugins.builtin.ask_user_question_tool import AskUserQuestionTool

class InteractiveAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are an assistant that gathers user preferences."
        )
        self.register_tool(AskUserQuestionTool())
```

The agent can then ask questions dynamically:

```python
# The LLM will call the tool with this structure
questions = [{
    "question": "Which database should we use?",
    "header": "Database",
    "options": [
        {
            "label": "PostgreSQL",
            "description": "Relational, ACID compliant, perfect for structured data"
        },
        {
            "label": "MongoDB",
            "description": "Document store, flexible schema, great for rapid development"
        }
    ],
    "multi_select": False
}]
```

**Features**:
- Single-select and multi-select questions
- Question validation (1-4 questions, 2-4 options each)
- Users can always select "Other" for custom input
- Interactive CLI presentation with formatted output

---

## Creating Agents

Agents orchestrate conversations with LLMs and execute tools based on the model's decisions.

### Basic Agent

```python
from obelix.domain.agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are a specialized assistant for data analysis.",
        )
```

### Agent with Tools

Tools can be registered via the `tools` parameter or the `register_tool()` method:

```python
from obelix.domain.agent import BaseAgent

# Option 1: tools parameter (single tool or list)
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    tools=[CalculatorTool, WeatherTool]  # Classes auto-instantiated
)

# Option 2: tools parameter with dependencies
api_client = WeatherAPI(api_key="...")
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    tools=[CalculatorTool, WeatherTool(api_client)]  # Mix classes and instances
)

# Option 3: register_tool() method
class ToolEquippedAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are an assistant with calculation capabilities.",
        )
        self.register_tool(CalculatorTool())
```

### Hooks System

Hooks intercept agent lifecycle events enabling **validation**, **error recovery**, **context injection**, and **flow control**. Each hook receives an `AgentStatus` and can return a `HookDecision` to control execution.

**Common use cases:**
- **Output validation**: Ensure LLM response contains required elements, retry if missing
- **Error recovery**: Inject context (e.g., database schema) when tools fail
- **Context injection**: Add plans or data before LLM calls
- **Result enrichment**: Transform tool results (e.g., fetch docs for error codes)

```python
from obelix.domain.agent import BaseAgent
from obelix.domain.agent.hooks import AgentEvent, HookDecision, AgentStatus
from obelix.domain.model import SystemMessage


class ValidatingAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="...")

        # Validate output contains required tag, retry if missing
        self.on(AgentEvent.BEFORE_FINAL_RESPONSE) \
            .when(self._missing_answer_tag) \
            .handle(
                decision=HookDecision.RETRY,
                effects=[self._inject_format_guidance]
            )

    def _missing_answer_tag(self, ctx: AgentStatus) -> bool:
        content = ctx.assistant_message.content or ""
        return "<answer>" not in content

    def _inject_format_guidance(self, ctx: AgentStatus) -> None:
        ctx.conversation_history.append(
            SystemMessage(content="Your response must include <answer>...</answer> tags.")
        )
```

#### Hook Decisions

| Decision | Effect |
|----------|--------|
| `CONTINUE` | Proceed normally (default) |
| `RETRY` | Restart current LLM iteration |
| `FAIL` | Raise RuntimeError |
| `STOP` | Return immediately with provided value |

#### AgentStatus Fields

| Event | `iteration` | `tool_call` | `tool_result` | `assistant_message` | `error` |
|-------|:-----------:|:-----------:|:-------------:|:-------------------:|:-------:|
| `BEFORE_LLM_CALL` | x | - | - | - | - |
| `AFTER_LLM_CALL` | x | - | - | x | - |
| `BEFORE_TOOL_EXECUTION` | x | x | - | - | - |
| `AFTER_TOOL_EXECUTION` | x | - | x | - | - |
| `ON_TOOL_ERROR` | x | - | x | - | x |
| `BEFORE_FINAL_RESPONSE` | x | - | - | x | - |
| `QUERY_END` | x | - | - | - | - |

**Always available**: `ctx.agent`, `ctx.conversation_history`

#### Hook API

```python
self.on(AgentEvent.EVENT_NAME) \
    .when(condition_fn) \           # fn(ctx) -> bool
    .handle(
        decision=HookDecision.X,    # CONTINUE | RETRY | FAIL | STOP
        value=transform_fn,         # Optional: fn(ctx, current_value) -> new_value
        effects=[effect_fn, ...]    # Optional: list of fn(ctx) -> None
    )
```

---

### Tool Policy

Tool policies enforce that specific tools must be called before the agent can respond. Useful for ensuring required actions are performed.

```python
from obelix.domain.model.tool_message import ToolRequirement

agent = BaseAgent(
    system_message="You are a SQL assistant.",
    tool_policy=[
        ToolRequirement(
            tool_name="sql_executor",
            min_calls=1,
            require_success=True,
            error_message="You must execute the SQL query before responding."
        )
    ]
)
```

If violated: agent retries with guidance message, or fails at max iterations.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tool_name` | `str` | required | Tool to enforce |
| `min_calls` | `int` | `1` | Minimum calls required |
| `require_success` | `bool` | `False` | Must succeed (not error) |
| `error_message` | `str` | `None` | Custom guidance message |

### Sub-Agent Orchestration

Obelix supports hierarchical agent composition. Any agent can register other agents as sub-agents via `register_agent()`, or use the **Agent Factory** for centralized composition.

#### Direct Registration

```python
from obelix.domain.agent import BaseAgent

class SQLAnalyzerAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You are a SQL expert.")
        self.register_tool(SQLQueryTool())

class CoordinatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(system_message="You coordinate database tasks.")

# Any agent can register sub-agents directly
coordinator = CoordinatorAgent()
coordinator.register_agent(
    SQLAnalyzerAgent(),
    name="sql_analyzer",
    description="Analyzes SQL errors and suggests fixes",
)

response = coordinator.execute_query("Why is this query failing? Error: ORA-00942")
```

#### Using Agent Factory

```python
from obelix.domain.agent import BaseAgent
from obelix.domain.agent.agent_factory import AgentFactory

factory = AgentFactory()

factory.register("sql_analyzer", SQLAnalyzerAgent,
                 subagent_description="Analyzes SQL errors and suggests fixes")

factory.register("coordinator", CoordinatorAgent)

# Create orchestrator with sub-agents
coordinator = factory.create("coordinator", subagents=["sql_analyzer"])

response = coordinator.execute_query("Why is this query failing? Error: ORA-00942")
```

#### Stateless vs Stateful

| `stateless` | Behavior |
|-------------|----------|
| `False` (default) | Conversation history preserved across calls, serialized execution |
| `True` | Each call isolated on a copy, allows parallel execution |

```python
# Direct
coordinator.register_agent(translator, name="translator",
                           description="Translates text", stateless=True)

# Via factory
factory.register("translator", TranslatorAgent,
                 subagent_description="Translates text",
                 stateless=True)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_message` | `str` | required | System prompt for the agent |
| `provider` | `AbstractLLMProvider` | `None` | LLM provider (uses GlobalConfig if None) |
| `max_iterations` | `int` | `15` | Maximum agent loop iterations |
| `tools` | `ToolBase / List[ToolBase]` | `None` | Tools to register (classes or instances) |
| `tool_policy` | `List[ToolRequirement]` | `None` | Required tool calls before responding |
| `exit_on_success` | `List[str]` | `None` | Tool names that end the loop on success |

### Execution Methods

```python
# Synchronous (for CLI)
response = agent.execute_query("Analyze this data")

# Asynchronous (for FastAPI/async contexts)
response = await agent.execute_query_async("Analyze this data")

# Access response
print(response.content)       # Text response
print(response.tool_results)  # Tool execution results
print(response.error)         # Error message if any
```

---

## Connections and Providers

Obelix uses a layered architecture: **Connection -> Provider -> Agent**.

### Architecture Overview

```
Agent      (orchestrates conversation, executes tools)
   |
Provider   (model parameters, invoke(), message formatting)
   |
Connection (credentials, singleton client, thread-safe)
```

### Creating a Provider

1. **Connection**: Manages credentials and the API client (singleton, thread-safe)
2. **Provider**: Uses the connection and defines model parameters

```python
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider
from obelix.domain.agent import BaseAgent

# 1. Create connection (reads ANTHROPIC_API_KEY from env)
connection = AnthropicConnection()

# 2. Create provider with connection and model parameters
provider = AnthropicProvider(
    connection=connection,
    model_id="claude-sonnet-4-20250514",
    max_tokens=4000,
    temperature=0.2
)

# 3. Pass provider to agent
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
)
```

### Using GlobalConfig (Recommended)

`GlobalConfig` is a singleton that manages global provider configuration, allowing you to set up a provider once and reuse it across your application.

#### Step 1: Initialize Connection

```python
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection

connection = AnthropicConnection(api_key="your_api_key")
```

#### Step 2: Configure GlobalConfig

```python
from obelix.infrastructure.config import GlobalConfig
from obelix.infrastructure.providers import Providers

config = GlobalConfig()
config.set_provider(Providers.ANTHROPIC, connection=connection)
```

#### Step 3: Use GlobalConfig in Agents

Any agent will automatically use the configured provider if no provider is passed:

```python
from obelix.domain.agent import BaseAgent

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    # provider=None -> uses GlobalConfig
)

response = agent.execute_query("What can you help me with?")
print(response.content)
```

#### Customize Provider Parameters

```python
provider = config.get_current_provider_instance(
    model_id="claude-opus-4-20250514",
    max_tokens=4096,
    temperature=0.5
)

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
)
```

### Available Connections

| Connection | Credentials |
|------------|-------------|
| `AnthropicConnection(api_key)` | Anthropic API key |
| `OpenAIConnection(api_key, base_url=None)` | OpenAI API key (or any OpenAI-compatible API) |
| `OCIConnection(oci_config)` | OCI config (file path or dict) |
| `IBMConnection(api_key, project_id, url)` | IBM Watson X credentials |

---

## Using the Logger

Obelix uses [Loguru](https://github.com/Delgan/loguru) for structured logging with automatic rotation and colored console output.

### Setup (Once at Application Start)

```python
from obelix.infrastructure.logging import setup_logging

# Default configuration
setup_logging()

# Custom configuration
setup_logging(
    level="DEBUG",           # Minimum level for file logging
    console_level="INFO",    # Minimum level for console output
    log_dir="logs",          # Log directory
    log_filename="app.log"   # Log file name
)
```

### Usage in Modules

```python
from obelix.infrastructure.logging import get_logger

logger = get_logger(__name__)

def my_function():
    logger.debug("Detailed debug information")
    logger.info("Operation started")
    logger.success("Operation completed successfully")
    logger.warning("Something unusual happened")
    logger.error("An error occurred")
    logger.critical("Fatal error - cannot continue")
```

### Log Levels

| Level | Value | Usage |
|-------|-------|-------|
| TRACE | 5 | Extreme detail for deep debugging |
| DEBUG | 10 | Development information |
| INFO | 20 | Normal application events |
| SUCCESS | 25 | Successful operations (Loguru-specific) |
| WARNING | 30 | Unusual but handled situations |
| ERROR | 40 | Errors that prevent an operation |
| CRITICAL | 50 | Fatal errors requiring shutdown |

---

## Project Structure

```
src/
└── obelix/
    ├── domain/                          # Business logic
    │   ├── agent/                       # Agent infrastructure
    │   │   ├── base_agent.py            # BaseAgent class (execution engine)
    │   │   ├── agent_factory.py         # AgentFactory for agent creation/composition
    │   │   ├── subagent_wrapper.py      # SubAgentWrapper (BaseAgent -> ToolBase bridge)
    │   │   ├── hooks.py                 # Hooks system + AgentEvent/AgentStatus
    │   │   └── event_contracts.py       # Event validation specifications
    │   ├── model/                       # Message types (Pydantic)
    │   │   ├── human_message.py         # HumanMessage
    │   │   ├── assistant_message.py     # AssistantMessage, AssistantResponse
    │   │   ├── system_message.py        # SystemMessage
    │   │   ├── tool_message.py          # ToolMessage, ToolCall, ToolResult
    │   │   └── standard_message.py      # StandardMessage union type
    │   └── tool/                        # Tool infrastructure
    │       ├── tool_base.py             # Abstract ToolBase
    │       └── tool_decorator.py        # @tool decorator
    ├── ports/
    │   └── outbound/                    # Abstract interfaces (ABCs)
    │       ├── llm_provider.py          # AbstractLLMProvider
    │       ├── llm_connection.py        # AbstractLLMConnection
    │       └── embedding_provider.py    # AbstractEmbeddingProvider
    ├── adapters/
    │   └── outbound/                    # Provider implementations
    │       ├── anthropic/               # Anthropic Claude
    │       ├── openai/                  # OpenAI + compatible APIs
    │       ├── oci/                     # Oracle Cloud Infrastructure
    │       ├── ibm/                     # IBM Watson X
    │       ├── ollama/                  # Ollama (local models)
    │       └── vllm/                    # vLLM (self-hosted)
    ├── infrastructure/                  # Cross-cutting concerns
    │   ├── config.py                    # GlobalConfig singleton
    │   ├── logging.py                   # Loguru configuration
    │   └── providers.py                 # Providers enum
    └── plugins/                         # Optional extensions
        ├── builtin/                     # Built-in tools (AskUserQuestionTool)
        └── mcp/                         # Model Context Protocol support
```

## Supported LLM Providers

| Provider | Module | Description |
|----------|--------|-------------|
| OpenAI | `adapters/outbound/openai/` | GPT models + OpenAI-compatible APIs |
| Anthropic | `adapters/outbound/anthropic/` | Claude models (native SDK) |
| Oracle Cloud (OCI) | `adapters/outbound/oci/` | OCI Generative AI |
| IBM Watson | `adapters/outbound/ibm/` | WatsonX AI |
| Ollama | `adapters/outbound/ollama/` | Local models |
| vLLM | `adapters/outbound/vllm/` | Self-hosted inference |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Obelix Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```