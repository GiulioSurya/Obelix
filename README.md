# Obelix

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

A multi-provider LLM agent framework with tool support, hooks system, and seamless integration with major AI providers.

## Features

- **Multi-Provider Support**: OpenAI, Anthropic, Oracle Cloud (OCI), IBM Watson, Ollama, vLLM
- **Tool System**: Declarative tool creation with automatic validation
- **Sub-Agent Orchestration**: Compose hierarchical agent workflows with `@orchestrator` and `@subagent`
- **Parallel Tool Calls**: Execute multiple tools concurrently for improved performance
- **Hooks System**: Intercept and modify agent behavior at runtime
- **Async-First Architecture**: All providers use async `invoke()` - native async clients (Anthropic, OpenAI, Ollama) or `asyncio.to_thread()` for sync SDKs (OCI, IBM, vLLM). Event loop never blocks.
- **Thread-Safe Execution**: Parallel agent and tool calls don't share mutable state
- **Loguru Logging**: Structured logging with rotation and color output

---

## Requirements

- **Python 3.13+**
- One or more LLM provider credentials

## Installation

```bash
# Clone the repository
git clone https://github.com/obelix/obelix.git
cd obelix

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies only
pip install -e .

# Install with specific provider(s)
pip install -e ".[openai]"             # OpenAI GPT models (also works with OpenAI-compatible APIs)
pip install -e ".[anthropic]"          # Anthropic Claude
pip install -e ".[oci]"                # Oracle Cloud Infrastructure
pip install -e ".[ibm]"                # IBM Watson X
pip install -e ".[ollama]"             # Ollama (local models)
pip install -e ".[vllm]"               # vLLM (self-hosted inference)

# Install with MCP support
pip install -e ".[mcp]"

# Install multiple providers
pip install -e ".[anthropic,oci]"

# Install all LLM providers
pip install -e ".[all-llm]"

# Install everything (all providers + MCP)
pip install -e ".[all]"

# Install with development tools
pip install -e ".[dev]"
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
┌─────────────────────────────────────────────────────────────────┐
│  LLM generates tool call with wrong arguments                   │
│         ↓                                                       │
│  Pydantic validates arguments → ValidationError                 │
│         ↓                                                       │
│  Error caught and wrapped in ToolResult(status=ERROR)           │
│         ↓                                                       │
│  ToolResult added to conversation history (built-in behavior)   │
│         ↓                                                       │
│  LLM sees the error and generates corrected arguments           │
│         ↓                                                       │
│  Tool executes successfully ✓                                   │
└─────────────────────────────────────────────────────────────────┘
```

This works out of the box - no hooks required. The error message from Pydantic is descriptive enough for the LLM to understand what went wrong and fix it.

-----

## Quick Start

```python
from src.base_agent.base_agent import BaseAgent
from src.logging_config import setup_logging

# Initialize logging
setup_logging()

# Create a simple agent
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    agent_name="SimpleAgent"
)

# Execute a query
response = agent.execute_query("Hello, how can you help me?")
print(response.content)
```

---

## Creating Tools

Tools allow agents to perform concrete actions like fetching data, creating charts, or calling external APIs.

### Basic Tool Structure

Tools are created by extending `ToolBase` and using the `@tool` decorator:

```python
from src.tools.tool_decorator import tool
from src.tools.tool_base import ToolBase
from pydantic import Field
from typing import Any

@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(ToolBase):
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First operand")
    b: float = Field(..., description="Second operand")

    def execute(self) -> dict:
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

    def execute(self) -> dict:
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
- Implement `def execute(self)` method (the decorator handles async wrapping)
- The decorator handles validation, error wrapping, and result formatting

### Interactive Questions Tool

The `AskUserQuestionTool` allows agents to gather user input through structured, interactive questions with multiple-choice options:

```python
from src.tools.tool.ask_user_question_tool import AskUserQuestionTool

class InteractiveAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are an assistant that gathers user preferences.",
            agent_name="PreferenceAgent"
        )

        # Register the question tool
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
from src.base_agent.base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, provider=None):
        super().__init__(
            system_message="You are a specialized assistant for data analysis.",
            provider=provider,
            agent_name="DataAnalysisAgent",
            description="Analyzes data and generates insights"
        )
```

### Agent with Tools

Tools can be registered via the `tools` parameter or the `register_tool()` method:

```python
from src.base_agent.base_agent import BaseAgent
from src.tools.my_tool import CalculatorTool, WeatherTool

# Option 1: tools parameter (single tool or list)
agent = BaseAgent(
    system_message="You are a helpful assistant.",
    agent_name="MyAgent",
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
            agent_name="CalculatorAgent"
        )
        self.register_tool(CalculatorTool())
```

### Agent with Hooks

Hooks allow intercepting events during the agent lifecycle. Each hook receives an `AgentStatus` with relevant information.

```python
from src.base_agent.base_agent import BaseAgent
from src.base_agent.hooks import AgentEvent
from src.messages.human_message import HumanMessage

class SmartAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are an intelligent assistant.",
            agent_name="SmartAgent"
        )
        self._schema_injected = False

        # Inject database schema on "invalid identifier" errors
        self.on(AgentEvent.ON_TOOL_ERROR) \
            .when(self._is_invalid_identifier) \
            .inject(self._create_schema_message)

    def _is_invalid_identifier(self, agent_status) -> bool:
        # agent_status.error contains the error message from the tool
        # agent_status.tool_result contains the full ToolResult object
        return (
            agent_status.error is not None and
            "invalid identifier" in agent_status.error and
            not self._schema_injected
        )

    def _create_schema_message(self, agent_status) -> HumanMessage:
        self._schema_injected = True
        # agent_status.agent gives access to the agent instance
        # agent_status.conversation_history gives access to the full history
        return HumanMessage(content="Database schema: ... Please correct the query.")
```

### AgentStatus

The `AgentStatus` object provides access to the current state. Available fields depend on the event:

| Event | `iteration` | `tool_call` | `tool_result` | `assistant_message` | `error` |
|-------|:-----------:|:-----------:|:-------------:|:-------------------:|:-------:|
| `ON_QUERY_START` | 0 | - | - | - | - |
| `BEFORE_LLM_CALL` | ✓ | - | - | - | - |
| `AFTER_LLM_CALL` | ✓ | - | - | ✓ | - |
| `BEFORE_TOOL_EXECUTION` | ✓ | ✓ | - | - | - |
| `AFTER_TOOL_EXECUTION` | ✓ | - | ✓ | - | - |
| `ON_TOOL_ERROR` | ✓ | - | ✓ | - | ✓ |
| `ON_QUERY_END` | ✓ | - | - | - | - |
| `ON_MAX_ITERATIONS` | ✓ | - | - | - | - |

**Always available**: `agent_status.agent` (the agent instance), `agent_status.conversation_history` (list of messages)

### Hook Methods

| Method | Description |
|--------|-------------|
| `.when(fn)` | Condition to trigger the hook: `fn(agent_status) -> bool` |
| `.inject(fn)` | Append a message to history: `fn(agent_status) -> Message` |
| `.inject_at(pos, fn)` | Insert message at position: `fn(agent_status) -> Message` |
| `.transform(fn)` | Transform the current value: `fn(value, agent_status) -> new_value` |
| `.do(fn)` | Execute side effect: `fn(agent_status)` |

### Sub-Agent Orchestration

Obelix supports hierarchical agent composition through the `@orchestrator` and `@subagent` decorators. An orchestrator agent can register other agents as "virtual tools", delegating specialized tasks through function calling.

#### Creating a Sub-Agent

Sub-agents are specialized agents that can be registered and invoked by orchestrators:

```python
from src.base_agent.subagent_decorator import subagent
from src.base_agent.base_agent import BaseAgent
from pydantic import Field

@subagent(name="sql_analyzer", description="Analyzes SQL errors and suggests fixes")
class SQLAnalyzerAgent(BaseAgent):
    # Optional context fields (in addition to mandatory 'query')
    error_context: str = Field(default="", description="Error context from database")
    schema_info: str = Field(default="", description="Optional schema information")

    def __init__(self):
        super().__init__(
            system_message="You are a SQL expert that analyzes errors and suggests fixes.",
            agent_name="SQLAnalyzer"
        )
        # Sub-agents can have their own tools
        self.register_tool(SQLQueryTool())
```

**Key points**:
- `@subagent` requires `name` and `description` (validated at import time)
- Define optional context fields using Pydantic `Field`
- The `query` field is automatically added as the first parameter
- Sub-agents are autonomous and can have their own tools and hooks

#### Stateless vs Stateful Sub-Agents

The `stateless` parameter controls how sub-agents handle concurrent calls and conversation history:

```python
# Stateless: allows parallel calls, each on isolated copy
@subagent(name="translator", description="Translates text", stateless=True)
class TranslatorAgent(BaseAgent):
    ...

# Stateful (default): calls serialized with lock, preserves conversation history
@subagent(name="analyst", description="Analyzes data with context")
class AnalystAgent(BaseAgent):
    ...
```

| `stateless` | Parallel Calls | Conversation History | Use Case |
|-------------|----------------|---------------------|----------|
| `True` | Yes (concurrent) | Fork of current state, discarded after | One-shot tasks, translations, validations |
| `False` (default) | No (serialized via lock) | Preserved across calls | Multi-turn analysis, agents needing context |

**How stateless works**:
- Each call creates a shallow copy of the agent with a forked `conversation_history`
- The copy is discarded after execution (original agent unchanged)
- `agent_usage` is still accumulated on the original agent
- Multiple calls can run truly in parallel without state collision

#### Creating an Orchestrator

Orchestrators coordinate multiple sub-agents:

```python
from src.base_agent.orchestrator_decorator import orchestrator
from src.base_agent.base_agent import BaseAgent

@orchestrator(name="db_coordinator", description="Coordinates database operations")
class DatabaseCoordinator(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You coordinate database tasks and delegate to specialists.",
            agent_name="DBCoordinator"
        )
        # Orchestrators can have their own tools
        self.register_tool(DatabaseConnectionTool())

# Create orchestrator and register sub-agents
coordinator = DatabaseCoordinator()
coordinator.register_agent(SQLAnalyzerAgent())
coordinator.register_agent(QueryOptimizerAgent())
```

**Key points**:
- `@orchestrator` adds the `register_agent()` method
- Only agents decorated with `@subagent` can be registered
- Sub-agents are wrapped as tools and appear in the LLM's tool schema
- Orchestrators can mix regular tools and sub-agents

#### How Sub-Agents are Invoked

When the LLM calls a registered sub-agent, it provides arguments like a regular tool:

```json
{
  "tool_calls": [{
    "name": "sql_analyzer",
    "arguments": {
      "query": "Why is this query failing?",
      "error_context": "ORA-00942: table or view does not exist",
      "schema_info": "Tables: users, orders, products"
    }
  }]
}
```

The framework automatically:
1. Validates arguments against the sub-agent's schema
2. Combines `query` and context fields into a complete query
3. Executes the sub-agent in isolation (thread-safe copy)
4. Returns the result as a `ToolResult` to the orchestrator

#### Benefits

- **Separation of concerns**: Each sub-agent focuses on a specific task
- **Reusability**: Sub-agents can be registered with multiple orchestrators
- **Parallel execution**: Multiple sub-agents can run concurrently
- **No coupling**: Sub-agents don't know about orchestrators
- **Declarative composition**: Build complex workflows with simple decorators

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_message` | `str` | - | System prompt for the agent |
| `provider` | `AbstractLLMProvider` | `None` | LLM provider (uses GlobalConfig if None) |
| `agent_name` | `str` | `None` | Agent name (uses class name if None) |
| `description` | `str` | `None` | Agent capabilities description |
| `agent_comment` | `bool` | `True` | If True, LLM comments on tool results |

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

Obelix uses a layered architecture: **Connection → Provider → Agent**.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                               │
│  (orchestrates conversation, executes tools)                │
├─────────────────────────────────────────────────────────────┤
│                        Provider                             │
│  (model parameters, invoke(), message formatting)           │
├─────────────────────────────────────────────────────────────┤
│                       Connection                            │
│  (credentials, singleton client, thread-safe)               │
└─────────────────────────────────────────────────────────────┘
```

### Creating a Provider

1. **Connection**: Manages credentials and the API client (singleton, thread-safe)
2. **Provider**: Uses the connection and defines model parameters

```python
from src.connections.llm_connection import AnthropicConnection
from src.llm_providers.anthropic_provider import AnthropicProvider
from src.base_agent.base_agent import BaseAgent

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
    agent_name="MyAgent"
)
```

### Using GlobalConfig (Recommended)

`GlobalConfig` is a singleton that manages global provider configuration, allowing you to set up a provider once and reuse it across your application.

#### Step 1: Initialize Connection

Create a connection by passing the required credentials:

```python
from src.connections.llm_connection import AnthropicConnection
# or: IBMConnection, OCIConnection, etc.

connection = AnthropicConnection(api_key="your_api_key")
```

#### Step 2: Configure GlobalConfig

```python
from src.config import GlobalConfig
from src.providers import Providers

config = GlobalConfig()
config.set_provider(Providers.ANTHROPIC, connection=connection)
```

#### Step 3: Use GlobalConfig in Agents

Any agent will automatically use the configured provider if no provider is passed:

```python
from src.base_agent.base_agent import BaseAgent

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    agent_name="MyAgent"
    # provider=None → uses GlobalConfig
)

response = agent.execute_query("What can you help me with?")
print(response.content)
```

#### Customize Provider Parameters

You can customize model parameters while using GlobalConfig:

```python
provider = config.get_current_provider_instance(
    model_id="claude-opus-4-20250514",
    max_tokens=4096,
    temperature=0.5
)

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
    agent_name="MyAgent"
)
```

### Available Connections

| Connection | Credentials |
|------------|-------------|
| `OpenAIConnection(api_key, base_url=None)` | OpenAI API key (or any OpenAI-compatible API) |
| `AnthropicConnection(api_key)` | Anthropic API key |
| `OCIConnection(oci_config)` | OCI config (file path or dict) |
| `IBMConnection(api_key, project_id, url)` | IBM Watson X credentials |

---

## Using the Logger

Obelix uses [Loguru](https://github.com/Delgan/loguru) for structured logging with automatic rotation and colored console output.

### Setup (Once at Application Start)

```python
from src.logging_config import setup_logging

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
from src.logging_config import get_logger

# Get a logger bound to current module
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

### Logging with Context

```python
# With variables (f-string)
logger.debug(f"Processing user {user_id} with params {params}")

# With exception traceback
try:
    risky_operation()
except Exception:
    logger.exception("Operation failed")  # Includes full traceback
```

### Log Output Format

**File** (with rotation at 50 MB, 7 days retention):
```
2024-01-15 10:30:45.123 | INFO     | src.agents.my_agent:execute:42 | Query executed
```

**Console** (with colors):
```
INFO     | src.agents.my_agent:execute | Query executed
```

---

## Project Structure

```
obelix/
├── src/
│   ├── base_agent/                # Agent infrastructure
│   │   ├── base_agent.py          # BaseAgent class
│   │   ├── hooks.py               # Hooks system
│   │   ├── orchestrator_decorator.py  # @orchestrator decorator
│   │   ├── subagent_decorator.py      # @subagent decorator
│   │   └── agents/                # Concrete agent implementations
│   ├── tools/                     # Tool implementations
│   │   ├── tool_base.py           # Abstract ToolBase
│   │   └── tool_decorator.py      # @tool decorator
│   ├── llm_providers/             # LLM provider implementations
│   │   ├── anthropic_provider.py
│   │   ├── oci_provider.py
│   │   ├── ibm_provider.py
│   │   └── ollama_provider.py
│   ├── connections/               # LLM connections (singleton clients)
│   ├── messages/                  # Message types
│   ├── logging_config.py          # Loguru configuration
│   └── config.py                  # Global configuration
├── config/                        # YAML configuration files
├── tests/                         # Test suite
├── LICENSE                        # Apache 2.0 License
├── setup.py                       # Package configuration
└── README.md                      # This file
```

## Supported LLM Providers

| Provider | Module | Description |
|----------|--------|-------------|
| OpenAI | `openai_provider.py` | GPT models + OpenAI-compatible APIs |
| Anthropic | `anthropic_provider.py` | Claude models (native SDK) |
| Oracle Cloud (OCI) | `oci_provider.py` | OCI Generative AI |
| IBM Watson | `ibm_provider.py` | WatsonX AI |
| Ollama | `ollama_provider.py` | Local models |
| vLLM | `vllm_provider.py` | Self-hosted inference |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Obelix Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
