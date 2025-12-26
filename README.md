# Obelix

![Python](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)

A multi-provider LLM agent framework with tool support, hooks system, and seamless integration with major AI providers.

## Features

- **Multi-Provider Support**: Anthropic, Oracle Cloud (OCI), IBM Watson, Ollama, vLLM
- **Tool System**: Declarative tool creation with automatic validation
- **Hooks System**: Intercept and modify agent behavior at runtime
- **Async/Sync Execution**: Support for both synchronous and asynchronous workflows
- **Loguru Logging**: Structured logging with rotation and color output

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
pip install -e ".[anthropic]"          # Anthropic Claude
pip install -e ".[oci]"                # Oracle Cloud Infrastructure
pip install -e ".[ibm]"                # IBM Watson X
pip install -e ".[ollama]"             # Ollama (local models)

# Install with database support
pip install -e ".[oracle]"             # Oracle Database
pip install -e ".[postgres]"           # PostgreSQL

# Install multiple extras
pip install -e ".[anthropic,oracle]"

# Install all LLM providers
pip install -e ".[all-llm]"

# Install all databases
pip install -e ".[all-db]"

# Install everything
pip install -e ".[all]"

# Install with development tools
pip install -e ".[dev]"
```

### Available Extras

| Extra | Description |
|-------|-------------|
| `anthropic` | Anthropic Claude provider |
| `oci` | Oracle Cloud Infrastructure Generative AI |
| `ibm` | IBM Watson X AI |
| `ollama` | Ollama local models |
| `oracle` | Oracle Database connection |
| `postgres` | PostgreSQL connection |
| `all-llm` | All LLM providers |
| `all-db` | All database connections |
| `all` | Everything |
| `dev` | Development tools (pytest, coverage) |

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

Tools allow agents to perform concrete actions like executing SQL queries, creating charts, or calling external APIs.

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

    async def execute(self) -> dict:
        """Execute the calculation. Attributes are already populated by the decorator."""
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
@tool(name="sql_executor", description="Executes SQL queries on the database")
class SqlExecutorTool(ToolBase):
    sql_query: str = Field(..., description="Valid SQL query to execute")

    def __init__(self, db_connection):
        self.db_connection = db_connection

    async def execute(self) -> dict:
        result = self.db_connection.execute_query(self.sql_query)
        return {
            "columns": result.columns,
            "data": result.data,
            "row_count": len(result.data)
        }
```

### Key Points

- Use the `@tool` decorator with `name` and `description` (both required)
- Define input parameters using Pydantic `Field`
- Implement `async def execute(self)` method
- The decorator handles validation, error wrapping, and result formatting

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

```python
from src.base_agent.base_agent import BaseAgent
from src.tools.my_tool import MyTool

class ToolEquippedAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            system_message="You are an assistant with calculation capabilities.",
            agent_name="CalculatorAgent"
        )

        # Register tools
        self.register_tool(CalculatorTool())
```

### Agent with Hooks

Hooks allow intercepting events during the agent lifecycle:

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

    def _is_invalid_identifier(self, ctx) -> bool:
        return (
            ctx.error is not None and
            "invalid identifier" in ctx.error and
            not self._schema_injected
        )

    def _create_schema_message(self, ctx) -> HumanMessage:
        self._schema_injected = True
        return HumanMessage(content="Database schema: ... Please correct the query.")
```

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
│   ├── base_agent/           # Agent infrastructure
│   │   ├── base_agent.py     # BaseAgent class
│   │   ├── hooks.py          # Hooks system
│   │   └── agents/           # Concrete agent implementations
│   ├── tools/                # Tool implementations
│   │   ├── tool_base.py      # Abstract ToolBase
│   │   └── tool_decorator.py # @tool decorator
│   ├── llm_providers/        # LLM provider implementations
│   │   ├── anthropic_provider.py
│   │   ├── oci_provider.py
│   │   ├── ibm_provider.py
│   │   └── ollama_provider.py
│   ├── connections/          # Database and LLM connections
│   ├── messages/             # Message types
│   ├── logging_config.py     # Loguru configuration
│   └── config.py             # Global configuration
├── config/                   # YAML configuration files
├── tests/                    # Test suite
├── LICENSE                   # Apache 2.0 License
├── setup.py                  # Package configuration
└── README.md                 # This file
```

## Supported LLM Providers

| Provider | Module | Description |
|----------|--------|-------------|
| Anthropic | `anthropic_provider.py` | Claude models |
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
