# Tools Guide

Tools extend agents with custom capabilities. An agent can call tools to gather information, perform calculations, execute code, or request user input. This guide covers defining, registering, and executing tools in Obelix.

---

## Overview

Tools are a core part of the agent loop. When an LLM decides it needs to perform an action, it generates a tool call. The agent then executes the tool, collects the result, and feeds it back to the LLM for further reasoning.

### What Are Tools?

A tool is a callable component that performs a specific task. Tools have:
- **Name and description**: For LLM discovery and understanding
- **Input schema**: Parameters the LLM provides
- **Output schema** (optional): Expected response format (mainly for deferred tools)
- **execute()** method: The logic that runs when invoked
- **is_deferred flag**: Whether execution happens on the server or the client

### Tool Protocol

All tools in Obelix follow the `Tool` Protocol (structural typing). This means you don't inherit from a base class; instead, you implement the required members:

```python
from typing import Protocol, runtime_checkable
from obelix.core.model.tool_message import MCPToolSchema, ToolCall, ToolResult

@runtime_checkable
class Tool(Protocol):
    """Contract that all tools must satisfy."""

    tool_name: str
    tool_description: str
    is_deferred: bool

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
    def create_schema(self) -> MCPToolSchema: ...
```

Required members:
- **`tool_name: str`** - Unique identifier for the tool
- **`tool_description: str`** - Human-readable description of what the tool does
- **`is_deferred: bool`** - If `True`, execution is delegated to the client; if `False`, execution happens on the server
- **`async def execute(tool_call: ToolCall) -> ToolResult`** - Async method that executes the tool
- **`def create_schema() -> MCPToolSchema`** - Returns the tool's input and output schema

---

## Creating Tools with `@tool` Decorator

The `@tool` decorator is the primary way to create tools. It automatically handles schema generation, validation, and execution wrapping.

### Basic Syntax

```python
from obelix.core.tool.tool_decorator import tool
from pydantic import Field

@tool(name="tool_name", description="What the tool does", is_deferred=False)
class MyTool:
    param1: str = Field(..., description="Parameter description")
    param2: int = Field(default=10, description="Optional parameter")

    async def execute(self) -> dict:
        # Your logic here
        return {"result": f"Processed {self.param1}"}
```

### Decorator Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | Yes | Tool name for LLM discovery |
| `description` | str | Yes | Human-readable description |
| `is_deferred` | bool | No | If `True`, execution is delegated to client (default: `False`) |

### Input Parameters

Input parameters are defined using Pydantic `Field()` annotations on class attributes:

```python
from pydantic import Field

@tool(name="calculator", description="Basic arithmetic")
class Calculator:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field("add", description="Operation: add, subtract, multiply")

    async def execute(self) -> dict:
        if self.operation == "add":
            return {"result": self.a + self.b}
        elif self.operation == "subtract":
            return {"result": self.a - self.b}
        # ...
```

The decorator automatically extracts these fields and creates a Pydantic validation schema. When the LLM calls the tool, the decorator validates the provided arguments against this schema.

### Sync vs Async Execution

The decorator handles both sync and async `execute()` methods:

```python
# Sync execute
@tool(name="cpu_bound", description="CPU-intensive work")
class CPUBoundTool:
    data: str = Field(...)

    def execute(self) -> dict:  # Sync!
        # This runs in a thread pool (asyncio.to_thread)
        return {"processed": self.data.upper()}

# Async execute
@tool(name="network_bound", description="Network call")
class NetworkBoundTool:
    url: str = Field(...)

    async def execute(self) -> dict:  # Async!
        # This runs directly with await
        async with httpx.AsyncClient() as client:
            response = await client.get(self.url)
        return {"status": response.status_code}
```

Sync methods are automatically executed in a thread to avoid blocking the async event loop.

### Automatic Validation

The decorator validates input arguments using Pydantic:

```python
@tool(name="divide", description="Divide two numbers")
class DivideTool:
    a: float = Field(..., description="Numerator")
    b: float = Field(..., gt=0, description="Denominator (must be > 0)")

    async def execute(self) -> dict:
        return {"result": self.a / self.b}
```

If the LLM provides invalid arguments (e.g., `b=0`), the decorator automatically catches the validation error and returns a `ToolResult` with status `ERROR`.

### Inspecting Schema

Use `print_schema()` to quickly inspect the generated schema:

```python
@tool(name="my_tool", description="...")
class MyTool:
    param: str = Field(...)

    async def execute(self) -> dict:
        return {}

# Print the schema
MyTool.print_schema()

# Output:
# {
#   "name": "my_tool",
#   "description": "...",
#   "inputSchema": { ... },
#   "outputSchema": { ... }
# }
```

---

## OutputSchema (Optional)

For tools that return structured responses (especially deferred tools), you can declare an `OutputSchema` inner class:

```python
from pydantic import BaseModel

@tool(name="bash", description="Execute shell command", is_deferred=True)
class BashTool:
    command: str = Field(..., description="The command to run")

    class OutputSchema(BaseModel):
        """Expected response format from the client."""
        stdout: str = Field(default="", description="Standard output")
        stderr: str = Field(default="", description="Standard error")
        exit_code: int = Field(default=0, description="Process exit code")

    def execute(self) -> None:
        return None  # Deferred: client provides the output
```

The `OutputSchema`:
- **Is optional**: Not all tools need one. If omitted, defaults to `{"type": "object", "additionalProperties": true}`
- **Is detected automatically**: The decorator looks for an inner class named `OutputSchema` that inherits from `BaseModel`
- **Appears in tool schema**: Used in the tool's `outputSchema` field for client understanding
- **Is useful for deferred tools**: Tells the client/executor what format to return
- **Can be used on any tool**: Even normal (non-deferred) tools can declare their response contract

Example with OutputSchema used for documentation:

```python
@tool(name="get_user", description="Fetch user by ID")
class GetUserTool:
    user_id: int = Field(..., description="User ID")

    class OutputSchema(BaseModel):
        id: int
        name: str
        email: str
        created_at: str

    async def execute(self) -> dict:
        # Server-side execution
        user = await fetch_user(self.user_id)
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "created_at": user.created_at.isoformat(),
        }
```

---

## Tool Registration

Tools are registered on an agent instance or passed at construction time.

### Register After Construction

```python
from obelix.core.agent import BaseAgent
from obelix.core.tool.tool_decorator import tool

@tool(name="calculator", description="Arithmetic")
class Calculator:
    a: float = Field(...)
    b: float = Field(...)

    async def execute(self) -> dict:
        return {"sum": self.a + self.b}

agent = BaseAgent(system_message="You are helpful")
agent.register_tool(Calculator())
```

### Pass at Construction

```python
@tool(name="calculator", description="Arithmetic")
class Calculator:
    # ...

# Pass class or instance
agent = BaseAgent(
    system_message="You are helpful",
    tools=[Calculator]  # Class or instance, both work
)
```

### Multiple Tools

```python
agent = BaseAgent(
    system_message="You are helpful",
    tools=[CalculatorTool, StringTool, NetworkTool]
)
```

### Access Registered Tools

```python
for tool in agent.registered_tools:
    print(tool.tool_name, tool.tool_description)
```

---

## Tool Execution Lifecycle

When an agent executes a query with tools, the following happens:

```
LLM generates tool calls
        ↓
BaseAgent detects tool calls
        ↓
BEFORE_TOOL_EXECUTION hook
        ↓
Find tool by name
        ↓
Tool.execute(tool_call) — runs in parallel for multiple tools
        ↓
AFTER_TOOL_EXECUTION hook
        ↓
Tool returns ToolResult
        ↓
[If deferred] Stop loop, yield to caller
[If normal] Continue loop, feed ToolResult back to LLM
        ↓
ON_TOOL_ERROR hook (if error occurred)
```

### Parallel Execution

All tools in the same iteration run in parallel using `asyncio.gather()`:

```python
# LLM calls: calculator, get_user, search_api (all at once)
# BaseAgent runs all three in parallel
# Results fed back together
```

### Hook Integration

You can intercept tool execution with hooks:

```python
from obelix.core.agent.hooks import AgentEvent, HookDecision

agent.on(AgentEvent.BEFORE_TOOL_EXECUTION).when(
    lambda s: s.tool_call.name == "dangerous_tool"
).handle(
    decision=HookDecision.RETRY,  # Skip the tool
    effects=[lambda s: print("Blocked dangerous_tool")]
)

agent.on(AgentEvent.ON_TOOL_ERROR).when(
    lambda s: "timeout" in s.error.lower()
).handle(
    decision=HookDecision.RETRY,  # Retry the whole iteration
)
```

See [Hooks API](hooks.md) for complete hook documentation.

### Tool Policy

Enforce tool call requirements using `ToolRequirement`:

```python
from obelix.core.model.tool_message import ToolRequirement

tool_policy = [
    ToolRequirement(
        tool_name="search",
        min_calls=1,
        require_success=True,
        error_message="Must call search_api at least once successfully"
    )
]

agent = BaseAgent(
    system_message="...",
    tool_policy=tool_policy,
)
```

---

## Normal Tools vs Deferred Tools

This is the key distinction in Obelix's tool system.

### Normal Tools (is_deferred=False)

Normal tools execute on the **server** (where the agent runs):

```python
@tool(name="calculator", description="Add two numbers", is_deferred=False)
class Calculator:
    a: float = Field(...)
    b: float = Field(...)

    async def execute(self) -> dict:
        return {"result": self.a + self.b}
```

**Flow**:
```
LLM generates ToolCall(name="calculator", arguments={"a": 5, "b": 3})
    ↓
Agent finds tool, calls execute()
    ↓
execute() runs on server, returns {"result": 8}
    ↓
Agent injects ToolResult into history
    ↓
LLM continues reasoning with the result
```

**Characteristics**:
- `execute()` returns a dict or object
- The dict becomes `ToolResult.result`
- `inputSchema` describes what the LLM must provide
- `outputSchema` is informational (describes the response format)
- Execution is synchronous from the LLM's perspective

### Deferred Tools (is_deferred=True)

Deferred tools delegate execution to a **client** (e.g., A2A client, CLI runner, human):

```python
@tool(name="bash", description="Execute shell command", is_deferred=True)
class BashTool:
    command: str = Field(..., description="The command to run")

    class OutputSchema(BaseModel):
        stdout: str = Field(default="", description="Standard output")
        stderr: str = Field(default="", description="Standard error")
        exit_code: int = Field(default=0, description="Process exit code")

    def execute(self) -> None:
        return None  # Always return None for deferred tools
```

**Flow**:
```
LLM generates ToolCall(name="bash", arguments={"command": "ls -la"})
    ↓
Agent finds tool, calls execute()
    ↓
execute() returns None (deferred signature)
    ↓
Agent detects is_deferred=True + None result
    ↓
Agent stops loop and yields StreamEvent(deferred_tool_calls=[...])
    ↓
Caller/Consumer receives the tool call
    ↓
Consumer executes action (locally, remotely, or manually)
    ↓
Consumer sends result back to agent
    ↓
Agent validates result against OutputSchema
    ↓
Agent injects ToolMessage with result
    ↓
Agent resumes loop with resume_after_deferred()
    ↓
LLM continues reasoning with the result
```

**Characteristics**:
- `execute()` always returns `None`
- The agent loop **stops** when a deferred tool is called
- `inputSchema` describes what the LLM sends
- `OutputSchema` (required) describes what the client must send back
- Execution is asynchronous from the LLM's perspective (waits for client response)

### Comparison Table

| Aspect | Normal | Deferred |
|--------|--------|----------|
| `is_deferred` | `False` | `True` |
| `execute()` return | dict/object | `None` |
| Execution location | Server | Client |
| Blocking? | Server blocks briefly | LLM waits for client |
| `OutputSchema` | Optional | Recommended |
| Use cases | Calculate, search, filter | Shell, file I/O, user input |

### When to Use Each

**Use normal tools for**:
- Calculations, transformations
- Database queries
- API calls to external services
- Any operation the server can perform synchronously

**Use deferred tools for**:
- Shell command execution (delegate to client environment)
- File system operations (client may have different paths)
- User interaction (ask the client/user for input)
- Long-running operations (don't block the LLM)

---

## Built-in Tools

Obelix provides several built-in tools for common use cases.

### BashTool

Deferred tool for shell command execution. The client receives the command and executes it locally.

**Import**:
```python
from obelix.plugins.builtin import BashTool
```

**Usage**:
```python
agent = BaseAgent(system_message="You are helpful", tools=[BashTool])

# LLM can call: bash -c "git status"
# Agent stops, yields deferred_tool_calls
# Client executes locally, returns { stdout: "...", stderr: "...", exit_code: 0 }
# Agent resumes
```

**Schema**:
- **Input**: `command` (str), `description` (str), `timeout` (int, 1-600s), `working_directory` (optional str)
- **Output**: `stdout`, `stderr`, `exit_code`

### AskUserQuestionTool

Interactive tool for asking users questions with structured options. Blocks on stdin.

**Import**:
```python
from obelix.plugins.builtin.ask_user_question_tool import AskUserQuestionTool
```

**Usage**:
```python
agent = BaseAgent(system_message="...", tools=[AskUserQuestionTool])

# LLM calls the tool:
# {
#   "questions": [
#     {
#       "question": "Which library?",
#       "header": "Library",
#       "options": [
#         {"label": "React", "description": "UI library"},
#         {"label": "Vue", "description": "Framework"}
#       ],
#       "multi_select": false
#     }
#   ]
# }

# Tool displays options, blocks on input(), returns user choice
```

**Schema**:
- **Input**: `questions` (list of Question objects with options)
- **Output**: dict with `answers` mapping question text to chosen option(s)

### RequestUserInputTool

A2A-specific deferred tool for requesting input from remote clients (when using A2A server).

**Import**:
```python
from obelix.plugins.builtin.request_user_input_tool import RequestUserInputTool
```

**Usage**: Automatically registered when using `AgentFactory.a2a_serve()`. The LLM calls it when input is needed:

```python
# LLM calls:
# {
#   "question": "Which format?",
#   "options": [
#     {"label": "JSON", "description": "JSON format"},
#     {"label": "CSV", "description": "CSV format"}
#   ]
# }

# A2A executor emits input-required event
# Client responds with chosen option
# Agent resumes
```

**Schema**:
- **Input**: `question` (str), `options` (optional list of QuestionOption)
- **Output**: User's choice as string or None

---

## Other Tool Implementations

### SubAgentWrapper

Wraps another `BaseAgent` as a tool, enabling hierarchical agent composition.

**Usage**:
```python
from obelix.core.agent.subagent_wrapper import SubAgentWrapper
from obelix.core.agent.agent_factory import AgentFactory

# Create sub-agent
analyzer_agent = AnalyzerAgent(system_message="Analyze data")

# Wrap it as a tool
analyzer_wrapper = SubAgentWrapper(
    agent=analyzer_agent,
    name="analyzer",
    description="Analyzes data and extracts insights",
    stateless=True  # Each call gets a fresh copy
)

# Register on orchestrator
orchestrator = BaseAgent(system_message="...", tools=[analyzer_wrapper])
```

**Options**:
- `stateless=True`: Each invocation gets a copy of the agent (parallel-safe, no shared history)
- `stateless=False`: All invocations share the agent's conversation history (serialized with a lock)

**Schema**: Input schema is automatically generated from fields on the wrapped agent class.

### MCPTool

Wraps tools from MCP servers. Requires MCP plugin.

**Import**:
```python
from obelix.plugins.mcp.mcp_tool import MCPTool
```

See [Obelix documentation on MCP](../README.md#model-context-protocol-mcp-support) for details.

---

## Tool Message Types

### ToolCall

Represents a tool call made by the LLM:

```python
from obelix.core.model.tool_message import ToolCall

call = ToolCall(
    id="call_123abc",
    name="calculator",
    arguments={"a": 5, "b": 3}
)
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique ID for this tool call |
| `name` | str | Name of the tool to invoke |
| `arguments` | dict | Arguments provided by the LLM |

### ToolResult

Result of a single tool execution:

```python
from obelix.core.model.tool_message import ToolResult, ToolStatus

result = ToolResult(
    tool_name="calculator",
    tool_call_id="call_123abc",
    result={"sum": 8},
    status=ToolStatus.SUCCESS,
    execution_time=0.015
)
```

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Name of the tool |
| `tool_call_id` | str | ID of the corresponding ToolCall |
| `result` | Any | The return value from execute() |
| `status` | ToolStatus | SUCCESS, ERROR, or TIMEOUT |
| `error` | str \| None | Error message if status is ERROR |
| `execution_time` | float \| None | Time spent executing (seconds) |

### ToolStatus

Enum for tool execution status:

```python
from obelix.core.model.tool_message import ToolStatus

ToolStatus.SUCCESS  # Tool executed successfully
ToolStatus.ERROR    # Tool execution failed
ToolStatus.TIMEOUT  # Tool execution timed out
```

### ToolMessage

Message containing a batch of tool results, added to conversation history:

```python
from obelix.core.model.tool_message import ToolMessage

tool_results = [result1, result2]
message = ToolMessage(tool_results=tool_results)

# Automatically added to agent.conversation_history
```

### MCPToolSchema

Schema describing a tool's interface (generated automatically):

```python
from obelix.core.model.tool_message import MCPToolSchema

schema = MCPToolSchema(
    name="calculator",
    description="Add two numbers",
    inputSchema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    outputSchema={
        "type": "object",
        "properties": {
            "result": {"type": "number"}
        }
    }
)
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Tool identifier |
| `description` | str | Human-readable description |
| `inputSchema` | dict | JSON Schema for input parameters |
| `outputSchema` | dict \| None | JSON Schema for response |
| `title` | str \| None | Friendly name for UI |
| `annotations` | dict \| None | Behavioral hints |

---

## Building a Deferred Tool (Step-by-Step)

Here's a complete walkthrough of creating a deferred tool that requests file upload from the client:

### Step 1: Define the Tool Class

```python
from obelix.core.tool.tool_decorator import tool
from pydantic import BaseModel, Field

@tool(
    name="request_file_upload",
    description=(
        "Request the user to upload a file. The client will handle the file "
        "selection and upload, then return the file path or content."
    ),
    is_deferred=True
)
class RequestFileUploadTool:
    file_type: str = Field(
        ...,
        description="Type of file requested (e.g., 'CSV', 'JSON', 'PDF')"
    )
    description: str = Field(
        ...,
        description="User-friendly description of what file is needed"
    )
```

### Step 2: Declare OutputSchema

```python
    class OutputSchema(BaseModel):
        """Expected response from client after file upload."""

        file_path: str = Field(..., description="Path to the uploaded file")
        file_size: int = Field(..., description="Size in bytes")
        file_name: str = Field(..., description="Original filename")
```

### Step 3: Implement execute()

```python
    def execute(self) -> None:
        """
        Return None to signal deferred execution.

        The BaseAgent detects is_deferred=True + None result
        and stops the loop, yielding the tool call to the caller.
        """
        return None
```

### Step 4: Register on Agent

```python
agent = BaseAgent(
    system_message="You are a data analyst",
    tools=[RequestFileUploadTool]
)
```

### Step 5: What the Consumer Does

When the agent stops with `deferred_tool_calls`, the consumer receives:

```python
ToolCall(
    id="call_abc123",
    name="request_file_upload",
    arguments={
        "file_type": "CSV",
        "description": "Sales data for Q4"
    }
)
```

The consumer:
1. Presents a file upload dialog to the user
2. Validates the uploaded file
3. Creates a `ToolResult` matching the `OutputSchema`:

```python
result = ToolResult(
    tool_name="request_file_upload",
    tool_call_id="call_abc123",
    result={
        "file_path": "/tmp/sales_q4.csv",
        "file_size": 45280,
        "file_name": "sales_q4.csv"
    },
    status=ToolStatus.SUCCESS
)
```

4. Injects the result into the agent's history:

```python
agent.conversation_history.append(ToolMessage(tool_results=[result]))
```

5. Resumes the agent loop:

```python
async for event in agent.resume_after_deferred():
    if event.is_final:
        print(event.assistant_response.content)
```

### Full Example

```python
from obelix.core.tool.tool_decorator import tool
from obelix.core.agent import BaseAgent
from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from pydantic import BaseModel, Field

@tool(
    name="request_file_upload",
    description="Request user to upload a file",
    is_deferred=True
)
class RequestFileUploadTool:
    file_type: str = Field(...)
    description: str = Field(...)

    class OutputSchema(BaseModel):
        file_path: str
        file_size: int
        file_name: str

    def execute(self) -> None:
        return None

# Create and use
agent = BaseAgent(system_message="You are helpful", tools=[RequestFileUploadTool])

# Agent queries
response = agent.execute_query("I need to analyze sales data. Can you request a CSV file from the user?")

# Check if deferred
for event in agent.execute_query_stream("..."):
    if event.deferred_tool_calls:
        print(f"Deferred: {event.deferred_tool_calls}")

        # Consumer handles the tool call
        result = ToolResult(
            tool_name="request_file_upload",
            tool_call_id=event.deferred_tool_calls[0].id,
            result={
                "file_path": "/data/sales.csv",
                "file_size": 10240,
                "file_name": "sales.csv"
            },
            status=ToolStatus.SUCCESS
        )

        # Inject and resume
        agent.conversation_history.append(ToolMessage(tool_results=[result]))
        async for resume_event in agent.resume_after_deferred():
            if resume_event.is_final:
                print(f"Final: {resume_event.assistant_response.content}")
```

---

## Common Patterns

### Tool with Constructor Dependencies

Tools can have `__init__` for dependency injection:

```python
@tool(name="database_query", description="Query the database")
class DatabaseQueryTool:
    connection: DatabaseConnection  # Set in __init__

    query: str = Field(..., description="SQL query")

    def __init__(self, connection: DatabaseConnection):
        self.connection = connection

    async def execute(self) -> dict:
        result = await self.connection.execute(self.query)
        return {"rows": result}

# Usage
db_conn = DatabaseConnection("postgresql://...")
agent = BaseAgent(system_message="...", tools=[DatabaseQueryTool(db_conn)])
```

### Tool with Validation

Use Pydantic's validation to enforce constraints:

```python
from pydantic import Field, field_validator

@tool(name="safe_divide", description="Divide two numbers safely")
class SafeDivideTool:
    a: float = Field(...)
    b: float = Field(
        ...,
        gt=0,  # Must be > 0
        description="Divisor (must not be zero)"
    )

    async def execute(self) -> dict:
        return {"result": self.a / self.b}
```

### Tool with Rich Error Handling

The decorator captures exceptions automatically, but you can add extra context:

```python
@tool(name="api_call", description="Call external API")
class APICallTool:
    endpoint: str = Field(...)

    async def execute(self) -> dict:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.endpoint, timeout=5)
                response.raise_for_status()
                return {"data": response.json()}
        except httpx.TimeoutException:
            raise RuntimeError(f"API request timed out after 5s to {self.endpoint}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"API error: {e.response.status_code} {e.response.text}")
```

---

## Troubleshooting

### "Tool validation failed"

**Symptom**: `ToolResult` has status=ERROR with validation error message.

**Fix**: Ensure all tool parameters have `Field()` definitions with descriptions:

```python
# ❌ Wrong
@tool(name="bad", description="...")
class BadTool:
    param: str

# ✅ Correct
@tool(name="good", description="...")
class GoodTool:
    param: str = Field(..., description="Parameter description")
```

### "Tool not found"

**Symptom**: `ToolResult` error says "Tool not found or not executable".

**Fix**: Register the tool before executing a query:

```python
# ❌ Wrong
agent = BaseAgent(system_message="...")
response = agent.execute_query("Use my_tool")

# ✅ Correct
agent = BaseAgent(system_message="...", tools=[MyTool])
response = agent.execute_query("Use my_tool")
```

### "Deferred tool always stops the loop"

**Symptom**: Agent stops even when you want it to continue after the deferred tool.

**Expected behavior**: Deferred tools **always** stop the loop. The consumer must manually resume with `resume_after_deferred()` after injecting the result. This is by design.

### "Deferred tool execute() not returning None"

**Symptom**: Deferred tool's `execute()` returns a dict instead of None, causing agent to not stop.

**Fix**: Deferred tools must always return `None`:

```python
# ❌ Wrong
@tool(name="bash", is_deferred=True)
class BashTool:
    def execute(self) -> None:
        return {"stdout": ""}  # Returns dict, not None!

# ✅ Correct
@tool(name="bash", is_deferred=True)
class BashTool:
    def execute(self) -> None:
        return None  # Always None
```

---

## See Also

- [BaseAgent Guide](base_agent.md#registering-tools) - Tool registration and execution
- [Agent Factory Guide](agent_factory.md#registering-sub-agents) - Creating sub-agents as tools
- [Hooks API](hooks.md) - Intercepting tool execution with hooks
- [A2A Server Guide](a2a_server.md) - Exposing agents with tools over HTTP
