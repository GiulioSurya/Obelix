# Tools Guide

Tools extend agents with custom capabilities — calculations, API calls, shell commands, user interaction, and more. This guide covers everything you need to create, register, and use tools in Obelix.

---

## Quick Start

```python
from obelix.core.tool.tool_decorator import tool
from pydantic import Field

@tool(name="calculator", description="Add two numbers")
class Calculator:
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    async def execute(self) -> dict:
        return {"result": self.a + self.b}
```

Register it on an agent:

```python
from obelix.core.agent import BaseAgent

agent = BaseAgent(
    system_message="You are a helpful assistant.",
    provider=provider,
    tools=[Calculator],
)
```

---

## The `@tool` Decorator

The decorator is the primary way to create tools. It handles schema generation, input validation, error handling, and thread safety automatically.

```python
@tool(name="my_tool", description="What the tool does", is_deferred=False)
class MyTool:
    param: str = Field(..., description="Parameter description")

    async def execute(self) -> dict:
        return {"result": self.param.upper()}
```

### Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | Required | Tool name the LLM uses to call it |
| `description` | str | Required | What the tool does (shown to LLM) |
| `is_deferred` | bool | `False` | If `True`, execution is delegated to the client |

### Input Parameters

Define inputs as Pydantic `Field()` annotations on the class. The decorator extracts them into a validation schema automatically.

```python
@tool(name="search", description="Search a database")
class SearchTool:
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    include_metadata: bool = Field(default=False, description="Include metadata")

    async def execute(self) -> dict:
        # self.query, self.limit, self.include_metadata are already validated
        results = await db.search(self.query, limit=self.limit)
        return {"results": results}
```

Pydantic constraints (`ge`, `le`, `gt`, `max_length`, etc.) are enforced automatically. If the LLM provides invalid arguments, the decorator returns a `ToolResult` with status `ERROR` and the validation message — the LLM can then self-correct.

### Sync vs Async Execute

Both work. Sync methods run in a thread pool automatically:

```python
# Async (preferred for I/O-bound work)
@tool(name="fetch", description="Fetch a URL")
class FetchTool:
    url: str = Field(...)

    async def execute(self) -> dict:
        async with httpx.AsyncClient() as client:
            r = await client.get(self.url)
        return {"status": r.status_code, "body": r.text}

# Sync (fine for CPU-bound or simple operations)
@tool(name="hash", description="Compute SHA256 hash")
class HashTool:
    text: str = Field(...)

    def execute(self) -> dict:  # runs in asyncio.to_thread()
        import hashlib
        return {"hash": hashlib.sha256(self.text.encode()).hexdigest()}
```

### Constructor Dependencies

Tools can have `__init__` for dependency injection. Constructor parameters are **not** part of the input schema — only `Field()` annotations are.

```python
@tool(name="sql_query", description="Execute a SQL query")
class SQLQueryTool:
    query: str = Field(..., description="SQL query to execute")

    def __init__(self, connection):
        self._connection = connection

    async def execute(self) -> dict:
        rows = await self._connection.execute(self.query)
        return {"rows": rows, "count": len(rows)}

# Register with dependency
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[SQLQueryTool(db_connection)],
)
```

### Thread Safety

The decorator creates an isolated copy of the tool instance for each execution via `copy.copy()`. This means multiple parallel tool calls don't share mutable state — no locks needed.

### Inspecting the Schema

Every `@tool`-decorated class gets a `print_schema()` classmethod:

```python
Calculator.print_schema()
```

Output:
```json
{
  "name": "calculator",
  "description": "Add two numbers",
  "inputSchema": {
    "properties": {
      "a": { "description": "First number", "type": "number" },
      "b": { "description": "Second number", "type": "number" }
    },
    "required": ["a", "b"],
    "type": "object"
  },
  "outputSchema": {
    "type": "object",
    "additionalProperties": true
  }
}
```

---

## OutputSchema

By default, a tool's `outputSchema` is a generic `{"type": "object", "additionalProperties": true}`. You can declare an explicit `OutputSchema` inner class to document the response contract.

### Why Use It

1. **For deferred tools**: the client needs to know what format to return. Without `OutputSchema`, the client has to guess. With it, the A2A executor validates the response automatically.

2. **For documentation**: even on normal tools, `OutputSchema` makes the tool self-describing. `print_schema()` shows both input and output contracts, which is useful for debugging and for other systems that consume the schema.

3. **For the LLM**: the output schema is included in the tool definition sent to the LLM, helping it understand what data it will get back.

### How to Declare It

Define a Pydantic `BaseModel` named `OutputSchema` as an inner class:

```python
from pydantic import BaseModel, Field

@tool(name="geocode", description="Convert address to coordinates")
class GeocodeTool:
    address: str = Field(..., description="Address to geocode")

    class OutputSchema(BaseModel):
        latitude: float = Field(..., description="Latitude in decimal degrees")
        longitude: float = Field(..., description="Longitude in decimal degrees")
        confidence: float = Field(..., description="Confidence score 0-1")

    async def execute(self) -> dict:
        result = await geocoding_api.lookup(self.address)
        return {
            "latitude": result.lat,
            "longitude": result.lng,
            "confidence": result.score,
        }
```

The decorator detects it automatically — no extra configuration needed.

### OutputSchema on Deferred Tools

For deferred tools, `OutputSchema` is essential. It tells the client exactly what to return:

```python
@tool(name="bash", description="Execute shell command", is_deferred=True)
class BashTool:
    command: str = Field(..., description="The command to run")

    class OutputSchema(BaseModel):
        stdout: str = Field(default="", description="Standard output")
        stderr: str = Field(default="", description="Standard error")
        exit_code: int = Field(default=0, description="Process exit code")

    def execute(self) -> None:
        return None  # deferred
```

When the A2A executor receives the client's response, it validates it against `OutputSchema`. If the client returns `{"stdout": "hello", "exit_code": 0}`, the executor accepts it. If the client returns something unexpected, it falls back to `{"answer": raw_text}`.

### When to Skip It

- Simple tools where the output is obvious (e.g., `{"result": 42}`)
- Prototype/internal tools where no external consumer reads the schema
- Tools where the output structure varies dynamically

---

## System Prompt Fragments

A tool can contribute text to the agent's system message by defining a `system_prompt_fragment()` method. When the tool is registered, `BaseAgent.register_tool()` calls this method and appends the returned string to the system message.

### Why Use It

Some tools need the LLM to know about the execution environment — which shell is available, what database schema exists, which APIs are accessible. Instead of requiring the caller to manually build a system message with this context, the tool injects it automatically.

### How It Works

```python
@tool(name="sql_query", description="Execute SQL")
class SQLQueryTool:
    query: str = Field(...)

    def __init__(self, schema_info: str):
        self._schema_info = schema_info

    async def execute(self) -> dict:
        ...

    def system_prompt_fragment(self) -> str | None:
        return (
            "\n\n## Database Schema\n"
            f"{self._schema_info}\n"
            "Write standard SQL. Use double quotes for identifiers."
        )
```

When registered:

```python
agent = BaseAgent(
    system_message="You are a data analyst.",
    provider=provider,
    tools=[SQLQueryTool(schema_info="Tables: users(id, name, email), orders(id, user_id, total)")],
)
# agent.system_message.content now includes:
# "You are a data analyst.\n\n## Database Schema\nTables: users(...), orders(...)\n..."
```

The LLM sees the schema as part of its instructions and generates better queries.

### Rules

- Return `str` to inject content, `None` to skip
- Called once at registration time, not on every execution
- The fragment is appended to the end of the existing system message
- Keep fragments focused — don't repeat information already in the system message

### Built-in Example: BashTool

When `BashTool` has an executor, its `system_prompt_fragment()` injects:

```
## Shell Environment
- Platform: Windows
- OS Version: Windows 11 Enterprise 10.0.26200
- Shell: bash (use Unix shell syntax, not Windows -- e.g., /dev/null not NUL, forward slashes in paths)
- Working directory: /c/Users/GLoverde/Projects
- Drive mounts: C: -> /c/, D: -> /d/
- Use Unix-style paths with forward slashes (e.g. /c/Users/... not C:\Users\...)
```

This tells the LLM which shell it's working with, so it generates compatible commands. When `BashTool` is deferred (no executor), `system_prompt_fragment()` returns `None` — the client already knows its own environment.

---

## Normal Tools vs Deferred Tools

### Normal Tools (`is_deferred=False`)

Execute on the **server** where the agent runs. The agent loop continues seamlessly.

```
LLM → ToolCall → execute() runs on server → ToolResult → LLM continues
```

### Deferred Tools (`is_deferred=True`)

Delegate execution to the **client**. The agent loop stops and waits for the client to provide the result.

```
LLM → ToolCall → execute() returns None → loop stops
    → client executes → sends result back → loop resumes → LLM continues
```

### When to Use Each

| Normal | Deferred |
|--------|----------|
| Calculations, transformations | Shell commands on client machine |
| Database queries | File operations on client filesystem |
| API calls to external services | User interaction (ask for clarification) |
| Any server-side operation | Long-running client-side operations |

### Dynamic `is_deferred`

A tool can switch between modes at instance creation time. `BashTool` does this:

```python
@tool(name="bash", description="Execute shell command", is_deferred=True)
class BashTool:
    command: str = Field(...)

    def __init__(self, executor=None):
        self._executor = executor
        self.is_deferred = executor is None  # override class-level default

    async def execute(self) -> dict | None:
        if self._executor is None:
            return None  # deferred: client executes
        return await self._executor.execute(self.command)  # normal: server executes
```

- `BashTool()` → deferred (client executes)
- `BashTool(executor=LocalShellExecutor())` → normal (server executes)

---

## Tool Registration

### At Construction

```python
agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[Calculator, SearchTool()],  # class or instance, both work
)
```

Classes are auto-instantiated. Use instances when the tool needs constructor dependencies.

### After Construction

```python
agent = BaseAgent(system_message="...", provider=provider)
agent.register_tool(SQLQueryTool(connection=db))
```

### What `register_tool()` Does

1. Adds the tool to `agent.registered_tools`
2. Calls `system_prompt_fragment()` if the tool defines it
3. Appends the fragment (if any) to `agent.system_message.content`

---

## Tool Execution Lifecycle

When the LLM generates tool calls, the agent dispatches them:

```
1. BEFORE_TOOL_EXECUTION hook (per tool)
2. Execute all tools in parallel (asyncio.gather)
3. AFTER_TOOL_EXECUTION hook (per tool)
4. ON_TOOL_ERROR hook (if any tool failed)
5. If deferred tools detected → stop loop, yield deferred_tool_calls
6. If exit_on_success matches → end loop
7. Otherwise → feed results to LLM, continue loop
```

### Parallel Execution

All tool calls in the same LLM response run in parallel. Each gets an isolated copy of the tool instance (via `copy.copy`), so no shared mutable state.

### Error Handling

The `@tool` decorator wraps `execute()` in a try/except:
- **ValidationError** (bad arguments from LLM): returns `ToolResult(status=ERROR)` with formatted validation message
- **Any exception**: returns `ToolResult(status=ERROR, error=str(e))`
- Error messages are auto-truncated to 2000 characters

The LLM sees the error and can self-correct in the next iteration.

---

## Built-in Tools

### BashTool

Shell command execution with pluggable backend.

```python
from obelix.plugins.builtin import BashTool
```

**Deferred mode** (default — client executes):

```python
agent = BaseAgent(system_message="...", provider=provider, tools=[BashTool()])
```

**Local mode** (server executes via `LocalShellExecutor`):

```python
from obelix.adapters.outbound.shell import LocalShellExecutor

agent = BaseAgent(
    system_message="...",
    provider=provider,
    tools=[BashTool(executor=LocalShellExecutor())],
)
```

**Input fields**: `command` (str), `description` (str), `timeout` (1-600s), `working_directory` (optional)

**OutputSchema**: `stdout`, `stderr`, `exit_code`

**System prompt fragment**: in local mode, injects platform, shell, cwd, and drive mount info. In deferred mode, returns `None`.

#### LocalShellExecutor

Auto-detects the best shell: Git Bash > WSL bash > cmd.exe (Windows), bash > zsh > sh (Unix).

```python
from obelix.adapters.outbound.shell import LocalShellExecutor

executor = LocalShellExecutor()          # auto-detect shell
executor = LocalShellExecutor(shell="/bin/zsh")  # explicit shell

# Direct usage (outside of BashTool)
result = await executor.execute("ls -la", timeout=10, working_directory="/tmp")
# {"stdout": "...", "stderr": "", "exit_code": 0}
```

Key features:
- Uses `create_subprocess_exec(shell, "-c", command)` — not `create_subprocess_shell` (which uses cmd.exe on Windows)
- Output truncated to 50K chars to avoid bloating LLM context
- Timeout with process kill, returns `exit_code=-1`
- `shell_info` property: platform, shell path, shell name, cwd, home, drive mounts

#### Custom Executors

Implement `AbstractShellExecutor` for Docker, SSH, cloud shells, etc.:

```python
from obelix.ports.outbound.shell_executor import AbstractShellExecutor

class DockerExecutor(AbstractShellExecutor):
    def __init__(self, container: str):
        self._container = container

    @property
    def shell_info(self) -> dict:
        return {"platform": "Docker", "shell": "/bin/bash", "shell_name": "bash"}

    async def execute(self, command, timeout=120, working_directory=None):
        # docker exec logic
        return {"stdout": ..., "stderr": ..., "exit_code": ...}
```

### RequestUserInputTool

A2A-specific deferred tool for requesting input from remote clients. Auto-registered by `AgentFactory.a2a_serve()`.

```python
from obelix.plugins.builtin.request_user_input_tool import RequestUserInputTool
```

**Input fields**: `question` (str), `options` (optional list of `QuestionOption` with `label` and `description`)

**No OutputSchema** — the A2A executor falls back to `{"answer": text}`.

When the LLM needs clarification, it calls this tool. The A2A server emits `input-required`, the client responds, and the agent resumes.

---

## SubAgentWrapper

Wraps another `BaseAgent` as a callable tool for hierarchical composition. You don't create this directly — use `agent.register_agent()`:

```python
coordinator.register_agent(
    analyzer_agent,
    name="analyzer",
    description="Analyzes data and generates insights",
    stateless=True,
)
```

- `stateless=True`: each call gets a fresh copy (parallel-safe, no shared history)
- `stateless=False`: calls share conversation history (serialized with a lock)

See [BaseAgent Guide — Registering Sub-Agents](base_agent.md#registering-sub-agents).

---

## Tool Protocol

All tools satisfy the `Tool` Protocol (structural typing — no base class inheritance):

```python
@runtime_checkable
class Tool(Protocol):
    tool_name: str
    tool_description: str
    is_deferred: bool

    async def execute(self, tool_call: ToolCall) -> ToolResult: ...
    def create_schema(self) -> MCPToolSchema: ...
```

The `@tool` decorator, `SubAgentWrapper`, and `MCPTool` all satisfy this protocol. You can also implement it manually for advanced use cases.

---

## Message Types Reference

### ToolCall

What the LLM generates when it wants to use a tool:

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique ID for this call |
| `name` | str | Tool name |
| `arguments` | dict | Arguments from the LLM |

### ToolResult

What `execute()` produces:

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | str | Tool name |
| `tool_call_id` | str | Matching ToolCall ID |
| `result` | Any | Return value from execute() |
| `status` | ToolStatus | `SUCCESS`, `ERROR`, or `TIMEOUT` |
| `error` | str \| None | Error message (auto-truncated to 2000 chars) |
| `execution_time` | float \| None | Seconds |

### ToolMessage

Batch of results added to conversation history after tool execution:

```python
ToolMessage(tool_results=[result1, result2])
```

### MCPToolSchema

Schema describing a tool's interface (generated by `create_schema()`):

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Tool identifier |
| `description` | str | Human-readable description |
| `inputSchema` | dict | JSON Schema for input parameters |
| `outputSchema` | dict \| None | JSON Schema for response (from `OutputSchema` or generic default) |

---

## See Also

- [BaseAgent Guide](base_agent.md) — tool registration, execution flow, deferred tools
- [Agent Factory](agent_factory.md) — composing agents as sub-agent tools
- [Hooks Guide](hooks.md) — intercepting tool execution
- [A2A Server](a2a_server.md) — how deferred tools work in the A2A protocol
