# A2A Server Guide

The [A2A (Agent-to-Agent) protocol](https://a2a-protocol.org/latest/) enables Obelix
agents to be exposed as HTTP services compliant with the A2A specification. This allows
agents to be discovered, invoked, and orchestrated by other systems using standardized
JSON-RPC 2.0 methods.

> **Spec version**: this implementation targets the
> [A2A Protocol Specification RC v1.0](https://a2a-protocol.org/latest/specification/).
> For a detailed gap analysis and roadmap, see [A2A Compliance](a2a_compliance.md).

This guide covers:
- How the A2A server works
- Exposing agents via `factory.serve()`
- Endpoints and JSON-RPC methods
- Complete working examples
- Deployment considerations

---

## What is A2A?

[A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/topics/what-is-a2a/) is an open
standard for inter-agent communication, originally developed by Google and donated to
the Linux Foundation. It defines:

- A machine-readable **Agent Card** that describes capabilities, skills, and metadata
  ([Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/))
- HTTP endpoints for sending messages and managing tasks
  ([Specification](https://a2a-protocol.org/latest/specification/))
- **JSON-RPC 2.0** as the primary method call mechanism (one of 3 supported bindings)
- **Task management** with status tracking, conversation history, and artifacts
  ([Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/))

A2A complements [MCP (Model Context Protocol)](https://modelcontextprotocol.io/): while
MCP connects agents to tools and data sources, A2A enables agents to collaborate with
each other as peers
([A2A and MCP](https://a2a-protocol.org/latest/topics/a2a-and-mcp/)).

An Obelix agent exposed via A2A becomes a network service that other agents (or systems)
can discover and invoke.

### Current Implementation Scope

Obelix currently implements the **A2A Server** role (inbound adapter):

| Capability | Status |
|------------|--------|
| Agent Card (discovery) | Implemented |
| SendMessage | Implemented |
| GetTask | Implemented |
| ListTasks | Implemented |
| CancelTask | Implemented |
| Streaming (SSE) | Not yet |
| Push Notifications | Not yet |
| A2A Client (outbound) | Not yet |

For the full compliance matrix, see [A2A Compliance](a2a_compliance.md).

---

## Quick Start

### Starting an A2A Server

Use `AgentFactory.serve()` to expose an agent as an A2A service:

```python
from obelix.core.agent.agent_factory import AgentFactory

factory = AgentFactory()
factory.register("my_agent", MyAgent, subagent_description="Does work")

# Start the A2A server
factory.serve(
    "my_agent",
    host="0.0.0.0",
    port=8000,
    description="My awesome agent",
)
```

The server will start on `http://0.0.0.0:8000` and log:
```
AgentFactory: starting A2A server | agent=my_agent host=0.0.0.0 port=8000
```

### Dependencies

The A2A server requires additional dependencies. Install with:

```bash
uv sync --extra serve
```

This installs:
- `fastapi>=0.115.0` - Web framework
- `uvicorn[standard]>=0.30.0` - ASGI server
- `a2a-sdk>=0.3.10` - A2A protocol support

---

## Exposing Agents with Sub-Agents

A2A servers can expose a coordinator agent with multiple sub-agents. Sub-agents are automatically registered as "skills" in the agent card:

```python
from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory

class MathAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You are a math expert with a calculator tool.",
            **kwargs
        )

class ReportAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You generate formatted reports.",
            **kwargs
        )

class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            system_message="You coordinate math and reporting tasks.",
            **kwargs
        )

factory = AgentFactory()
factory.register(
    "math",
    MathAgent,
    subagent_description="Performs mathematical calculations"
)
factory.register(
    "report",
    ReportAgent,
    subagent_description="Generates formatted reports"
)
factory.register("coordinator", CoordinatorAgent)

# Serve the coordinator with math and report as sub-agents
factory.serve(
    "coordinator",
    host="0.0.0.0",
    port=8000,
    subagents=["math", "report"],
    description="Coordinator with math and reporting capabilities",
)
```

---

## Endpoints

The A2A server exposes three HTTP endpoints:

### Agent Card
```
GET /.well-known/agent-card.json
```

> **Note**: the spec defines the well-known path as `/.well-known/a2a/agent-card`
> ([RFC 8615](https://datatracker.ietf.org/doc/html/rfc8615)). The current
> implementation uses `/.well-known/agent-card.json`. This will be aligned in a
> future release (see [A2A Compliance > Agent Card](a2a_compliance.md#agent-card-compliance)).

Returns the A2A-compliant agent card (metadata and capabilities):

```json
{
  "id": "coordinator",
  "name": "coordinator",
  "version": "0.1.0",
  "description": "Coordinator with math and reporting capabilities",
  "provider": {
    "organization": "Obelix"
  },
  "url": "http://localhost:8000",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "capabilities": {
    "streaming": false,
    "pushNotifications": false,
    "extendedAgentCard": false
  },
  "skills": [
    {
      "id": "math",
      "name": "math",
      "description": "Performs mathematical calculations"
    },
    {
      "id": "report",
      "name": "report",
      "description": "Generates formatted reports"
    }
  ],
  "securitySchemes": {},
  "security": []
}
```

### Health Check
```
GET /health
```

Returns server status:

```json
{
  "status": "ok"
}
```

### JSON-RPC 2.0 Dispatcher
```
POST /
```

Accepts JSON-RPC 2.0 requests for agent methods. See "JSON-RPC Methods" below.

---

## JSON-RPC Methods

Ref: [Specification > Operations](https://a2a-protocol.org/latest/specification/)

All methods use JSON-RPC 2.0 format:

```json
{
  "jsonrpc": "2.0",
  "method": "MethodName",
  "params": { ... },
  "id": "request-id"
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "result": { ... }
}
```

Or on error:
```json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "error": {
    "code": -32001,
    "message": "Task not found"
  }
}
```

### SendMessage

Send a message to the agent and receive a response. Creates a task to track execution.

> **Spec note**: the full spec supports a `configuration` object with
> `acceptedOutputModes`, `historyLength`, `returnImmediately`, and
> `taskPushNotificationConfig`. These are not yet implemented.
> See [A2A Compliance > Operations](a2a_compliance.md#operations-json-rpc-methods).

**Method**: `SendMessage`

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `message` | Object | Message object with `parts` array |
| `message.parts` | Array | Array of message parts (currently text only) |
| `message.parts[].text` | String | Text content of the message |
| `contextId` | String (Optional) | Context identifier for grouping related tasks |

**Example Request**:

```json
{
  "jsonrpc": "2.0",
  "method": "SendMessage",
  "params": {
    "message": {
      "parts": [
        {
          "text": "Calculate 15 * 7 and create a report"
        }
      ]
    },
    "contextId": "session-123"
  },
  "id": "msg-001"
}
```

**Response**:

```json
{
  "jsonrpc": "2.0",
  "id": "msg-001",
  "result": {
    "id": "task-uuid",
    "contextId": "session-123",
    "status": {
      "state": "completed"
    },
    "artifacts": [
      {
        "artifactId": "artifact-uuid",
        "parts": [
          {
            "type": "text",
            "text": "The calculation is complete: 15 * 7 = 105. I've generated a report..."
          }
        ]
      }
    ],
    "metadata": {}
  }
}
```

**Task Lifecycle**:

1. Task created with state `"working"`
2. Agent processes the message asynchronously
3. On success: state becomes `"completed"`, response stored in artifacts
4. On failure: state becomes `"failed"`, error message in status

> **Spec note**: the full spec defines additional states: `INPUT_REQUIRED` (agent needs
> more info from user), `AUTH_REQUIRED`, and `REJECTED`. These are not yet supported.
> See [Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/).

---

### GetTask

Retrieve a previously created task by ID.

**Method**: `GetTask`

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | String | Task ID (from SendMessage response) |

**Example Request**:

```json
{
  "jsonrpc": "2.0",
  "method": "GetTask",
  "params": {
    "id": "task-uuid"
  },
  "id": "get-001"
}
```

**Response**:

```json
{
  "jsonrpc": "2.0",
  "id": "get-001",
  "result": {
    "id": "task-uuid",
    "contextId": "session-123",
    "status": {
      "state": "completed"
    },
    "artifacts": [ ... ],
    "metadata": {}
  }
}
```

**Error** (task not found):

```json
{
  "jsonrpc": "2.0",
  "id": "get-001",
  "error": {
    "code": -32001,
    "message": "Task 'invalid-id' not found"
  }
}
```

---

### ListTasks

List tasks with optional filtering and pagination.

**Method**: `ListTasks`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contextId` | String (Optional) | - | Filter by context ID |
| `status` | String (Optional) | - | Filter by status: `"working"`, `"completed"`, `"failed"`, `"canceled"` |
| `pageSize` | Integer | 50 | Number of tasks per page |
| `pageToken` | String (Optional) | - | Token from previous response for pagination |

**Example Request** (all tasks):

```json
{
  "jsonrpc": "2.0",
  "method": "ListTasks",
  "params": {},
  "id": "list-001"
}
```

**Example Request** (filtered):

```json
{
  "jsonrpc": "2.0",
  "method": "ListTasks",
  "params": {
    "contextId": "session-123",
    "status": "completed",
    "pageSize": 10
  },
  "id": "list-002"
}
```

**Response**:

```json
{
  "jsonrpc": "2.0",
  "id": "list-001",
  "result": {
    "tasks": [
      {
        "id": "task-1",
        "contextId": "session-123",
        "status": { "state": "completed" },
        "artifacts": [ ... ],
        "metadata": {}
      },
      {
        "id": "task-2",
        "contextId": "session-123",
        "status": { "state": "working" },
        "artifacts": [],
        "metadata": {}
      }
    ],
    "nextPageToken": "task-3"
  }
}
```

---

### CancelTask

Cancel a task that is still in the `"working"` state.

**Method**: `CancelTask`

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | String | Task ID |

**Example Request**:

```json
{
  "jsonrpc": "2.0",
  "method": "CancelTask",
  "params": {
    "id": "task-uuid"
  },
  "id": "cancel-001"
}
```

**Response** (success):

```json
{
  "jsonrpc": "2.0",
  "id": "cancel-001",
  "result": {
    "id": "task-uuid",
    "contextId": "session-123",
    "status": {
      "state": "canceled"
    },
    "artifacts": [],
    "metadata": {}
  }
}
```

**Error** (task not found):

```json
{
  "jsonrpc": "2.0",
  "id": "cancel-001",
  "error": {
    "code": -32001,
    "message": "Task 'invalid-id' not found"
  }
}
```

**Error** (task cannot be canceled):

```json
{
  "jsonrpc": "2.0",
  "id": "cancel-001",
  "error": {
    "code": -32003,
    "message": "Task 'task-uuid' is in state 'completed' and cannot be canceled"
  }
}
```

---

## Complete Example

Here's a complete example that defines agents, creates a factory, and serves them via A2A:

```python
import os
from dotenv import load_dotenv
from pydantic import Field

from obelix.core.agent import BaseAgent
from obelix.core.agent.agent_factory import AgentFactory
from obelix.core.tool.tool_base import Tool
from obelix.core.tool.tool_decorator import tool
from obelix.adapters.outbound.anthropic.connection import AnthropicConnection
from obelix.adapters.outbound.anthropic.provider import AnthropicProvider

load_dotenv()


# Define a tool
@tool(name="calculator", description="Performs basic arithmetic operations")
class CalculatorTool(Tool):
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")

    async def execute(self) -> dict:
        ops = {
            "add": lambda: self.a + self.b,
            "subtract": lambda: self.a - self.b,
            "multiply": lambda: self.a * self.b,
            "divide": lambda: self.a / self.b if self.b != 0 else "Error: division by zero",
        }
        return {"result": ops.get(self.operation, "Unknown operation")()}


# Define agents
class MathAgent(BaseAgent):
    def __init__(self, **kwargs):
        connection = AnthropicConnection(api_key=os.getenv("ANTHROPIC_API_KEY"))
        provider = AnthropicProvider(
            connection=connection,
            model_id="claude-sonnet-4-20250514",
        )
        super().__init__(
            system_message="You are a math expert with a calculator tool. Use it to solve equations.",
            provider=provider,
            **kwargs,
        )
        self.register_tool(CalculatorTool())


class ReportAgent(BaseAgent):
    def __init__(self, **kwargs):
        connection = AnthropicConnection(api_key=os.getenv("ANTHROPIC_API_KEY"))
        provider = AnthropicProvider(
            connection=connection,
            model_id="claude-sonnet-4-20250514",
        )
        super().__init__(
            system_message=(
                "You are an agent specialized in creating reports. "
                "You will receive calculation results and format them into a structured report."
            ),
            provider=provider,
            **kwargs,
        )


class CoordinatorAgent(BaseAgent):
    def __init__(self, **kwargs):
        connection = AnthropicConnection(api_key=os.getenv("ANTHROPIC_API_KEY"))
        provider = AnthropicProvider(
            connection=connection,
            model_id="claude-sonnet-4-20250514",
        )
        super().__init__(
            system_message=(
                "You are an orchestrator agent with a Math Agent and a Report Agent. "
                "Use the Math Agent for calculations and the Report Agent to format results."
            ),
            provider=provider,
            **kwargs,
        )


# Create factory and register agents
factory = AgentFactory()

factory.register(
    name="math",
    cls=MathAgent,
    subagent_description="A math expert that can perform calculations",
    stateless=True,
)

factory.register(
    name="report",
    cls=ReportAgent,
    subagent_description="A report writer that formats calculation results",
    stateless=True,
)

factory.register(name="coordinator", cls=CoordinatorAgent)

# Start the A2A server
if __name__ == "__main__":
    factory.serve(
        "coordinator",
        host="0.0.0.0",
        port=8000,
        subagents=["math", "report"],
        description="Coordinator agent with math and reporting sub-agents",
        provider_name="Obelix",
        provider_url="https://github.com/GiulioSurya/Obelix",
    )
```

**Run the server**:

```bash
# Install dependencies
uv sync --extra serve

# Start the A2A server
uv run python example.py
```

**Test the server**:

```bash
# Check agent card
curl http://localhost:8000/.well-known/agent-card.json

# Check health
curl http://localhost:8000/health

# Send a message
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "SendMessage",
    "params": {
      "message": {
        "parts": [{"text": "Calculate 25 * 4 and create a report"}]
      },
      "contextId": "test-session"
    },
    "id": "msg-001"
  }'
```

---

## serve() Parameters

The `AgentFactory.serve()` method accepts these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | String | Required | Name of the registered agent to serve |
| `host` | String | `"0.0.0.0"` | Bind address (use `"127.0.0.1"` for localhost only) |
| `port` | Integer | `8000` | Port to bind to |
| `agent_id` | String (Optional) | agent name | ID in the agent card |
| `version` | String | `"0.1.0"` | Version string in agent card |
| `description` | String (Optional) | system message (truncated) | Agent description in card |
| `provider_name` | String | `"Obelix"` | Organization name in card |
| `provider_url` | String (Optional) | - | Organization URL in card |
| `log_level` | String | `"info"` | Uvicorn log level |
| `subagents` | List[String] (Optional) | - | Sub-agent names to attach |
| `subagent_config` | Dict (Optional) | - | Per-subagent constructor overrides |
| `**create_overrides` | Any | - | Additional kwargs for agent creation |

---

## Error Codes

Ref: [Specification > Errors](https://a2a-protocol.org/latest/specification/)

The A2A server uses standard JSON-RPC error codes plus A2A-specific codes:

| Code | Meaning |
|------|---------|
| `-32700` | Parse error (invalid JSON) |
| `-32600` | Invalid Request (missing method) |
| `-32601` | Method not found (unknown method) |
| `-32602` | Invalid params |
| `-32603` | Internal error (exception during execution) |
| `-32001` | Task not found (A2A-specific) |
| `-32002` | Unsupported operation (spec method not implemented) |
| `-32003` | Task not cancelable (terminal state) |

---

## Best Practices

### 1. Use Stateless Sub-Agents

When exposing a coordinator with sub-agents, register them as `stateless=True` for parallel execution:

```python
factory.register(
    "math",
    MathAgent,
    subagent_description="...",
    stateless=True,  # Safe for parallel calls
)
```

### 2. Bind to Localhost in Development

Use `host="127.0.0.1"` during development:

```python
factory.serve("agent", host="127.0.0.1", port=8000)
```

### 3. Use Context IDs for Session Tracking

Send `contextId` with related messages to group them:

```json
{
  "jsonrpc": "2.0",
  "method": "SendMessage",
  "params": {
    "message": { "parts": [{"text": "..."}] },
    "contextId": "user-session-123"
  },
  "id": "msg-001"
}
```

Then list tasks for that session:

```json
{
  "jsonrpc": "2.0",
  "method": "ListTasks",
  "params": {
    "contextId": "user-session-123"
  },
  "id": "list-001"
}
```

### 4. Handle Long-Running Queries

For long-running agent queries, poll task status instead of blocking:

```python
# Send message (returns immediately with task ID)
task_id = send_message_response["result"]["id"]

# Poll status
while True:
    task = get_task(task_id)
    if task["status"]["state"] in ("completed", "failed", "canceled"):
        break
    time.sleep(1)
```

### 5. Configure Uvicorn Logging

Adjust log level based on environment:

```python
# Development
factory.serve("agent", log_level="debug")

# Production
factory.serve("agent", log_level="warning")
```

---

## Deployment

### Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project
COPY . .

# Sync dependencies with serve extra
RUN uv sync --extra serve

# Expose port
EXPOSE 8000

# Run the A2A server
CMD ["uv", "run", "python", "main.py"]
```

Build and run:

```bash
docker build -t obelix-agent .
docker run -p 8000:8000 obelix-agent
```

### Kubernetes

A basic Kubernetes manifest:

```yaml
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
        image: obelix-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: obelix-secrets
              key: anthropic-api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
```

---

## Integration with A2A Clients

The exposed agent can be discovered and used by any A2A-compliant client. Example using curl:

```bash
# 1. Get agent capabilities
curl http://localhost:8000/.well-known/agent-card.json

# 2. Send message and get task ID
RESPONSE=$(curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "SendMessage",
    "params": {
      "message": {
        "parts": [{"text": "What is 10 + 20?"}]
      }
    },
    "id": "msg-001"
  }')

TASK_ID=$(echo $RESPONSE | jq -r '.result.id')

# 3. Poll for completion
for i in {1..30}; do
  curl -X POST http://localhost:8000/ \
    -H "Content-Type: application/json" \
    -d "{
      \"jsonrpc\": \"2.0\",
      \"method\": \"GetTask\",
      \"params\": {\"id\": \"$TASK_ID\"},
      \"id\": \"get-001\"
    }" | jq '.result.status.state'

  sleep 1
done
```

---

## See Also

- [A2A Compliance](a2a_compliance.md) - Gap analysis and implementation roadmap
- [Agent Factory Guide](agent_factory.md) - Managing agents and sub-agents
- [BaseAgent Guide](base_agent.md) - Agent fundamentals
- [README - Installation](../README.md#installation) - Setting up the project

### A2A Protocol References

- [A2A Specification (RC v1.0)](https://a2a-protocol.org/latest/specification/) - Full protocol definition
- [What is A2A](https://a2a-protocol.org/latest/topics/what-is-a2a/) - Protocol overview
- [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/) - How agents find each other
- [Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/) - Task lifecycle and states
- [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/) - SSE and push notifications
- [A2A and MCP](https://a2a-protocol.org/latest/topics/a2a-and-mcp/) - How the two protocols complement each other
- [Python SDK](https://a2a-protocol.org/latest/sdk/python/api/) - Official Python SDK reference
- [A2A GitHub](https://github.com/a2aproject/A2A) - Protocol source (spec/a2a.proto is the canonical source)
