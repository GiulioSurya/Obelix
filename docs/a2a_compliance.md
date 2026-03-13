# A2A Protocol Compliance

This document maps the Obelix A2A implementation against the
[A2A Protocol Specification v1.0](https://a2a-protocol.org/latest/specification/).
It serves as a gap analysis and roadmap for full compliance.

> **Spec reference**: the canonical source of truth is
> https://a2a-protocol.org/latest/specification/ (v1.0, Linux Foundation).
> **SDK**: [`a2a-sdk`](https://github.com/a2aproject/a2a-python) v0.3.25.
> **Last updated**: 2026-03-13.

---

## Current Architecture

Since 2026-03-13, the Obelix A2A server **delegates protocol handling entirely
to `a2a-sdk`**. Obelix code is limited to a single bridge class.

```
AgentFactory.a2a_serve()
  |
  v
a2a-sdk infrastructure (managed by SDK):
  A2AFastAPIApplication   -- FastAPI app, routes, agent card endpoint
  DefaultRequestHandler   -- JSON-RPC dispatch, task lifecycle
  InMemoryTaskStore       -- task persistence
  AgentCard (Pydantic)    -- agent card model
  |
  v
ObelixAgentExecutor (our only code):
  RequestContext.get_user_input() -> BaseAgent.execute_query_async() -> EventQueue events
```

**Files**:
- `src/obelix/adapters/inbound/a2a/executor.py` — `ObelixAgentExecutor(AgentExecutor)`
- `src/obelix/core/agent/agent_factory.py` — `a2a_serve()`, `_build_agent_card()`

**Implication**: data model compliance (Task, Message, Part, Artifact, errors,
JSON-RPC method names, well-known URI path) is handled by the SDK. Our gaps
are now about **what Obelix behavior we expose**, not schema formatting.

---

## Operations (JSON-RPC Methods)

Ref: [Specification > Operations](https://a2a-protocol.org/latest/specification/)

### Handled by SDK

| Method | Status | Notes |
|--------|--------|-------|
| `message/send` | **Working** | Blocking mode. Routes to `ObelixAgentExecutor.execute()` |
| `tasks/get` | **Working** | Via `InMemoryTaskStore` |
| `tasks/cancel` | **Working** | Routes to `ObelixAgentExecutor.cancel()` |
| `message/stream` | **SDK-ready, executor TODO** | SDK has SSE infra; executor needs streaming support |
| `tasks/resubscribe` | **SDK-ready, executor TODO** | SDK handles; needs streaming executor |

### Not yet wired

| Method | Blocker | Notes |
|--------|---------|-------|
| `tasks/pushNotificationConfig/set` | No push config store | SDK has `InMemoryPushNotificationConfigStore` ready |
| `tasks/pushNotificationConfig/get` | No push config store | |
| `tasks/pushNotificationConfig/list` | No push config store | |
| `tasks/pushNotificationConfig/delete` | No push config store | |
| `agent/authenticatedExtendedCard` | No auth | Needs security middleware |

---

## Data Model Compliance

**Handled by SDK**: Task, TaskState (all 9 states including `input-required`,
`auth-required`, `rejected`, `submitted`, `unknown`), TaskStatus, Message,
Part (TextPart, FilePart, DataPart), Artifact — all Pydantic models from
`a2a.types`.

### What Obelix controls (in executor.py)

| Aspect | Current | Gap |
|--------|---------|-----|
| Task states used | `working`, `completed`, `failed`, `canceled` | `input-required` not emitted (see Phase 2) |
| Artifact parts | TextPart only | No FilePart/DataPart for structured results |
| Message history | User message stored by SDK | Agent response not added as Message to history |
| Error metadata | `{"error": str(e)}` in metadata | Could be richer (stack trace, error type) |

---

## Agent Card Compliance

**Built using `a2a.types.AgentCard`** (Pydantic model from SDK).

| Field | Status | Notes |
|-------|--------|-------|
| `name` | OK | Agent name |
| `description` | OK | From `a2a_serve()` param or system message fallback |
| `url` | OK | `http://{host}:{port}` |
| `version` | OK | From `a2a_serve()` param |
| `provider` | OK | `AgentProvider(organization=..., url=...)` |
| `capabilities` | OK | `streaming=False`, `pushNotifications=False` |
| `skills` | OK | Derived from tools + subagents, with `tags` |
| `defaultInputModes` | OK | `["text/plain"]` |
| `defaultOutputModes` | OK | `["text/plain"]` |
| `protocolVersion` | OK | Set by SDK (`"0.3.0"` currently) |
| `preferredTransport` | OK | Set by SDK (`"JSONRPC"`) |
| `securitySchemes` | Not set | No auth implemented |
| `security` | Not set | No auth implemented |
| `additional_interfaces` | Not set | Only JSON-RPC binding |
| `signatures` | Not set | No card signing |

### Well-Known URI

| Aspect | Status | Notes |
|--------|--------|-------|
| Path | `/.well-known/agent-card.json` | SDK default; spec v1.0 also accepts this path |
| Content-Type | `application/json` | SDK handles response serialization |

---

## Transport & Bindings

| Binding | Status | Notes |
|---------|--------|-------|
| **JSON-RPC 2.0** | **Working** | POST to `/` via `A2AFastAPIApplication` |
| **HTTP/REST** | **SDK-ready** | `a2a.server.apps.rest.fastapi_app` exists in SDK |
| **gRPC** | **SDK has types** | `a2a.grpc.a2a_pb2` exists; server not wired |

---

## Update Delivery

| Mechanism | Status | Notes |
|-----------|--------|-------|
| **Polling** | **Working** | Client calls `tasks/get` |
| **Streaming (SSE)** | **SDK-ready** | `A2AFastAPIApplication` handles SSE; executor needs `invoke_stream` bridge |
| **Push Notifications** | **SDK-ready** | Needs `PushNotificationConfigStore` + `PushNotificationSender` |

---

## Authentication & Security

**No authentication implemented.** The SDK supports wiring security via
`context_builder` on `A2AFastAPIApplication`, and the `AgentCard` model
supports `securitySchemes` declarations.

---

## Error Codes

**All standard JSON-RPC and A2A error codes are handled by the SDK**:
`JSONParseError`, `InvalidRequestError`, `MethodNotFoundError`,
`InvalidParamsError`, `InternalError`, `TaskNotFoundError`,
`TaskNotCancelableError`, `UnsupportedOperationError`,
`ContentTypeNotSupportedError`, `InvalidAgentResponseError`, etc.

---

## A2A Client (Outbound)

The current implementation is **server-only** (inbound adapter). The SDK also
provides client infrastructure:

| SDK Module | Description |
|------------|-------------|
| `a2a.client.client` | A2A client with send_message, get_task, etc. |
| `a2a.client.card_resolver` | Agent Card fetcher + validator |
| `a2a.client.transports.jsonrpc` | JSON-RPC transport |
| `a2a.client.transports.rest` | REST transport |
| `a2a.client.transports.grpc` | gRPC transport |

### Proposed Obelix integration

```
src/obelix/
  adapters/outbound/
    a2a/
      remote_agent_wrapper.py   # Wraps a2a.client as SubAgentWrapper-compatible tool
```

```python
# Remote agent used exactly like a local sub-agent
agent.register_agent(
    RemoteAgentWrapper(url="https://other-agent.example.com"),
    name="remote_math",
    description="Remote math agent via A2A",
)
```

---

## Implementation Roadmap

### Phase 1: SDK Migration (DONE - 2026-03-13)

- [x] Replace custom controller/handlers/task_store with `a2a-sdk`
- [x] Implement `ObelixAgentExecutor(AgentExecutor)` bridge
- [x] Build `AgentCard` using SDK Pydantic model
- [x] Rename `serve()` -> `a2a_serve()`
- [x] Update `a2a-sdk` to v0.3.25
- [x] Remove dead code (controller.py, handlers.py, task_store.py)

### Phase 2: Executor enrichment (High Priority)

- [ ] **`input-required` flow**: detect `AskUserQuestionTool` result, emit `TaskStatusUpdateEvent(state=input-required)`, resume on next `message/send` with same `contextId`
- [ ] **Streaming executor**: implement `execute()` variant that uses `BaseAgent.execute_query_stream()` and emits `TaskArtifactUpdateEvent` chunks via `EventQueue`
- [ ] **Structured artifacts**: emit `DataPart` for tool results, not just `TextPart`
- [ ] **Agent response in history**: add agent Message to task history alongside user message

### Phase 3: A2A Client - Outbound (Medium Priority)

- [ ] **RemoteAgentWrapper**: wrap `a2a.client` as a tool compatible with `register_agent()`
- [ ] **Agent Card resolver**: discover remote agents via well-known URI
- [ ] **Context propagation**: forward `contextId` across local/remote agent boundaries

### Phase 4: Push Notifications (Low Priority)

- [ ] Wire `InMemoryPushNotificationConfigStore` into `DefaultRequestHandler`
- [ ] Implement `PushNotificationSender` (httpx POST to webhook URL)

### Phase 5: Security & Enterprise (Low Priority)

- [ ] **Authentication middleware** via `context_builder` on `A2AFastAPIApplication`
- [ ] **SecuritySchemes** in AgentCard (API Key, Bearer)
- [ ] **Extended Agent Card** endpoint
- [ ] **Tenant support**

### Phase 6: Additional Bindings (Low Priority)

- [ ] **HTTP/REST binding**: wire `a2a.server.apps.rest.fastapi_app`
- [ ] **Persistent TaskStore**: replace `InMemoryTaskStore` with DB-backed (SQLite/PostgreSQL)

---

## Reference Links

### A2A Protocol

| Resource | URL |
|----------|-----|
| Homepage | https://a2a-protocol.org/latest/ |
| Specification | https://a2a-protocol.org/latest/specification/ |
| Definitions | https://a2a-protocol.org/latest/definitions/ |
| Key Concepts | https://a2a-protocol.org/latest/topics/key-concepts/ |
| What is A2A | https://a2a-protocol.org/latest/topics/what-is-a2a/ |
| Agent Discovery | https://a2a-protocol.org/latest/topics/agent-discovery/ |
| Streaming & Async | https://a2a-protocol.org/latest/topics/streaming-and-async/ |
| Life of a Task | https://a2a-protocol.org/latest/topics/life-of-a-task/ |
| A2A and MCP | https://a2a-protocol.org/latest/topics/a2a-and-mcp/ |
| Enterprise Features | https://a2a-protocol.org/latest/topics/enterprise-ready/ |
| Python SDK API | https://a2a-protocol.org/latest/sdk/python/api/ |
| Python Tutorial | https://a2a-protocol.org/latest/tutorials/python/1-introduction/ |

### GitHub Repositories

| Repo | URL |
|------|-----|
| A2A Main | https://github.com/a2aproject/A2A |
| Python SDK | https://github.com/a2aproject/a2a-python |
| Samples | https://github.com/a2aproject/a2a-samples |
| Pydantic AI A2A | https://ai.pydantic.dev/a2a/ |

### Related Standards

| Standard | URL | Relation |
|----------|-----|----------|
| JSON-RPC 2.0 | https://www.jsonrpc.org/specification | Transport binding |
| RFC 8615 (Well-Known URIs) | https://datatracker.ietf.org/doc/html/rfc8615 | Agent card discovery |
| MCP | https://modelcontextprotocol.io/ | Complementary protocol (tools, not agents) |

---

## See Also

- [Agent Factory Guide](agent_factory.md) - AgentFactory and `a2a_serve()` method
