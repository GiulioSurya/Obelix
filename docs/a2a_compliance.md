# A2A Protocol Compliance

This document maps the Obelix A2A implementation against the
[A2A Protocol Specification v1.0](https://a2a-protocol.org/latest/specification/).
It serves as a complete gap analysis, behavioral reference, and implementation roadmap.

> **Spec reference**: https://a2a-protocol.org/latest/specification/ (v1.0, Linux Foundation).
> **SDK**: [`a2a-sdk`](https://github.com/a2aproject/a2a-python) v0.3.25.
> **Last updated**: 2026-03-13.

---

## Current Architecture

Since 2026-03-13, the Obelix A2A server **delegates protocol handling entirely
to `a2a-sdk`**. Obelix code is limited to a single bridge class.

```
AgentFactory.a2a_serve(endpoint=...)
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
are about **what Obelix behavior we expose**, not schema formatting.

---

## 1. JSON-RPC Methods (Complete Spec v1.0)

Ref: [Specification > Operations](https://a2a-protocol.org/latest/specification/)

### 1.1 Message Methods

| Method | Spec Description | Status | Notes |
|--------|-----------------|--------|-------|
| `message/send` | Send a message, get back Task or Message. Blocking. | **Working** | Routes to `ObelixAgentExecutor.execute()` |
| `message/stream` | Send a message, get SSE stream of events. | **SDK-ready, executor TODO** | SDK has SSE infra via `A2AFastAPIApplication`. Executor must use `execute_query_stream()` and emit `TaskStatusUpdateEvent` + `TaskArtifactUpdateEvent` via `EventQueue`. |

**`message/send` response**: spec allows returning either a `Task` (long-running) or a
`Message` (immediate response). Currently we always create a Task. Returning a direct
`Message` for simple queries (no tool calls, fast response) is a potential optimization.

**`message/stream` SSE event format**: each SSE `data:` line contains a JSON-RPC 2.0
Response wrapping a `StreamResponse` (discriminated union of Task | Message |
TaskStatusUpdateEvent | TaskArtifactUpdateEvent). Stream closes on terminal states.

### 1.2 Task Methods

| Method | Spec Description | Status | Notes |
|--------|-----------------|--------|-------|
| `tasks/get` | Retrieve current task state by ID. Params: `id`, `historyLength?`, `tenant?`. | **Working** | Via `InMemoryTaskStore` |
| `tasks/list` | List tasks with filtering and pagination. | **TODO (NEW in v1.0)** | Params: `contextId?`, `status?`, `pageSize?`, `pageToken?`, `historyLength?`, `statusTimestampAfter?`, `includeArtifacts?`, `tenant?`. Returns `{tasks[], nextPageToken, pageSize, totalSize}`. Need to verify SDK support. |
| `tasks/cancel` | Request task cancellation. | **Partial** | SDK routes to `ObelixAgentExecutor.cancel()`. But cancel is "fake" — emits `canceled` state without actually interrupting `execute_query_async()`. Needs cancellation token. |
| `tasks/resubscribe` | Reconnect SSE stream for an active task. | **SDK-ready, executor TODO** | Requires streaming executor. Client uses this to reconnect after broken SSE connection. |

### 1.3 Push Notification Config Methods

These are **conditional** — required only if `AgentCard.capabilities.push_notifications = true`.

| Method | Spec Description | Status | Notes |
|--------|-----------------|--------|-------|
| `tasks/pushNotificationConfig/set` | Create webhook config for task updates. | **TODO** | SDK has `InMemoryPushNotificationConfigStore` ready. |
| `tasks/pushNotificationConfig/get` | Retrieve webhook configuration. | **TODO** | |
| `tasks/pushNotificationConfig/list` | List webhook configurations for a task. | **TODO** | Supports pagination. |
| `tasks/pushNotificationConfig/delete` | Remove webhook configuration. | **TODO** | |

### 1.4 Agent Card Methods

| Method | Spec Description | Status | Notes |
|--------|-----------------|--------|-------|
| `GET /.well-known/agent-card.json` | Public agent card discovery. | **Working** | SDK default route. |
| `agent/authenticatedExtendedCard` | Authenticated card with restricted info. | **TODO** | Conditional: only if `capabilities.extended_agent_card = true`. Needs auth middleware. |

---

## 2. Task Lifecycle (State Machine)

Ref: [Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/)

### 2.1 All Task States

| State | Category | Description | Obelix Status |
|-------|----------|-------------|---------------|
| `submitted` | Initial | Task acknowledged, not yet processing. | **Never emitted** — we jump straight to `working`. Should emit on task creation before processing starts. |
| `working` | Active | Agent is processing. | **OK** |
| `input-required` | Interrupted | Agent needs more input from the client. Client must send another `message/send` with same `contextId`. | **Never emitted** — see Section 2.2. |
| `auth-required` | Interrupted | Agent needs authentication. Client must authenticate and retry. | **Never emitted** — enterprise scenario, low priority. |
| `completed` | Terminal | Successfully finished. | **OK** |
| `failed` | Terminal | Execution error. | **OK** (but error message could be richer, see Section 2.3) |
| `canceled` | Terminal | Client-initiated cancellation. | **OK** (but cancellation is not real — see Section 1.2) |
| `rejected` | Terminal | Agent declined the task. | **Never emitted** — useful when agent determines it cannot handle the request. |

### 2.2 `input-required` Flow

This is NOT about Obelix's `AskUserQuestionTool` (which is an internal tool).
This is a **protocol-level** concept: the A2A server tells the A2A client
"I need more information before I can continue".

**Spec flow**:
1. Agent determines it needs clarification
2. Server emits `TaskStatusUpdateEvent(state=input-required)` with a Message explaining what's needed
3. Client sends another `message/send` with same `contextId`
4. Server resumes processing (state back to `working`)

**Obelix implementation TODO**: detect when the agent's response is a question
(heuristic or explicit signal) and emit `input-required` instead of `completed`.
Resume when next message arrives on same `contextId`.

### 2.3 Error Reporting in TaskStatus

Current: `TaskStatus(state=failed, message=None)` with `metadata={"error": str(e)}`.

Spec: `TaskStatus.message` should be a `Message` object with details. Better:
```python
TaskStatus(
    state=TaskState.failed,
    message=Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=f"Execution failed: {e}"))],
        message_id=str(uuid.uuid4()),
    ),
)
```

### 2.4 Task Immutability

Spec: once a task reaches a **terminal state** (completed, failed, canceled, rejected),
it CANNOT be restarted. Follow-up work creates a NEW task in the same `contextId`.
The SDK enforces this via `InMemoryTaskStore`.

---

## 3. Async Tasks & Update Delivery

Ref: [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/)

The spec defines three mechanisms for clients to receive updates on long-running tasks.

### 3.1 Polling (Synchronous)

**Status: Working**

Client sends `message/send` → gets back `Task` → polls `tasks/get` periodically.

Simple but inefficient for long tasks. Currently the only supported pattern.

### 3.2 Streaming (SSE)

**Status: SDK-ready, executor TODO**

Client sends `message/stream` → gets SSE stream with real-time events.

**Requirements for Obelix**:
- `AgentCard.capabilities.streaming` must be `true`
- Executor must use `BaseAgent.execute_query_stream()` instead of `execute_query_async()`
- Emit `TaskStatusUpdateEvent` for state changes (submitted → working → completed)
- Emit `TaskArtifactUpdateEvent` for incremental artifact chunks with `append=true` and `last_chunk=true` on final chunk
- Stream closes when task reaches terminal state

**SSE event types** (each is a `StreamResponse` discriminated union):
| Type | When |
|------|------|
| `Task` | Initial task state after creation |
| `TaskStatusUpdateEvent` | State transitions (working, input-required, completed, etc.) |
| `TaskArtifactUpdateEvent` | New/updated artifact chunks |
| `Message` | Direct agent message (alternative to Task) |

**Reconnection**: if SSE connection drops, client calls `tasks/resubscribe` with the task ID.

### 3.3 Push Notifications (Webhooks)

**Status: TODO**

For disconnected/async scenarios. Client provides a webhook URL, server POSTs updates.

**Flow**:
1. Client includes `TaskPushNotificationConfig` in `SendMessageConfiguration`, OR calls `tasks/pushNotificationConfig/set` separately
2. Config contains: `url` (webhook), `token?` (validation), `authentication?` (credentials for server to use when calling webhook)
3. Server POSTs `StreamResponse` payloads to webhook URL on significant state changes
4. Client receives notification, optionally calls `tasks/get` for full state

**Security requirements** (from spec):
- Server SHOULD NOT blindly POST to any client-provided URL (prevent SSRF/DDoS)
- Server should implement: domain allowlisting, ownership verification, egress firewalls
- Client webhook MUST verify authenticity (JWT signatures, HMAC, API keys)
- Replay attack prevention: validate timestamps, use nonces

**SDK support**: `InMemoryPushNotificationConfigStore` for config storage, but
`PushNotificationSender` (the actual httpx POST logic) must be implemented.

### 3.4 `SendMessageConfiguration`

Included in `message/send` and `message/stream` requests. **Currently ignored by executor.**

| Field | Type | Description | Obelix Status |
|-------|------|-------------|---------------|
| `accepted_output_modes` | `string[]` | MIME types the client can handle (e.g. `["text/plain", "application/json"]`) | **Ignored** — always returns `text/plain` |
| `history_length` | `int` | Max messages to include in task history | **Ignored** — SDK may handle this |
| `task_push_notification_config` | `TaskPushNotificationConfig` | Inline webhook config | **Ignored** — push not implemented |
| `return_immediately` | `bool` | **Fire-and-forget mode**: server returns Task in `submitted` state immediately, processes async. Client polls/streams/gets-pushed later. | **Not supported** — always blocks until completion |

**`return_immediately` is the key async pattern**: client sends request, gets back a Task ID
instantly, then uses polling/SSE/push to track progress. This is critical for long-running
agent tasks (minutes/hours).

---

## 4. Data Model Compliance

**Handled by SDK**: all Pydantic models from `a2a.types`.

### 4.1 Message

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `message_id` | Required, unique ID | **Not set by executor** — SDK may generate |
| `role` | `user` or `agent` | OK (SDK sets user, executor should set agent) |
| `parts` | `Part[]` — content | Only `TextPart` used (see 4.2) |
| `context_id` | Conversation grouping | Managed by SDK |
| `task_id` | Associated task | Managed by SDK |
| `reference_task_ids` | References to related tasks | **Not used** — needed for follow-up tasks |
| `metadata` | Custom key-value | Not used |
| `extensions` | Extension URIs | Not used |

### 4.2 Part Types

| Type | Spec | Obelix Status |
|------|------|---------------|
| `TextPart` | `text` string | **OK** — only type used |
| `FilePart` (raw) | `raw` bytes + `media_type` + `filename` | **Not used** — for binary file content |
| `FilePart` (url) | `url` string + `media_type` + `filename` | **Not used** — for file references |
| `DataPart` | `data` JSON value + `media_type` | **Not used** — should use for structured tool results |

### 4.3 Artifact

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `artifact_id` | Required, unique within task | OK — `uuid.uuid4()` |
| `name` | Optional display name | **Not set** |
| `description` | Optional description | **Not set** |
| `parts` | `Part[]` (min 1) | OK — single `TextPart` |
| `metadata` | Custom key-value | Not used |
| `extensions` | Extension URIs | Not used |

**Streaming artifacts**: `TaskArtifactUpdateEvent` has `append` (bool) and `last_chunk` (bool)
fields for incremental delivery. Not used — we send a single complete artifact.

### 4.4 Task

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `id` | Unique task ID | OK — SDK manages |
| `context_id` | Conversation context | OK — SDK manages |
| `status` | `TaskStatus` | Partial — see Section 2 |
| `artifacts` | `Artifact[]` | OK — single artifact per task |
| `history` | `Message[]` — conversation | **Incomplete** — user message stored by SDK, agent response NOT added as Message |
| `metadata` | Custom key-value | Not used |

### 4.5 Context Management (`contextId`)

Spec behavior:
- **First interaction**: server generates new `contextId`
- **Subsequent messages**: client includes same `contextId` to continue conversation
- **Multiple tasks**: same `contextId` can contain parallel or sequential tasks
- **State management**: server uses `contextId` to maintain conversational state

**Obelix status (DONE)**: `ObelixAgentExecutor` stores conversation history per
`contextId` in `_ContextEntry`. Each request creates a fresh agent via the factory,
injects the context's history, executes, and persists the updated history. Multi-turn
conversations work correctly. LRU eviction (default 1024 contexts) prevents unbounded
memory growth. Per-context `asyncio.Lock` serializes concurrent requests on the same
context while allowing full parallelism across different contexts.

---

## 5. Agent Card Compliance (v1.0)

Ref: [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/)

**Built using `a2a.types.AgentCard`** (Pydantic model from SDK).

### 5.1 Fields

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `name` | Required | OK |
| `description` | Required | OK — from `a2a_serve()` param or system message fallback |
| `version` | Required | OK — from `a2a_serve()` param |
| `url` | Agent endpoint URL | OK — from `endpoint` param or derived from `host:port` |
| `provider` | `AgentProvider(organization, url)` | OK |
| `capabilities` | `AgentCapabilities` | OK — `streaming=False`, `push_notifications=False` |
| `skills` | `AgentSkill[]` | OK — derived from tools + subagents |
| `default_input_modes` | `string[]` MIME types | OK — `["text/plain"]` |
| `default_output_modes` | `string[]` MIME types | OK — `["text/plain"]` |
| `supported_interfaces` | `AgentInterface[]` | **Not set** — v1.0 replaces flat `url` with structured interfaces: `{url, protocol_binding, protocol_version}` |
| `documentation_url` | Optional | **Not set** |
| `icon_url` | Optional | **Not set** |
| `security_schemes` | `map<string, SecurityScheme>` | **Not set** — no auth |
| `security_requirements` | `SecurityRequirement[]` | **Not set** — no auth |
| `signatures` | `AgentCardSignature[]` (JWS) | **Not set** — no card signing |

### 5.2 AgentSkill Detail

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `id` | Required | OK |
| `name` | Required | OK |
| `description` | Required | OK |
| `tags` | Required, `string[]` | OK — `["tool"]` or `["subagent"]` |
| `examples` | Optional, `string[]` | **Not set** — example prompts per skill |
| `input_modes` | Optional, per-skill MIME types | **Not set** |
| `output_modes` | Optional, per-skill MIME types | **Not set** |

### 5.3 AgentCapabilities Detail

| Field | Spec | Obelix Status |
|-------|------|---------------|
| `streaming` | Supports SSE | `False` — TODO |
| `push_notifications` | Supports webhooks | `False` — TODO |
| `extended_agent_card` | Supports authenticated card | `False` — TODO |
| `extensions` | `AgentExtension[]` | **Not set** |

### 5.4 Discovery Mechanisms

| Mechanism | Spec | Obelix Status |
|-----------|------|---------------|
| **Well-Known URI** | `GET /.well-known/agent-card.json` | **Working** — SDK serves it |
| **Curated Registry** | Central catalog, query by skill/tag | **Not implemented** — spec doesn't define registry API |
| **Direct Configuration** | Hardcoded URL, env vars, config files | **Supported** — via `endpoint` param |

**Caching**: spec recommends `Cache-Control` + `ETag` headers on agent card responses.
SDK may handle this; needs verification.

### 5.5 Well-Known URI

| Aspect | Status | Notes |
|--------|--------|-------|
| Path | `/.well-known/agent-card.json` | SDK default |
| Content-Type | `application/json` | SDK handles |
| HTTPS | **Not enforced** | Spec requires HTTPS in production |

---

## 6. Transport & Bindings

| Binding | Spec | Status | Notes |
|---------|------|--------|-------|
| **JSON-RPC 2.0** | Primary binding | **Working** | POST to `/` via `A2AFastAPIApplication` |
| **HTTP/REST** | Alternative binding | **SDK-ready** | `a2a.server.apps.rest.fastapi_app` exists in SDK |
| **gRPC** | Alternative binding | **SDK has types** | `a2a.grpc.a2a_pb2` exists; server not wired |

---

## 7. Authentication & Security

Ref: [Enterprise Features](https://a2a-protocol.org/latest/topics/enterprise-ready/)

**No authentication implemented.**

### 7.1 Security Scheme Types (from spec)

| Type | Description | Status |
|------|-------------|--------|
| `APIKeySecurityScheme` | API key in header/query/cookie | TODO |
| `HTTPAuthSecurityScheme` | HTTP Bearer/Basic | TODO |
| `OAuth2SecurityScheme` | OAuth2 flows (authCode, clientCredentials, deviceCode) | TODO |
| `OpenIdConnectSecurityScheme` | OIDC discovery URL | TODO |
| `MutualTlsSecurityScheme` | mTLS | TODO |

### 7.2 Implementation Path

- SDK supports `context_builder` on `A2AFastAPIApplication` for auth middleware
- `AgentCard` model supports `security_schemes` and `security_requirements`
- Extended Agent Card (`agent/authenticatedExtendedCard`) for restricted card info
- `Tenant` field on most request types for multi-tenancy

---

## 8. Error Codes

**All standard JSON-RPC and A2A error codes handled by SDK**:

| Code | Name | Description |
|------|------|-------------|
| -32700 | `JSONParseError` | Invalid JSON |
| -32600 | `InvalidRequestError` | Invalid JSON-RPC request |
| -32601 | `MethodNotFoundError` | Method not found |
| -32602 | `InvalidParamsError` | Invalid method parameters |
| -32603 | `InternalError` | Internal server error |
| -32001 | `TaskNotFoundError` | Task ID not found |
| -32002 | `TaskNotCancelableError` | Task cannot be canceled |
| -32003 | `UnsupportedOperationError` | Operation not supported |
| -32005 | `ContentTypeNotSupportedError` | Unsupported content type |
| -32006 | `InvalidAgentResponseError` | Agent returned invalid response |

---

## 9. Extensions

Spec v1.0 introduces an **extension mechanism** via `AgentExtension`:

```
AgentExtension:
  uri: string (required)    — unique extension identifier
  description: string       — what it does
  required: boolean         — must client support it?
  params: Struct            — configuration
```

Extensions are declared in `AgentCard.capabilities.extensions` and referenced
in `Message.extensions` and `Artifact.extensions`.

**Obelix status**: not implemented. No custom extensions defined.

---

## 10. A2A Client (Outbound)

The current implementation is **server-only** (inbound adapter). For agent-to-agent
communication, Obelix needs to also be an A2A **client**.

### 10.1 SDK Client Infrastructure

| SDK Module | Description |
|------------|-------------|
| `a2a.client.client` | A2A client: `send_message()`, `send_streaming_message()`, `get_task()`, `cancel_task()`, etc. |
| `a2a.client.card_resolver` | Fetch + validate Agent Card from well-known URI |
| `a2a.client.transports.jsonrpc` | JSON-RPC transport |
| `a2a.client.transports.rest` | REST transport |
| `a2a.client.transports.grpc` | gRPC transport |

### 10.2 Proposed Obelix Integration

```
src/obelix/
  adapters/outbound/
    a2a/
      remote_agent_wrapper.py   # Wraps a2a.client as ToolBase-compatible
```

```python
# Remote agent used exactly like a local sub-agent
agent.register_agent(
    RemoteAgentWrapper(endpoint="https://other-agent.example.com"),
    name="remote_math",
    description="Remote math agent via A2A",
)

# Or: discover from agent card first
card = await CardResolver.resolve("https://other-agent.example.com")
agent.register_agent(
    RemoteAgentWrapper.from_agent_card(card),
    name=card.name,
    description=card.description,
)
```

### 10.3 Client Capabilities Needed

| Capability | Description | Priority |
|------------|-------------|----------|
| `send_message` (blocking) | Basic request/response | High |
| `send_streaming_message` | SSE streaming from remote agent | Medium |
| `get_task` / polling | Track async remote tasks | Medium |
| Agent Card resolution | Auto-discover remote agent capabilities | Medium |
| `contextId` propagation | Forward context across agent boundaries | Medium |
| Push notification receiver | Webhook endpoint for async updates | Low |

---

## 11. Implementation Roadmap

### Phase 1: SDK Migration (DONE - 2026-03-13)

- [x] Replace custom controller/handlers/task_store with `a2a-sdk`
- [x] Implement `ObelixAgentExecutor(AgentExecutor)` bridge
- [x] Build `AgentCard` using SDK Pydantic model
- [x] Rename `serve()` -> `a2a_serve()`
- [x] Update `a2a-sdk` to v0.3.25
- [x] Remove dead code (controller.py, handlers.py, task_store.py)
- [x] Add `endpoint` param to `a2a_serve()` for production URLs

### Phase 2: Server Enrichment (High Priority)

Executor improvements to fully leverage the protocol.

#### Context isolation & thread safety (DONE - 2026-03-13)

- [x] **Agent factory pattern**: `ObelixAgentExecutor` receives a `Callable[[], BaseAgent]` instead of a single instance. Each request creates a fresh agent via the factory.
- [x] **Per-context conversation history**: `_ContextEntry` stores `list[StandardMessage]` per `context_id`. History injected into fresh agent before execution, saved back after.
- [x] **Per-context asyncio.Lock**: concurrent requests on the *same* context are serialized. Different contexts run in full parallel.
- [x] **LRU eviction**: `OrderedDict` with `max_contexts=1024` evicts oldest contexts when at capacity.
- [x] **CancelledError handling**: executor catches `asyncio.CancelledError`, emits `canceled` state, re-raises for proper SDK cleanup.

#### Remaining items (TODO)

- [ ] **`submitted` state**: emit `submitted` before `working` on task creation. Enables `return_immediately` (async tasks) for free — the SDK's `ResultAggregator` returns after first event when `blocking=False`.
- [ ] **Streaming SSE**: emit `TaskArtifactUpdateEvent` chunks with `append`/`last_chunk` during execution. Set `capabilities.streaming=true` in Agent Card. Note: A2A streaming is about incremental *events*, not token-by-token LLM streaming. The SDK `DefaultRequestHandler.on_message_send_stream()` wraps our EventQueue in SSE automatically.
- [ ] **`input-required` flow**: detect when agent needs clarification, emit `input-required` state with explanatory Message, resume on next `message/send` with same `contextId`. Note: this is a protocol-level concept, NOT related to Obelix's `AskUserQuestionTool`.
- [ ] **`rejected` state**: emit when agent determines it cannot handle the request
- [ ] **Structured artifacts**: emit `DataPart` for tool results, not just `TextPart`
- [ ] **Agent response in history**: add agent Message to task history alongside user message
- [ ] **Error messages**: use `Message` object in `TaskStatus.message` instead of bare metadata
- [ ] **`accepted_output_modes`**: respect client content negotiation from `SendMessageConfiguration`
- [ ] **`tasks/list`**: SDK's `TaskStore` ABC has no `list_tasks()`. Need custom store or skip.
- [ ] **`supported_interfaces`**: use `AgentInterface` in card instead of flat `url`

### Phase 3: A2A Client — Outbound (High Priority)

Enable Obelix agents to call external A2A agents.

- [ ] **`RemoteAgentWrapper`**: wrap `a2a.client` as ToolBase-compatible, register via `register_agent()`
- [ ] **Agent Card resolver**: discover remote agents via well-known URI, extract skills
- [ ] **Blocking send**: `send_message()` for simple request/response
- [ ] **Streaming receive**: `send_streaming_message()` for real-time results from remote
- [ ] **Async task tracking**: `get_task()` polling for `return_immediately` remote tasks
- [ ] **`contextId` propagation**: forward context across local/remote boundaries
- [ ] **Demo**: two Obelix agents on different ports talking via A2A

### Phase 4: Push Notifications (Medium Priority)

Async update delivery via webhooks.

- [ ] Wire `InMemoryPushNotificationConfigStore` into `DefaultRequestHandler`
- [ ] Implement `PushNotificationSender` (httpx POST to webhook URL)
- [ ] Respect `authentication` in `TaskPushNotificationConfig` when POSTing
- [ ] Security: domain allowlisting, ownership verification for webhook URLs
- [ ] Set `capabilities.push_notifications=true` in Agent Card
- [ ] Client-side: webhook receiver endpoint in `RemoteAgentWrapper`

### Phase 5: Security & Enterprise (Medium Priority)

- [ ] **Authentication middleware** via `context_builder` on `A2AFastAPIApplication`
- [ ] **API Key scheme**: `APIKeySecurityScheme` in card + validation middleware
- [ ] **Bearer/JWT scheme**: `HTTPAuthSecurityScheme` in card + JWT validation
- [ ] **Extended Agent Card**: `agent/authenticatedExtendedCard` endpoint
- [ ] **Tenant support**: propagate `tenant` field across requests
- [ ] **HTTPS enforcement**: require TLS in production mode
- [ ] **Agent Card caching**: `Cache-Control` + `ETag` headers

### Phase 6: Advanced Features (Low Priority)

- [ ] **OAuth2 flows**: `OAuth2SecurityScheme` (authCode, clientCredentials, deviceCode)
- [ ] **HTTP/REST binding**: wire `a2a.server.apps.rest.fastapi_app`
- [ ] **gRPC binding**: wire `a2a.grpc` server
- [ ] **Persistent TaskStore**: replace `InMemoryTaskStore` with SQLite/PostgreSQL
- [ ] **Extensions**: define and register custom `AgentExtension`s
- [ ] **Agent Card signing**: JWS signatures via `AgentCardSignature`
- [ ] **`reference_task_ids`**: support follow-up task references
- [ ] **Skill examples**: populate `AgentSkill.examples` with sample prompts
- [ ] **Multi-modal I/O**: `FilePart` support for binary content, per-skill input/output modes

---

## 12. Reference Links

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
| JWS (RFC 7515) | https://datatracker.ietf.org/doc/html/rfc7515 | Agent card signatures |

---

## See Also

- [Agent Factory Guide](agent_factory.md) — `AgentFactory` and `a2a_serve()` method
