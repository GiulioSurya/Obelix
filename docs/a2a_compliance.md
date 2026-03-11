# A2A Protocol Compliance

This document maps the Obelix A2A implementation against the
[A2A Protocol Specification (RC v1.0)](https://a2a-protocol.org/latest/specification/).
It serves as a gap analysis and roadmap for full compliance.

> **Spec reference**: the canonical source of truth for the protocol is
> [`spec/a2a.proto`](https://github.com/a2aproject/A2A) in the A2A GitHub repository.
> The current spec version is **Release Candidate v1.0** (latest stable: 0.3.0).
> See also: [What is A2A](https://a2a-protocol.org/latest/topics/what-is-a2a/),
> [Key Concepts](https://a2a-protocol.org/latest/topics/key-concepts/).

---

## Table of Contents

- [Architecture Layers](#architecture-layers)
- [Operations (JSON-RPC Methods)](#operations-json-rpc-methods)
- [Data Model Compliance](#data-model-compliance)
- [Agent Card Compliance](#agent-card-compliance)
- [Transport & Bindings](#transport--bindings)
- [Agent Discovery](#agent-discovery)
- [Update Delivery](#update-delivery)
- [Authentication & Security](#authentication--security)
- [Error Codes](#error-codes)
- [A2A Client (Outbound)](#a2a-client-outbound)
- [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Layers

The A2A spec defines a 3-layer architecture
([Specification > Architecture](https://a2a-protocol.org/latest/specification/)):

| Layer | Description | Obelix Status |
|-------|-------------|---------------|
| **Layer 1**: Canonical Data Model | Proto-defined objects (Task, Message, Part, Artifact, AgentCard) | Partial - simplified schemas |
| **Layer 2**: Abstract Operations | Binding-agnostic operations (SendMessage, GetTask, etc.) | 4 of 11 implemented |
| **Layer 3**: Protocol Bindings | JSON-RPC, gRPC, HTTP/REST | JSON-RPC only |

---

## Operations (JSON-RPC Methods)

Ref: [Specification > Operations](https://a2a-protocol.org/latest/specification/)

### Implemented

| Method | Spec | Obelix | Gaps |
|--------|------|--------|------|
| **SendMessage** | Returns `Task` or `Message` | Returns `Task` only | Missing: `configuration` param (acceptedOutputModes, historyLength, returnImmediately, taskPushNotificationConfig), `tenant` param, `metadata` param. Cannot return `Message` directly. |
| **GetTask** | `id`, `historyLength?`, `tenant?` | `id` only | Missing: `historyLength`, `tenant` |
| **ListTasks** | `contextId?`, `status?`, `pageSize?`, `pageToken?`, `historyLength?`, `statusTimestampAfter?`, `includeArtifacts?`, `tenant?` | `contextId`, `status`, `pageSize`, `pageToken` | Missing: `historyLength`, `statusTimestampAfter`, `includeArtifacts`, `tenant` |
| **CancelTask** | `id`, `metadata?`, `tenant?` | `id` only | Missing: `metadata`, `tenant` |

### Not Implemented

| Method | Priority | Description | Spec Ref |
|--------|----------|-------------|----------|
| **SendStreamingMessage** | High | SSE streaming responses | [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/) |
| **SubscribeToTask** | Medium | SSE subscription to existing task | [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/) |
| **GetExtendedAgentCard** | Low | Authenticated agent card with sensitive info | [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/) |
| **CreatePushNotificationConfig** | Low | Create webhook config for task updates | [Specification](https://a2a-protocol.org/latest/specification/) |
| **GetPushNotificationConfig** | Low | Get a push notification config | [Specification](https://a2a-protocol.org/latest/specification/) |
| **ListPushNotificationConfigs** | Low | List push notification configs | [Specification](https://a2a-protocol.org/latest/specification/) |
| **DeletePushNotificationConfig** | Low | Delete push notification config | [Specification](https://a2a-protocol.org/latest/specification/) |

---

## Data Model Compliance

Ref: [Specification > Data Model](https://a2a-protocol.org/latest/specification/),
[Definitions](https://a2a-protocol.org/latest/definitions/)

### Task

| Field (spec) | Type | Obelix | Notes |
|--------------|------|--------|-------|
| `id` | string | Implemented | UUID |
| `contextId` | string? | Implemented | |
| `status` | TaskStatus | Partial | Missing `timestamp` field in status |
| `messages` | Message[]? | **Missing** | Conversation history not stored on task |
| `artifacts` | Artifact[]? | Implemented | |
| `createdAt` | timestamp | Internal only | Stored in TaskRecord but not exposed in JSON |
| `updatedAt` | timestamp | Internal only | Stored in TaskRecord but not exposed in JSON |
| `metadata` | object? | Implemented | Always `{}` |

### TaskState

| State (spec) | Obelix | Notes |
|--------------|--------|-------|
| `WORKING` | `"working"` | Implemented |
| `COMPLETED` | `"completed"` | Implemented |
| `FAILED` | `"failed"` | Implemented |
| `CANCELED` | `"canceled"` | Implemented |
| `INPUT_REQUIRED` | **Missing** | Needed for multi-turn flows where agent asks user for more info |
| `AUTH_REQUIRED` | **Missing** | Needed when agent requires user authentication |
| `REJECTED` | **Missing** | Needed when agent refuses the task |

> **`INPUT_REQUIRED` is critical** for real multi-turn interactions. The spec describes
> it in [Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/) as the
> mechanism for agents to request additional input mid-execution.

### TaskStatus

| Field (spec) | Type | Obelix | Notes |
|--------------|------|--------|-------|
| `state` | TaskState | Implemented | |
| `message` | string? | Partial | Used only on error, spec allows on any state |
| `timestamp` | timestamp? | **Missing** | |

### Message

| Field (spec) | Type | Obelix | Notes |
|--------------|------|--------|-------|
| `id` | string? | **Missing** | Message identifier for deduplication |
| `role` | "user" \| "agent" | Implemented | |
| `parts` | Part[] | Implemented (text only) | |
| `createdAt` | timestamp? | **Missing** | |
| `referenceTaskIds` | string[]? | **Missing** | Cross-task references |

### Part

The spec defines Part as a union type (exactly one of):

| Part type | Obelix | Notes |
|-----------|--------|-------|
| `text` (string) | Implemented | Plain text content |
| `raw` (bytes) | **Missing** | Binary data |
| `url` (string URI) | **Missing** | File/resource reference |
| `data` (JSON) | **Missing** | Structured data |

Additional Part fields: `mediaType`, `filename`, `metadata` - all **missing**.

### Artifact

| Field (spec) | Type | Obelix | Notes |
|--------------|------|--------|-------|
| `id` | string | `artifactId` | Field name differs from spec (`id` vs `artifactId`) |
| `parts` | Part[] | Implemented (text only) | |
| `mimeType` | string? | **Missing** | |
| `title` | string? | **Missing** | |
| `createdAt` | timestamp? | **Missing** | |
| `metadata` | object? | **Missing** | |

---

## Agent Card Compliance

Ref: [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/),
[Specification > Agent Card](https://a2a-protocol.org/latest/specification/)

### Current Obelix Agent Card vs Spec

| Field (spec) | Obelix | Notes |
|--------------|--------|-------|
| `version` | `version` | OK |
| `metadata.name` | `name` (top-level) | **Wrong nesting** - spec puts name under `metadata` |
| `metadata.description` | `description` (top-level) | **Wrong nesting** |
| `provider` | Implemented | OK, but missing `logo` field |
| `capabilities` | Implemented | OK |
| `skills` | Implemented | Missing `inputSchema`, `outputSchema`, `acceptedInputTypes`, `produces` |
| `interfaces` | **Missing** | Array of `{type, url, version}` - required by spec |
| `extensions` | **Missing** | |
| `securitySchemes` | Empty `{}` | Not enforced |
| `security` | Empty `[]` | Not enforced |
| `signature` | **Missing** | Digital signature for card integrity |

### Fields in Obelix NOT in spec

| Field | Notes |
|-------|-------|
| `id` | Not in spec - agent identity is in `metadata.name` |
| `url` | Replaced by `interfaces[].url` in spec |
| `defaultInputModes` | Not in current spec (may be deprecated) |
| `defaultOutputModes` | Not in current spec (may be deprecated) |

### Well-Known URI

| Aspect | Spec | Obelix | Notes |
|--------|------|--------|-------|
| Path | `/.well-known/a2a/agent-card` | `/.well-known/agent-card.json` | **Different path** - spec uses IANA-registered path |
| Media type | `application/a2a+json` | `application/json` | **Different content type** |

---

## Transport & Bindings

Ref: [Specification > Protocol Bindings](https://a2a-protocol.org/latest/specification/)

The spec defines 3 binding types:

| Binding | Obelix | Notes |
|---------|--------|-------|
| **JSON-RPC 2.0** | Implemented | POST to `/` |
| **HTTP/REST** | **Missing** | REST endpoints like `POST /v1/messages`, `GET /v1/tasks/{id}` |
| **gRPC** | **Missing** | Protobuf over HTTP/2 |

### Service Parameters (Headers)

| Header | Obelix | Notes |
|--------|--------|-------|
| `A2A-Version` | **Missing** | Must be sent by both client and server |
| `A2A-Extensions` | **Missing** | Comma-separated extension URIs |

---

## Agent Discovery

Ref: [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/)

The spec defines 3 discovery strategies:

| Strategy | Obelix | Notes |
|----------|--------|-------|
| **Well-Known URI** | Partial | Endpoint exists but path differs from spec |
| **Curated Registry** | **Missing** | Centralized catalog (spec does NOT define standard API) |
| **Direct Configuration** | **Missing** | Static config, env vars |

> Note: The spec explicitly states that "The current A2A specification does not
> prescribe a standard API for curated registries" - this is left to implementations.

---

## Update Delivery

Ref: [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/)

The spec defines 3 mechanisms for receiving task updates:

| Mechanism | Obelix | Notes |
|-----------|--------|-------|
| **Polling** | Implemented | Client calls GetTask periodically |
| **Streaming (SSE)** | **Missing** | Real-time via persistent SSE connection |
| **Push Notifications** | **Missing** | HTTP POST to client webhook |

### `returnImmediately` behavior

The spec's `SendMessageConfiguration.returnImmediately` controls whether SendMessage
blocks until completion or returns immediately for async polling:

- `false` (default): blocks until terminal state - **this is what Obelix does**
- `true`: returns after task creation, client polls/subscribes - **not supported**

---

## Authentication & Security

Ref: [Specification > Security](https://a2a-protocol.org/latest/specification/),
[Enterprise Features](https://a2a-protocol.org/latest/topics/enterprise-ready/)

| Scheme | Obelix | Notes |
|--------|--------|-------|
| API Key (header/query) | **Missing** | |
| HTTP Basic | **Missing** | |
| HTTP Bearer (OAuth 2.0) | **Missing** | |
| OAuth 2.0 Flows | **Missing** | Authorization Code, Client Credentials, Device Code |
| OpenID Connect | **Missing** | |
| Mutual TLS | **Missing** | |

Currently the server has **no authentication at all**. The `securitySchemes` in the
agent card is empty and not enforced.

---

## Error Codes

Ref: [Specification > Errors](https://a2a-protocol.org/latest/specification/)

### Implemented

| Error | Code | Status |
|-------|------|--------|
| Parse error | -32700 | Implemented |
| Invalid request | -32600 | Implemented |
| Method not found | -32601 | Implemented |
| Invalid params | -32602 | Implemented |
| Internal error | -32603 | Implemented |
| TaskNotFoundError | -32001 | Implemented |
| UnsupportedOperationError | -32002 | Implemented |
| TaskNotCancelableError | -32003 | Implemented |

### Missing

| Error (spec) | Notes |
|--------------|-------|
| ContentTypeNotSupportedError | Reject unsupported MIME types in Parts |
| InvalidAgentResponseError | When agent produces malformed response |
| ExtendedAgentCardNotConfiguredError | For GetExtendedAgentCard |
| ExtensionSupportRequiredError | When required extension is missing |
| VersionNotSupportedError | Protocol version mismatch |
| AuthenticationError | HTTP 401 equivalent |
| AuthorizationError | HTTP 403 equivalent |

---

## A2A Client (Outbound)

The current implementation is **server-only** (inbound adapter). The spec defines both
roles:

| Role | Description | Obelix Status |
|------|-------------|---------------|
| **A2A Server** | Receives requests, executes tasks | Implemented (inbound adapter) |
| **A2A Client** | Discovers agents, sends requests | **Not implemented** |

To act as a client, an Obelix agent needs:

1. **Agent Card Resolver**: fetch and cache Agent Cards from well-known URIs or registries
2. **A2A Client**: HTTP client that sends JSON-RPC requests (SendMessage, GetTask, etc.)
3. **Remote Agent Wrapper**: wrap a remote A2A agent behind the same interface as
   `SubAgentWrapper`, so it can be used in `register_agent()` transparently

### Proposed architecture (hexagonal)

```
src/obelix/
  ports/outbound/
    a2a_client.py                  # ABC: discover(), send_message(), get_task()
  adapters/outbound/
    a2a/
      client.py                    # httpx-based A2A client
      agent_card_resolver.py       # Fetch + validate + cache Agent Cards
      remote_agent_wrapper.py      # Wraps remote A2A agent as local sub-agent
```

This would allow:

```python
# Remote agent used exactly like a local sub-agent
agent.register_agent(
    RemoteAgentWrapper(url="https://other-agent.example.com"),
    name="remote_math",
    description="Remote math agent via A2A",
)
```

Ref: [A2A and MCP](https://a2a-protocol.org/latest/topics/a2a-and-mcp/) explains how
A2A (agent-to-agent) complements MCP (agent-to-tool). An Obelix agent could use MCP
for tools and A2A for collaborating with other agents.

---

## Implementation Roadmap

Priorities based on interoperability impact.

### Phase 1: Schema Alignment (High Priority)

Align data models with spec to ensure interoperability with other A2A implementations.

- [ ] **Task schema**: add `messages`, expose `createdAt`/`updatedAt`, fix `status.timestamp`
- [ ] **TaskState**: add `INPUT_REQUIRED`, `AUTH_REQUIRED`, `REJECTED`
- [ ] **Message schema**: add `id`, `createdAt`, `referenceTaskIds`
- [ ] **Artifact schema**: rename `artifactId` to `id`, add `mimeType`, `title`, `metadata`
- [ ] **Agent Card**: restructure to match spec (metadata nesting, interfaces, remove deprecated fields)
- [ ] **Well-Known URI**: change path to `/.well-known/a2a/agent-card`
- [ ] **A2A-Version header**: add to all responses

### Phase 2: Core Features (High Priority)

- [ ] **`input-required` flow**: integrate with BaseAgent to support mid-execution input requests
- [ ] **Message history on Task**: store conversation messages alongside artifacts
- [ ] **SendMessageConfiguration**: support `acceptedOutputModes`, `historyLength`, `returnImmediately`
- [ ] **Additional Part types**: `url`, `data` (structured JSON)

### Phase 3: Streaming (Medium Priority)

- [ ] **SendStreamingMessage**: SSE transport for real-time responses
- [ ] **SubscribeToTask**: SSE subscription to existing task events
- [ ] **TaskStatusUpdateEvent** / **TaskArtifactUpdateEvent**: streaming event types

Ref: [Streaming & Async](https://a2a-protocol.org/latest/topics/streaming-and-async/)

### Phase 4: A2A Client - Outbound (Medium Priority)

- [ ] **AbstractA2AClient** port (ABC)
- [ ] **A2A Client adapter** (httpx-based)
- [ ] **Agent Card Resolver** (fetch, validate, cache)
- [ ] **RemoteAgentWrapper** (remote agent as local sub-agent)
- [ ] **Discovery strategies**: well-known URI, static config

### Phase 5: Security & Enterprise (Low Priority)

- [ ] **Authentication middleware**: API Key, Bearer token
- [ ] **SecuritySchemes enforcement**: validate against agent card declarations
- [ ] **Extended Agent Card**: authenticated endpoint with sensitive info
- [ ] **Tenant support**: multi-tenant `tenant` parameter on all methods
- [ ] **Missing error codes**: ContentTypeNotSupported, VersionNotSupported, auth errors

Ref: [Enterprise Features](https://a2a-protocol.org/latest/topics/enterprise-ready/)

### Phase 6: Additional Bindings (Low Priority)

- [ ] **HTTP/REST binding**: RESTful endpoints as alternative to JSON-RPC
- [ ] **Push Notifications**: webhook-based task update delivery
- [ ] **Persistent TaskStore**: replace in-memory store with DB-backed storage

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
| Extensions | https://a2a-protocol.org/latest/topics/extensions/ |
| Python SDK API | https://a2a-protocol.org/latest/sdk/python/api/ |
| Python Tutorial | https://a2a-protocol.org/latest/tutorials/python/1-introduction/ |
| Roadmap | https://a2a-protocol.org/latest/roadmap/ |

### GitHub Repositories

| Repo | URL |
|------|-----|
| A2A Main | https://github.com/a2aproject/A2A |
| Python SDK | https://github.com/a2aproject/a2a-python |
| JavaScript SDK | https://github.com/a2aproject/a2a-js |
| Samples | https://github.com/a2aproject/a2a-samples |

### Related Standards

| Standard | URL | Relation |
|----------|-----|----------|
| JSON-RPC 2.0 | https://www.jsonrpc.org/specification | Transport binding |
| RFC 8615 (Well-Known URIs) | https://datatracker.ietf.org/doc/html/rfc8615 | Agent card discovery path |
| MCP | https://modelcontextprotocol.io/ | Complementary protocol (tools, not agents) |

---

## See Also

- [A2A Server Guide](a2a_server.md) - Current implementation documentation
- [Agent Factory Guide](agent_factory.md) - AgentFactory and `serve()` method