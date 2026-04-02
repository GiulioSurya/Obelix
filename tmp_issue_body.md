## Problem Statement

A2A Protocol v1.0 has been officially released (Linux Foundation, March 2026 — [announcement](https://a2a-protocol.org/latest/announcing-1.0/)). The Python SDK (`a2a-sdk`) has a corresponding `v1.0.0-alpha.0` pre-release. Obelix currently pins `a2a-sdk>=0.3.25` and will be unable to communicate with v1.0-only clients or adopt new spec capabilities (signed Agent Cards, multi-tenancy, version negotiation) without migration.

## What Changed in A2A v1.0

### New Functional Capabilities

| Feature | Description | Spec Reference |
|---------|-------------|----------------|
| **Signed Agent Cards** | Agent cards can carry a cryptographic signature (`AgentCardSignature`). Clients can verify server identity and card integrity before trusting it. | [Spec §8.4](https://a2a-protocol.org/latest/specification/) |
| **Multi-tenancy** | A single A2A endpoint can host multiple logical agents for different tenants. Every operation accepts an optional `tenant` path parameter; servers must scope resources accordingly. | [Spec §13.1](https://a2a-protocol.org/latest/specification/) |
| **Version negotiation** | Client and server exchange `A2A-Version` headers. Mismatch returns an explicit `VersionNotSupportedError` instead of silent breakage. | [Spec §3.6](https://a2a-protocol.org/latest/specification/) |
| **Multiple concurrent streams** | Multiple clients can subscribe to the same task simultaneously. Events are broadcast to all active SSE streams. Previously only one stream per task was implied. | [Spec §3.1.2](https://a2a-protocol.org/latest/specification/) |
| **`GetExtendedAgentCard`** | Authenticated endpoint returning a richer agent card with restricted skills/capabilities not included in the public `/.well-known/agent-card.json`. | [Spec §3.1.11, §8.1](https://a2a-protocol.org/latest/specification/) |
| **`returnImmediately`** | Explicit field in `SendMessageConfiguration` — client can request non-blocking task creation (server responds immediately with the task ID). | [Spec §3.2.2](https://a2a-protocol.org/latest/specification/) |
| **Protocol bindings** | In addition to JSON+HTTP, gRPC and JSON-RPC are now first-class bindings. | [Spec §3](https://a2a-protocol.org/latest/specification/) |

### What Does NOT Change Functionally

The following work identically in v1.0: `message/send` → task lifecycle → `completed`, SSE streaming, deferred tools / `input-required`, task cancellation, push notifications (semantics), agent card discovery via `/.well-known/agent-card.json`.

### SDK Breaking Change: Pydantic → Protobuf Types

The `a2a-sdk v1.0.0-alpha.0` replaced all Pydantic models in `a2a.types` with protobuf-generated types. This does **not** change the protocol semantics, but it breaks every file that constructs or inspects `Part`, `TextPart`, `DataPart`, `FilePart`, `PushNotificationConfig`, `TaskIdParams`, `TaskQueryParams`. The SDK provides a `a2a.compat.v0_3.types` shim with the old Pydantic models for a gradual migration.

**References:**
- [A2A Spec v1.0](https://a2a-protocol.org/latest/specification/)
- [Announcing A2A v1.0](https://a2a-protocol.org/latest/announcing-1.0/)
- [a2a-sdk v1.0.0-alpha.0 release](https://github.com/a2aproject/a2a-python/releases/tag/v1.0.0-alpha.0)
- [a2a-python GitHub](https://github.com/a2aproject/a2a-python)
- [Life of a Task](https://a2a-protocol.org/latest/topics/life-of-a-task/)
- [Agent Discovery](https://a2a-protocol.org/latest/topics/agent-discovery/)

---

## Technical Context

Obelix delegates all A2A protocol handling to the SDK via `ObelixAgentExecutor(AgentExecutor)`. The `AgentExecutor.execute()`/`cancel()` interface is **unchanged** in v1.0 — our bridge class does not need signature changes. The migration risk is fully contained to the adapter layer (`src/obelix/adapters/inbound/a2a/`) and `agent_factory.py`; the core domain layer has zero SDK imports.

The core abstraction break is in `Part` construction and inspection. Currently: `Part(root=TextPart(text="..."))` — Pydantic discriminated union. In v1.0: `Part` is a protobuf message with `oneof` fields — construction and field access semantics change entirely.

---

## Affected Components

| Component | Key Files | Role |
|-----------|-----------|------|
| Part converter | `src/obelix/adapters/inbound/a2a/part_converter.py` | Bidirectional A2A Part ↔ Obelix ContentPart — widest spread of broken imports |
| Executor | `src/obelix/adapters/inbound/a2a/server/executor.py` | ObelixAgentExecutor — TextPart/DataPart construction for artifacts |
| Deferred tools | `src/obelix/adapters/inbound/a2a/server/deferred.py` | DataPart isinstance check for deferred result parsing |
| Helpers | `src/obelix/adapters/inbound/a2a/server/helpers.py` | TextPart construction for agent messages (19 lines total) |
| Push config store | `src/obelix/adapters/inbound/a2a/server/push_config_store.py` | SmartPushNotificationConfigStore — PushNotificationConfig + set_info() API incompatible |
| Push sender | `src/obelix/adapters/inbound/a2a/server/push_sender.py` | SmartPushNotificationSender — _dispatch_notification() signature incompatible at every parameter |
| Agent factory | `src/obelix/core/agent/agent_factory.py` | a2a_serve() / _build_agent_card() — AgentCapabilities field name needs verification |
| CLI client | `src/obelix/adapters/inbound/a2a/client/cli_client.py` | Largest file — 8 removed type imports + A2AClientHTTPError/A2AClientJSONError removed |
| Test suite | `tests/adapters/inbound/a2a/` | 4 test files — FakeRequestContext constructs v0.3 Part objects |
| Dependencies | `pyproject.toml:20` | a2a-sdk version pin |

---

## Technical Investigation

### Code References

| Location | Description |
|----------|-------------|
| `pyproject.toml:20` | `"a2a-sdk>=0.3.25"` — pin to update |
| `part_converter.py:11-18` | Imports `DataPart, FilePart, FileWithBytes, FileWithUri, TextPart` — all removed from v1.0 `a2a.types` |
| `part_converter.py:40,45,66` | `isinstance(root, TextPart/FilePart/DataPart)` — dispatch pattern breaks with proto types |
| `part_converter.py:86,91,100,105,112,116` | `Part(root=TextPart(...))` / `Part(root=DataPart(...))` — proto construction is different |
| `server/executor.py:37` | `from a2a.types import DataPart, TextPart` — removed |
| `server/executor.py:120,296,317,319,327,350,364,502,508` | TextPart/DataPart construction sites (9 locations) |
| `server/helpers.py:7` | `from a2a.types import TextPart` — removed |
| `server/deferred.py:7` | `from a2a.types import DataPart` — removed |
| `server/deferred.py:29` | `isinstance(part.root, DataPart)` — proto field check needed |
| `server/push_config_store.py:24` | `from a2a.types import PushNotificationConfig` — replaced by `TaskPushNotificationConfig` |
| `server/push_config_store.py:43` | `set_info(self, task_id, notification_config: PushNotificationConfig)` — v1.0 adds required `context: ServerCallContext` param |
| `server/push_sender.py:14` | `from a2a.types import PushNotificationConfig` — removed |
| `server/push_sender.py:36` | `super().__init__(httpx_client=..., config_store=...)` — v1.0 requires third `context` param |
| `server/push_sender.py:44` | `_dispatch_notification(self, task: Task, push_info: PushNotificationConfig)` — v1.0: `(self, event: PushNotificationEvent, push_info: TaskPushNotificationConfig, task_id: str)` |
| `server/push_sender.py:52` | `task.model_dump(mode="json")` — proto `Task` has no `.model_dump()` |
| `client/cli_client.py:23-42` | `A2AClientHTTPError, A2AClientJSONError` removed from `a2a.client`; 7 type imports removed from `a2a.types` |
| `client/cli_client.py:968,970,1013,1018,1099` | `except A2AClientHTTPError/A2AClientJSONError` — will silently fail to match, degrading UI error messages |
| `agent_factory.py:674` | `supports_authenticated_extended_card=False` — field name may have changed in v1.0 `AgentCapabilities` |
| `tests/adapters/inbound/a2a/test_server.py` | FakeRequestContext: `Part(root=TextPart(text=...))` — breaks under proto types |
| `tests/adapters/inbound/a2a/test_return_immediately.py` | Accesses `handler._running_agents_lock`, `handler._background_tasks` — private SDK internals, may be removed |
| `tests/adapters/inbound/a2a/test_webhook_resilience.py` | `PushNotificationConfig`, `Task.model_dump()` — Pydantic constructors and methods break |

### What Would Need to Change (dependency order)

1. **`pyproject.toml`**: Update pin to `a2a-sdk>=1.0.0a0` (or stable when available). Note: uv will not pick up pre-releases without explicit pin or `--pre`.
2. **`part_converter.py`** — fix first, it is the dependency of executor and deferred. Rewrite Part construction and isinstance dispatch.
3. **`helpers.py`** — trivial once TextPart source is resolved.
4. **`deferred.py`** — replace DataPart import; update isinstance check to proto field access.
5. **`push_config_store.py`** — rewrite `set_info()` to accept `TaskPushNotificationConfig` + `context: ServerCallContext`.
6. **`push_sender.py`** — rewrite `__init__` and `_dispatch_notification` to v1.0 signatures; replace `task.model_dump()`.
7. **`executor.py`** — update TextPart/DataPart imports and 9 construction sites.
8. **`agent_factory.py`** — verify `AgentCapabilities` field names against v1.0 proto stubs.
9. **`cli_client.py`** — replace all removed imports; replace `A2AClientHTTPError`/`A2AClientJSONError` with `A2AClientError` base class.
10. **Tests** — update `FakeRequestContext`, Part constructions, PushNotificationConfig usage across 4 test files.

### Alternative Approaches

**Option A — Use v0.3 compat layer (recommended for first pass)**: Import removed types from `a2a.compat.v0_3.types`. Minimal code change, `Part(root=TextPart(...))` pattern likely still works. Risk: compat `__init__.py` is empty (no explicit public API guarantee); behavior on server-side Part construction needs verification.

**Option B — Go native proto**: Rewrite all Part construction/inspection using proto `oneof` field access (e.g., `HasField("text")`). Correct long-term, significantly more invasive. Requires inspecting `a2a_pb2.pyi`.

**Option C — Wait for stable v1.0**: Keep v0.3.25. Alpha published March 17, 2026; no stable timeline visible.

Recommended: **Option A first**, then migrate to native proto after v1.0 stabilizes. Set `enable_v0_3_compat=True` on `A2AFastAPIApplication` during transition if existing v0.3 CLI clients need to keep working.

---

## Proposed Approach

Pin `a2a-sdk>=1.0.0a0` and migrate type imports in dependency order starting from `part_converter.py`, using the v0.3 compat layer for all removed Pydantic types. Migrate the push notification stack (`push_config_store.py` + `push_sender.py`) to the new `TaskPushNotificationConfig` + `ServerCallContext` API. Replace removed `A2AClientHTTPError`/`A2AClientJSONError` with `A2AClientError` base class in the CLI client. New v1.0 features (signed Agent Cards, multi-tenancy, version negotiation, `GetExtendedAgentCard`) are **out of scope** for this migration — they are additive and should be tracked as separate issues.

## Scope Assessment

- **Complexity:** High
- **Confidence:** Medium — clear path for Part types (compat layer), but proto `Part` field access semantics need verification before writing code; push sender `_dispatch_notification` change is confirmed breaking but `PushNotificationEvent` structure needs inspection
- **Estimated files to change:** 10 source files + 4 test files
- **Issue type:** `feat`

## Risks & Open Questions

- **Proto `Part` construction API**: Before writing code, inspect `a2a_pb2.pyi` to confirm `Part` oneof field names. All downstream work depends on this.
- **Compat layer server-side validity**: The v0.3 compat layer is documented for client-side use; does it work correctly for server-side Part construction?
- **`PushNotificationEvent` structure**: What data does it carry vs. `Task`? Can we still send a full task JSON payload to webhooks, or does the payload format need to change?
- **`test_return_immediately.py` accesses private SDK internals** (`_running_agents_lock`, `_background_tasks`): these may be removed. May need a full rewrite using public task store polling.
- **Alpha SDK stability**: `v1.0.0-alpha.0` may introduce further breaking changes. Decision needed: migrate to alpha now or wait for stable?
- **New v1.0 additive features** to track separately: signed Agent Cards, multi-tenancy, `GetExtendedAgentCard`, version negotiation header.

## Test Considerations

- Good existing coverage: `test_server.py`, `test_cancel.py`, `test_return_immediately.py`, `test_webhook_resilience.py` — all 4 need updates.
- Fix `FakeRequestContext` in `test_server.py` and `test_cancel.py` first — it is used pervasively and blocks all other test work.
- `test_webhook_resilience.py` may need a mock `ServerCallContext` — check if the SDK provides a test double.
- Gate: run `uv run pytest tests/adapters/inbound/a2a/` before declaring migration complete.

---
*Created by spike investigation. Use `build-from-issue` to plan and implement.*