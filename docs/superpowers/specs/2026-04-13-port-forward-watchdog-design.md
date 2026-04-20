# Port-forward Watchdog for OpenShell Deployer

**Date:** 2026-04-13
**Scope:** `src/obelix/adapters/outbound/openshell/deployer.py`
**Author:** Obelix dev session (2026-04-13)

## Context and Problem

The `OpenShellDeployer` exposes an agent running in an OpenShell sandbox via a local port-forward. The current implementation spawns `openshell forward start -d <port> <sandbox>` as a detached daemon (flag `-d`). The daemon creates an SSH tunnel `client → gateway → sandbox`.

**Observed failure mode:** the SSH tunnel dies silently after a period of inactivity (~70 minutes of idle observed on 2026-04-13, reproducing identical symptom on a sandbox from 2026-04-09). The consequences:

- `openshell forward list` shows the forward with status `dead`.
- The sandbox remains `Ready` in Kubernetes.
- `localhost:<port>` is no longer accepting TCP connections.
- The A2A client loses reachability, no recovery happens, the running `deploy.py` has no way to notice.

**Root cause (diagnosed):** SSH server-side idle timeout on the tunnel. No keepalive is configured, and `openshell forward start` does not expose SSH passthrough options.

**Not the real fix.** The correct production solution is a server-side **Activator** (Knative-style scale-to-zero with on-demand wake-up and always-on ingress). That requires OpenShell to expose per-sandbox HTTP ingress on the gateway, which it currently does not. This spec does NOT deliver the Activator — it ships a client-side watchdog as a tactical band-aid and documents the target architecture via TODO.

## Goals

1. Detect when the port-forward daemon has died.
2. Restart it automatically, with minimal downtime (<15 s in the worst case).
3. Log clearly when this happens, so operators can spot the pattern.
4. Not interfere with graceful shutdown (`destroy()`).
5. Leave a visible TODO explaining that the real fix is a server-side Activator.

## Non-goals

- Fixing the underlying SSH idle timeout. Requires changes to OpenShell CLI or SSH config access we do not have.
- On-demand forward activation (open only when a client connects). That is the Activator's job.
- Supporting scale-to-zero or multi-tenant routing. Same reason.
- Integration tests against a live gateway. Requires CI harness that does not exist.

## Approach

Tiny supervised healthcheck that runs inside the deployer's own async event loop. Two concerns:

1. **Liveness probe:** open a TCP connection to `127.0.0.1:<port>` with a short timeout. If the probe succeeds, the forward is alive; if it fails (refused/timeout), it is dead.
2. **Restart:** on detected death, call the existing `_stop_forward()` (cleans up the CLI-side record) and `_start_forward()` (re-spawns the daemon).

The probe is deliberately lightweight (no subprocess, no CLI parsing). Being a TCP connect, it stays close to what the real client does — the same semantics of "can I reach the agent?".

### Alternatives considered

- **Parsing `openshell forward list` output:** more accurate semantically (reads the CLI's internal state), but requires spawning a subprocess every cycle, parsing colored terminal output, and is brittle to CLI version changes. Rejected.
- **Replacing the daemon with a Popen-foreground supervised process:** removes the need for the CLI-managed daemon, but `openshell forward start` (without `-d`) has unknown behavior when stdin/stdout are redirected to DEVNULL, and it would require restructuring the deployer's process model. Higher risk for similar gain. Rejected.

## Components

All changes in `src/obelix/adapters/outbound/openshell/deployer.py`.

### New state on `OpenShellDeployer`

- `self._watchdog_task: asyncio.Task | None` — initialized to `None` in `__init__`.

### New methods

- **`_is_forward_alive() -> bool`** (async, private)
  - Opens a TCP connection to `("127.0.0.1", self._port)` wrapped in `asyncio.wait_for(..., timeout=_PROBE_TIMEOUT)`.
  - On success: closes the writer cleanly, returns `True`.
  - On `OSError` (refused, unreachable) or `asyncio.TimeoutError`: returns `False`.
  - Never raises.

- **`_watchdog_loop() -> None`** (async, private)
  - Header docstring includes the prominent TODO (see below).
  - Infinite `while True` with `asyncio.sleep(_WATCHDOG_INTERVAL)` at the top.
  - Each iteration: if `await self._is_forward_alive()` is False → log WARNING, call `_stop_forward()` then `_start_forward()`, log INFO on success. On restart failure (exception from `_start_forward`), log ERROR, `asyncio.sleep(_RESTART_BACKOFF)`, continue the loop.
  - Catches `asyncio.CancelledError` at the outer level to return cleanly.

- **`_start_watchdog() -> None`**
  - Creates the task via `asyncio.create_task(self._watchdog_loop())` and stores it in `self._watchdog_task`.

- **`_stop_watchdog() -> None`** (async)
  - If `self._watchdog_task` is set and not done: `cancel()`, then `await` (suppressing `CancelledError`).
  - Sets `self._watchdog_task = None`.

### Module-level constants

- `_WATCHDOG_INTERVAL = 10.0` — seconds between liveness probes.
- `_RESTART_BACKOFF = 5.0` — seconds to wait after a failed restart attempt.
- `_PROBE_TIMEOUT = 1.0` — seconds for the TCP probe.

### Call-site changes

- **`deploy()`**: after the successful `await self._start_forward()` (line ~465), call `self._start_watchdog()`.
- **`destroy()`**: at the top of the method, right after the idempotence guard, `await self._stop_watchdog()`. This must happen **before** `_stop_forward()` so the watchdog does not observe the forward going down during shutdown and try to restart it.

## Control Flow

```
deploy()
  ├── _validate()
  ├── _ensure_providers()
  ├── _build_image()
  ├── _create_sandbox()
  ├── _start_server()
  ├── _start_forward()
  └── _start_watchdog()               ← new

_watchdog_loop() (running in background):
  while True:
    await sleep(10s)
    if not await _is_forward_alive():
       logger.warning("forward dead, restarting")
       try:
         await _stop_forward()
         await _start_forward()
         logger.info("forward restored")
       except Exception:
         logger.error("restart failed")
         await sleep(5s)

destroy()
  ├── (idempotence guard)
  ├── _stop_watchdog()                ← new (must come first)
  ├── _stop_forward()
  ├── sandbox delete
  └── client close
```

## The TODO

Placed as a structured comment block at the top of `_watchdog_loop()`:

```python
async def _watchdog_loop(self) -> None:
    """Supervise the port-forward daemon; restart it if it dies.

    TODO(production): replace this watchdog with a server-side Activator.
        The current design is a tactical workaround. The `openshell forward
        start -d` daemon dies silently on SSH idle timeout (~70 min observed).
        This loop detects the death and restarts the forward from the client
        side, accepting up to `_WATCHDOG_INTERVAL` seconds of downtime per
        incident.

        The correct long-term architecture is a server-side Activator
        (Knative-style pattern):

          1. OpenShell gateway exposes each sandbox via an always-on HTTP
             ingress (similar to the existing `inference.local` route but
             for arbitrary sandbox services).
          2. A2A clients speak to the gateway URL, authenticated via mTLS
             or bearer token.
          3. The gateway routes requests to the sandbox pod, waking it
             from scale-to-zero on first request (cold start acceptable).
          4. No client-side tunnel or daemon is needed. Clients work from
             any network, including behind NAT.

        When the Activator is available, delete this watchdog and the
        `_start_forward`/`_stop_forward` path entirely. See
        docs/superpowers/specs/2026-04-13-port-forward-watchdog-design.md
        for context.
    """
```

## Testing Strategy

All unit tests, no live gateway. Tests live in `tests/adapters/outbound/openshell/test_deployer_watchdog.py` (new file).

### Test infrastructure

- Monkeypatch `_WATCHDOG_INTERVAL` and `_RESTART_BACKOFF` to `0.01` in all tests so the loop cycles fast.
- Monkeypatch `_PROBE_TIMEOUT` to `0.05` to keep any real timeout branches fast.
- Each test completes in <1 s.
- Use `pytest-asyncio` with `asyncio_mode = "auto"` (already in `pyproject.toml` per convention).
- Use `unittest.mock.AsyncMock` and `unittest.mock.patch`.

### Required tests

1. **`test_watchdog_no_restart_when_forward_alive`**
   Mock `asyncio.open_connection` to return a fake `(reader, writer)` with `writer.close()` and `writer.wait_closed()` as `AsyncMock`. Start watchdog on a deployer instance with `_start_forward` / `_stop_forward` patched as `AsyncMock`. Let it run for ~5 iterations (0.05 s). Assert `_start_forward.call_count == 0` and `_stop_forward.call_count == 0`.

2. **`test_watchdog_restarts_on_dead_forward`**
   Mock `asyncio.open_connection` to raise `ConnectionRefusedError` on the first call and return a valid pair on the second. Assert `_stop_forward` called exactly once, then `_start_forward` called exactly once, in that order (use a `Mock` with `method_calls` or sequential `side_effect` tracking). Assert no further calls after the forward is "restored".

3. **`test_watchdog_restarts_on_timeout`**
   Mock `asyncio.open_connection` to raise `asyncio.TimeoutError` once, then succeed. Same assertions as test 2. Ensures both refused and timeout are treated as "dead".

4. **`test_watchdog_backoff_on_restart_failure`**
   Mock `asyncio.open_connection` to always fail (dead forever). Mock `_start_forward` to raise `RuntimeError("sandbox gone")` on the first call, succeed on the second. Assert the loop survives the exception (task is not `done()`), logs an error, and on the next cycle retries successfully. Verify `asyncio.sleep` was called with `_RESTART_BACKOFF` at least once (patched to sleep 0).

5. **`test_watchdog_shutdown_graceful_on_cancel`**
   Start the watchdog. Call `_stop_watchdog()`. Assert `_watchdog_task.done()` is True, `_watchdog_task.cancelled()` is True, and no exception is raised out of `_stop_watchdog`. Run test twice in sequence to confirm idempotence.

6. **`test_destroy_cancels_watchdog_before_stopping_forward`**
   Patch `_stop_watchdog` and `_stop_forward` with `AsyncMock`s sharing a single `parent_mock = Mock()`. Call `destroy()`. Assert `parent_mock.method_calls` has `_stop_watchdog` called before `_stop_forward`. Ensures no race where the watchdog could interfere with shutdown.

7. **`test_is_forward_alive_closes_writer`**
   Fake writer with `AsyncMock` for `wait_closed` and `Mock` for `close`. Call `_is_forward_alive()` directly. Assert `close()` and `wait_closed()` were both called. Prevents socket leak regression.

### Out of scope (explicitly)

- Tests exercising real `asyncio.open_connection` against a real port. These would be integration tests requiring setup; add later if CI gains a test gateway.
- Tests verifying SSH tunnel internals. Not our code.
- Tests measuring actual timing accuracy. Flaky and not useful.

## Risks

- **False positives on slow systems:** the TCP probe might fail spuriously if the system is under load. Mitigation: the probe timeout is 1 s, and a single failure triggers a restart, not a crash. Worst case is an unnecessary restart (extra ~2 s of downtime). Acceptable.
- **Restart storms:** if the sandbox is genuinely unreachable (network partition, gateway down), the watchdog will loop `_stop_forward`/`_start_forward` every ~15 s forever. `_RESTART_BACKOFF` (5 s) slows this but does not stop it. Mitigation: acceptable for dev use; production would rely on the Activator, not this loop.
- **Watchdog interferes with destroy:** addressed by ordering `_stop_watchdog()` before `_stop_forward()` in `destroy()`. Covered by test 6.

## Success Criteria

- Running `deploy.py` overnight (>8 h idle) maintains a working `localhost:<port>` the next morning, with one or more WARNING/INFO log lines documenting the auto-recoveries.
- All 7 unit tests pass.
- `ruff check` and `ruff format` clean.
- `openshell forward list` shows at most 1 `dead` entry transiently, and no `dead` entries in steady state.
