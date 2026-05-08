"""Integration test infrastructure for outbound A2A scenarios.

These tests are SCAFFOLDS — they document scenarios for later full
implementation. The full version requires:
- Spawning real A2A servers (`a2a_serve`) on ephemeral ports.
- A scripted ScriptedProvider returning predetermined LLM tool calls.
- A `webhook_blackhole` fixture that drops outgoing pushes for fallback testing.
- Synthetic clock manipulation (monkey-patched `time.monotonic`) to avoid
  long real-time waits in polling-fallback scenarios.

For now, the scenarios are described in detail via pytest.skip messages
so a future implementer knows what each test must verify. The unit tests
under tests/adapters/inbound/a2a/ and tests/adapters/outbound/a2a/ cover
the bulk of the logic; these integration tests would verify the wire-level
contract.
"""
