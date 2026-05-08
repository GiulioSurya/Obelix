"""Test ObelixAgentExecutor._apply_webhook_metadata_patch — the REAL helper
extracted from executor.py:309-314 (TEMP-PATCH-SPEC-1).

NO MOCKS per iron rule. Uses _MinimalExec (Fake-via-subclass with empty __init__)
to instantiate the executor without requiring agent/factory wiring.
This is the same pattern used by tests/adapters/inbound/a2a/server/test_tracer_trace_reuse.py
"""

from __future__ import annotations

from obelix.adapters.inbound.a2a.server.context import ContextEntry
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor


class _MinimalExec(ObelixAgentExecutor):
    """Bypasses agent/factory wiring; only the helper-under-test is needed."""

    def __init__(self):
        pass


def test_first_request_sets_webhook_fields():
    executor = _MinimalExec()
    entry = ContextEntry()
    metadata = {
        "client_info": {"shell": "bash"},
        "client_webhook_url": "http://127.0.0.1:54321/webhook",
        "client_webhook_token": "token-abc",
    }
    executor._apply_webhook_metadata_patch(entry=entry, metadata=metadata)
    assert entry.client_webhook_url == "http://127.0.0.1:54321/webhook"
    assert entry.client_webhook_token == "token-abc"


def test_subsequent_request_does_not_overwrite():
    executor = _MinimalExec()
    entry = ContextEntry()
    entry.client_webhook_url = "http://first/webhook"
    entry.client_webhook_token = "first-token"
    executor._apply_webhook_metadata_patch(
        entry=entry,
        metadata={
            "client_webhook_url": "http://second/webhook",
            "client_webhook_token": "second-token",
        },
    )
    # First-write wins
    assert entry.client_webhook_url == "http://first/webhook"
    assert entry.client_webhook_token == "first-token"


def test_missing_metadata_no_change():
    executor = _MinimalExec()
    entry = ContextEntry()
    executor._apply_webhook_metadata_patch(entry=entry, metadata=None)
    executor._apply_webhook_metadata_patch(entry=entry, metadata={})
    assert entry.client_webhook_url is None
    assert entry.client_webhook_token is None


def test_partial_metadata_only_sets_what_is_present():
    """If only one of url/token is in metadata, only that one is set."""
    executor = _MinimalExec()
    entry = ContextEntry()
    executor._apply_webhook_metadata_patch(
        entry=entry,
        metadata={"client_webhook_url": "http://x/wb"},
    )
    assert entry.client_webhook_url == "http://x/wb"
    assert entry.client_webhook_token is None
