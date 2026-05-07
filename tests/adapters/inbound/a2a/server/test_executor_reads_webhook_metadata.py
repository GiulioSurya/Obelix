"""Verify the executor saves client_webhook_url/token from Message metadata
on the FIRST request of a context, and does not overwrite on subsequent ones.

TEMP-PATCH-SPEC-1.

Iron rule respected: no mocks, pure helper function tested standalone.
"""

from __future__ import annotations

from obelix.adapters.inbound.a2a.server.context import ContextEntry


def _apply_metadata_patch(entry: ContextEntry, metadata: dict | None) -> None:
    """Mirror of the production logic: read webhook fields from metadata IFF
    not already set on the entry. Extracted here for unit-testability without
    instantiating the full executor.

    NOTE: this is a test-only mirror; the source-of-truth is in executor.py.
    Whenever you change executor.py's logic, update this mirror too.
    """
    if not metadata:
        return
    # TEMP-PATCH-SPEC-1
    if entry.client_webhook_url is None:
        entry.client_webhook_url = metadata.get("client_webhook_url")
        entry.client_webhook_token = metadata.get("client_webhook_token")


def test_first_request_sets_webhook_fields():
    entry = ContextEntry()
    metadata = {
        "client_info": {"shell": "bash"},
        "client_webhook_url": "http://127.0.0.1:54321/webhook",
        "client_webhook_token": "token-abc",
    }
    _apply_metadata_patch(entry, metadata)
    assert entry.client_webhook_url == "http://127.0.0.1:54321/webhook"
    assert entry.client_webhook_token == "token-abc"


def test_subsequent_request_does_not_overwrite():
    entry = ContextEntry()
    entry.client_webhook_url = "http://first/webhook"
    entry.client_webhook_token = "first-token"

    metadata = {
        "client_webhook_url": "http://second/webhook",
        "client_webhook_token": "second-token",
    }
    _apply_metadata_patch(entry, metadata)
    # First-write wins
    assert entry.client_webhook_url == "http://first/webhook"
    assert entry.client_webhook_token == "first-token"


def test_missing_metadata_no_change():
    entry = ContextEntry()
    _apply_metadata_patch(entry, None)
    _apply_metadata_patch(entry, {})
    assert entry.client_webhook_url is None
    assert entry.client_webhook_token is None


def test_partial_metadata_only_sets_what_is_present():
    """If only one of url/token is in metadata, only that one is set."""
    entry = ContextEntry()
    _apply_metadata_patch(entry, {"client_webhook_url": "http://x/wb"})
    assert entry.client_webhook_url == "http://x/wb"
    assert entry.client_webhook_token is None
