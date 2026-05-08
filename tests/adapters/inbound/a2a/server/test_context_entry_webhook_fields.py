"""Verify ContextEntry exposes client_webhook_url/token slots — TEMP-PATCH-SPEC-1."""

from __future__ import annotations

from obelix.adapters.inbound.a2a.server.context import ContextEntry


def test_context_entry_has_client_webhook_slots():
    entry = ContextEntry()
    # Slots must exist (raises AttributeError if not in __slots__)
    assert entry.client_webhook_url is None
    assert entry.client_webhook_token is None


def test_context_entry_can_set_webhook_fields():
    entry = ContextEntry()
    entry.client_webhook_url = "http://127.0.0.1:54321/webhook"
    entry.client_webhook_token = "abc123"
    assert entry.client_webhook_url == "http://127.0.0.1:54321/webhook"
    assert entry.client_webhook_token == "abc123"
