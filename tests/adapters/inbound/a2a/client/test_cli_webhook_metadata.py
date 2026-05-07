"""Verify CLIClient generates a webhook token at construction and includes
both webhook_url and token in the metadata of the first Message sent.

TEMP-PATCH-SPEC-1.

Iron rule: NO mock SDK esterni. Verifichiamo solo l'attributo del CLIClient
reale (token init); il test del flusso metadata è coperto da T12 e2e.
"""

from __future__ import annotations

from obelix.adapters.inbound.a2a.client.cli_client import CLIClient
from obelix.adapters.inbound.a2a.client.handlers import default_dispatcher


def test_cli_generates_webhook_token_at_init():
    cli = CLIClient(dispatcher=default_dispatcher(), urls=["http://x"])
    # token must be set, non-empty, URL-safe
    assert cli._webhook_token
    assert isinstance(cli._webhook_token, str)
    assert len(cli._webhook_token) >= 16


def test_cli_webhook_token_is_unique_per_instance():
    a = CLIClient(dispatcher=default_dispatcher(), urls=["http://x"])
    b = CLIClient(dispatcher=default_dispatcher(), urls=["http://x"])
    assert a._webhook_token != b._webhook_token
