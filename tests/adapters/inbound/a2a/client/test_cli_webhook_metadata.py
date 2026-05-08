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


def test_cli_registers_push_config_with_token():
    """Bug 1 regression: the PushNotificationConfig the CLI registers with the
    server MUST carry the same token that the local WebhookServer expects.

    Without this, the server's BasePushNotificationSender posts to the local
    webhook with no X-A2A-Notification-Token header, the WebhookServer (T11)
    rejects with 401, and the CLI loses every push notification — having to
    rely on the polling fallback only.

    The fix is to set ``PushNotificationConfig(url=..., token=self._webhook_token)``
    in cli_client.py. We assert the wiring by inspecting the source.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[5] / (
        "src/obelix/adapters/inbound/a2a/client/cli_client.py"
    )
    text = src.read_text(encoding="utf-8")
    pattern = re.compile(
        r"PushNotificationConfig\s*\("
        r"[^)]*\burl\s*=\s*self\._webhook_url"
        r"[^)]*\btoken\s*=\s*self\._webhook_token",
        re.DOTALL,
    )
    assert pattern.search(text), (
        "cli_client.py must construct PushNotificationConfig "
        "with both url=self._webhook_url AND token=self._webhook_token. "
        "Without the token, the server's push notifications get rejected "
        "by the local WebhookServer (401)."
    )
