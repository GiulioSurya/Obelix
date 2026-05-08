"""Bug 3 regression: when ``a2a_serve`` is called with the default
``host="0.0.0.0"`` (bind-all), the webhook URL handed to remote agents
must NOT contain ``0.0.0.0`` — that's a bind address, not an endpoint.

Symptom: ``dispatch_agent`` registers a PushNotificationConfig with
``url=http://0.0.0.0:8005/webhook`` and the remote agent posts to it,
getting "All connection attempts failed" because 0.0.0.0 is not a routable
host (it's a wildcard bind on the server side).

Fix: rewrite ``0.0.0.0`` (and IPv6 ``::``) to ``127.0.0.1`` for the webhook
URL specifically. Same convention already used for the display URL at
agent_factory.py:788.

Iron rule: NO mock; test the helper directly.
"""

from __future__ import annotations

from obelix.core.agent.agent_factory import _resolve_webhook_host


def test_resolve_webhook_host_rewrites_ipv4_wildcard():
    assert _resolve_webhook_host("0.0.0.0") == "127.0.0.1"


def test_resolve_webhook_host_rewrites_ipv6_wildcard():
    assert _resolve_webhook_host("::") == "127.0.0.1"


def test_resolve_webhook_host_passes_concrete_host_through():
    assert _resolve_webhook_host("127.0.0.1") == "127.0.0.1"
    assert _resolve_webhook_host("localhost") == "localhost"
    assert _resolve_webhook_host("192.168.1.10") == "192.168.1.10"
    assert _resolve_webhook_host("example.com") == "example.com"
