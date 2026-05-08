"""Polling fallback: webhook fails → polling worker discovers state."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_polling_fallback_discovers_completed_when_webhook_drops():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn remote agent that completes after ~2s.\n"
        "2. Configure parent A with remote_agents=[remote_url].\n"
        "3. Block A's webhook port (firewall rule, or mock the webhook handler\n"
        "   to return 503 / drop the request) so push notifications are lost.\n"
        "4. Use a synthetic clock (monkey-patch time.monotonic in polling.py) to\n"
        "   avoid waiting 30s real time before polling kicks in.\n"
        "5. Advance clock; verify polling worker calls client.get_task and\n"
        "   discovers completed state.\n"
        "6. Verify <remote_task_update status='completed'> still arrives — same\n"
        "   shape as a successful webhook delivery would have produced.\n"
        "7. Verify poll_failures stays at 0 (success on first poll)."
    )
