"""End-to-end happy path: A dispatches to remote, receives completion
notification at next turn."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_dispatch_completes_and_notification_drained_at_next_turn():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn a real remote A2A agent via a2a_serve on an ephemeral port.\n"
        "2. Build parent A with remote_agents=[remote_url].\n"
        "3. Mock A's LLM provider to emit dispatch_agent('remote', 'hi').\n"
        "4. Trigger A's first turn — verify dispatch_agent returns synchronously\n"
        "   with task_id + status='submitted'.\n"
        "5. Wait up to 5s for the webhook to arrive (poll entry.pending_notifications).\n"
        "6. Trigger A's second turn — assert <remote_task_update status='completed'>\n"
        "   appears in agent.conversation_history before the LLM call.\n"
        "7. Verify the result_text in the notification matches the remote's reply."
    )
