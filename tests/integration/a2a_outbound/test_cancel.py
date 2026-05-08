"""Cancel of A's task with in-flight remotes (decision 7 — local-only)."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_cancel_revokes_tokens_no_wire_call():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn 3 remote agents that hang (never complete).\n"
        "2. Parent A dispatches to all 3 in one turn.\n"
        "3. Verify 3 entries in entry.remote_tasks (status='submitted'/'working').\n"
        "4. Trigger cancel on A's executor (via cancel() or CancelledError in loop).\n"
        "5. Verify all 3 tokens are revoked from registry.\n"
        "6. Verify all 3 statuses flipped to 'killed' with last_update +\n"
        "   last_update_monotonic both refreshed.\n"
        "7. Critical: spy on each remote's HTTP server — assert NO cancel_task\n"
        "   request was received (decision 7).\n"
        "8. Mock late webhooks from the (still running) remotes — assert each\n"
        "   returns 401."
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_cancel_then_new_dispatch_filo_conduttore():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Parent A dispatches t-001 to agent_1.\n"
        "2. Cancel A's turn — t-001's token revoked.\n"
        "3. Parent A in same context dispatches t-002 to agent_1.\n"
        "4. agent_1 (still running t-001) eventually fires the t-001 webhook → 401.\n"
        "5. agent_1 finishes t-002 → token valid → notification accodata cleanly.\n"
        "6. Verify t-002 notification arrives in next turn; t-001 ghost stays absent."
    )
