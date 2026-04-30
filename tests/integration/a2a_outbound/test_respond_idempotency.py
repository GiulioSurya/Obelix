"""respond_to_remote idempotency within an input_required cycle."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_double_respond_blocked_by_idempotency():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn remote that triggers input_required.\n"
        "2. Parent A receives the input_required notification.\n"
        "3. Parent A's LLM emits respond_to_remote(task_id, data) — first response.\n"
        "4. Verify the DataPart was sent on the wire to the remote.\n"
        "5. Same turn, mock LLM emits respond_to_remote(task_id, data2) — second.\n"
        "6. Assert second call returns ToolStatus.ERROR (gate already consumed).\n"
        "7. Assert only ONE DataPart was sent on the wire (no duplicate).\n"
        "8. Then simulate remote re-entering input_required (new cycle) and verify\n"
        "   respond is again allowed."
    )
