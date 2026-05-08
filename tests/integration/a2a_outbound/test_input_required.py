"""input_required round-trip: A delegates → remote asks input → A responds."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_input_required_propagates_and_respond_resumes():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn a remote A2A agent that triggers request_user_input on first turn.\n"
        "2. Parent A dispatches; remote enters input_required.\n"
        "3. Webhook delivers <remote_task_update status='input_required'> with\n"
        "   <deferred_tool_calls> JSON.\n"
        "4. Drain into A's next turn; mock LLM emits respond_to_remote(task_id, data).\n"
        "5. Verify the DataPart sent to the remote carries the same token as the\n"
        "   original dispatch (token reuse).\n"
        "6. Remote resumes, completes, sends second webhook.\n"
        "7. A's third turn shows <remote_task_update status='completed'>.\n"
        "8. Token revoked after final terminal state."
    )
