"""Multi-context isolation: parallel users on the same A get separate streams."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_dispatch_in_two_contexts_routes_separately():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn one remote agent on an ephemeral port.\n"
        "2. Build parent A serving on a second port via a2a_serve.\n"
        "3. Open two A2A client connections to A with distinct context_ids\n"
        "   (ctx-MARIO and ctx-LUCIA).\n"
        "4. Both contexts dispatch to the same remote with different queries.\n"
        "5. Wait for both webhooks to arrive.\n"
        "6. Trigger second turn on each context separately.\n"
        "7. Verify ctx-MARIO sees only Mario's notification, ctx-LUCIA only Lucia's.\n"
        "8. Verify token_map has 0 entries after both terminal (cleanup)."
    )
