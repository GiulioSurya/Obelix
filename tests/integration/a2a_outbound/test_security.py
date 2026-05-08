"""Token spoofing and eviction protection."""

import pytest


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_unknown_token_returns_401():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Spawn parent A with remote_agents=[].\n"
        "2. POST to A's /webhook directly (bypass the registry) with a random\n"
        "   X-A2A-Notification-Token header value.\n"
        "3. Assert HTTP 401 + 'unknown token' in body.\n"
        "4. Assert no state mutation in any context.\n"
        "5. Assert the warning was logged with the (truncated) token length."
    )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_eviction_protection_under_saturation():
    pytest.skip(
        "Integration scaffold. Full implementation requires:\n"
        "1. Configure parent A with max_contexts=2.\n"
        "2. Open 4 contexts each with one non-terminal remote_task in flight\n"
        "   (use a slow remote that doesn't complete during the test).\n"
        "3. Try to open a 5th context.\n"
        "4. Verify the store has exactly 4 entries (2x cap reached, forced\n"
        "   eviction kicks in but logs the warning).\n"
        "5. Verify the oldest non-evictable was force-evicted (its webhook\n"
        "   would now return 401 — log message was emitted).\n"
        "6. Verify all 5 contexts at this point."
    )
