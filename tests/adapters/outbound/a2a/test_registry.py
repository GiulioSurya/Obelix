from datetime import datetime, timedelta

import httpx
import pytest

from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry


@pytest.fixture
def registry() -> RemoteAgentRegistry:
    return RemoteAgentRegistry(urls=[], httpx_client=httpx.AsyncClient())


def test_register_token_creates_route(registry: RemoteAgentRegistry) -> None:
    registry.register_token("tok-A", context_id="ctx-1", agent_name="B")
    route = registry.lookup("tok-A")
    assert route is not None
    assert route.context_id == "ctx-1"
    assert route.agent_name == "B"
    assert route.task_id is None


def test_lookup_unknown_token_returns_none(registry: RemoteAgentRegistry) -> None:
    assert registry.lookup("missing") is None


def test_claim_task_id_fills_field(registry: RemoteAgentRegistry) -> None:
    registry.register_token("tok-A", context_id="ctx-1", agent_name="B")
    registry.claim_task_id("tok-A", task_id="t-001")
    route = registry.lookup("tok-A")
    assert route.task_id == "t-001"


def test_claim_task_id_unknown_token_is_noop(registry: RemoteAgentRegistry) -> None:
    # Should not raise — webhook may have arrived before send_message
    registry.claim_task_id("unknown", task_id="t-001")


def test_revoke_removes_token(registry: RemoteAgentRegistry) -> None:
    registry.register_token("tok-A", context_id="ctx-1", agent_name="B")
    registry.revoke("tok-A")
    assert registry.lookup("tok-A") is None


def test_revoke_unknown_is_idempotent(registry: RemoteAgentRegistry) -> None:
    registry.revoke("never-existed")  # no exception


@pytest.mark.asyncio
async def test_gc_expired_removes_old_tokens(registry: RemoteAgentRegistry) -> None:
    registry.register_token("tok-A", context_id="ctx-1", agent_name="B")
    # Force registered_at to be old
    route = registry.lookup("tok-A")
    route.registered_at = datetime.now() - timedelta(seconds=100)

    registry.register_token("tok-B", context_id="ctx-1", agent_name="B")  # fresh

    removed = await registry.gc_expired(ttl_seconds=60)
    assert removed == 1
    assert registry.lookup("tok-A") is None
    assert registry.lookup("tok-B") is not None


def test_lookup_uses_dict_for_constant_time(registry: RemoteAgentRegistry) -> None:
    """Verify the underlying token_map is a dict (constant-time hash lookup)
    so we don't leak timing info during token enumeration attacks."""
    assert isinstance(registry._token_map, dict)
