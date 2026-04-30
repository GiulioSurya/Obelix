from datetime import UTC, datetime, timedelta

import pytest

from obelix.adapters.outbound.a2a.registry import RemoteAgentRegistry


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
    route.registered_at = datetime.now(UTC) - timedelta(seconds=100)

    registry.register_token("tok-B", context_id="ctx-1", agent_name="B")  # fresh

    removed = await registry.gc_expired(ttl_seconds=60)
    assert removed == 1
    assert registry.lookup("tok-A") is None
    assert registry.lookup("tok-B") is not None


def test_token_map_is_dict(registry: RemoteAgentRegistry) -> None:
    """Verify the underlying token_map is a dict.

    Average-case O(1) hash lookup. Note: Python `dict` does not provide
    constant-time string equality on hash collisions, so this is not a
    formal timing-attack defense — but for 256-bit random tokens, hash
    collisions are astronomically improbable.
    """
    assert isinstance(registry._token_map, dict)


# ── resolve_all and card discovery ────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_all_happy_path(httpx_client, patched_resolver):
    from tests.adapters.outbound.a2a.conftest import make_fake_card

    patched_resolver.add("http://b:8001", make_fake_card("B", skills=["lookup"]))
    patched_resolver.add("http://c:8002", make_fake_card("C", skills=["bill"]))
    reg = RemoteAgentRegistry(
        urls=["http://b:8001", "http://c:8002"],
        httpx_client=httpx_client,
    )
    await reg.resolve_all()
    assert set(reg.names()) == {"B", "C"}
    assert reg.card_for("B").name == "B"


@pytest.mark.asyncio
async def test_resolve_all_one_failure_skipped(httpx_client, patched_resolver, caplog):
    import logging

    from tests.adapters.outbound.a2a.conftest import make_fake_card

    caplog.set_level(logging.WARNING)
    patched_resolver.add("http://b:8001", make_fake_card("B"))
    # http://c:8002 NOT added → fetch raises
    reg = RemoteAgentRegistry(
        urls=["http://b:8001", "http://c:8002"],
        httpx_client=httpx_client,
    )
    await reg.resolve_all()
    assert set(reg.names()) == {"B"}


@pytest.mark.asyncio
async def test_resolve_all_duplicate_names_get_discriminator(
    httpx_client, patched_resolver, caplog
):
    import logging

    from tests.adapters.outbound.a2a.conftest import make_fake_card

    caplog.set_level(logging.WARNING)
    patched_resolver.add("http://b1:8001", make_fake_card("B"))
    patched_resolver.add("http://b2:8001", make_fake_card("B"))
    reg = RemoteAgentRegistry(
        urls=["http://b1:8001", "http://b2:8001"],
        httpx_client=httpx_client,
    )
    await reg.resolve_all()
    names = sorted(reg.names())
    assert names == ["B", "B (1)"]


def test_card_for_unknown_raises(registry: RemoteAgentRegistry) -> None:
    with pytest.raises(KeyError):
        registry.card_for("nope")


@pytest.mark.asyncio
async def test_descriptions_returns_name_to_metadata(
    httpx_client, patched_resolver
) -> None:
    """After resolve, descriptions() returns a dict for system_prompt_fragment."""
    from tests.adapters.outbound.a2a.conftest import make_fake_card

    patched_resolver.add(
        "http://b:8001",
        make_fake_card("B", description="inventory", skills=["lookup", "stock"]),
    )
    reg = RemoteAgentRegistry(urls=["http://b:8001"], httpx_client=httpx_client)
    await reg.resolve_all()
    desc = reg.descriptions()
    assert "B" in desc
    assert desc["B"]["description"] == "inventory"
    assert "lookup" in desc["B"]["skills"]
