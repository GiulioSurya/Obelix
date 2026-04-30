from datetime import datetime

import pytest

from obelix.adapters.outbound.a2a.state import RemoteTaskState, TokenRoute


def _make_state(status: str) -> RemoteTaskState:
    return RemoteTaskState(
        task_id="t-001",
        agent_name="B",
        status=status,
        token="tok-x",
        created_at=datetime.now(),
        last_update=datetime.now(),
        last_update_monotonic=0.0,
        last_artifact=None,
        deferred_calls=None,
    )


@pytest.mark.parametrize(
    "status,expected",
    [
        ("submitted", False),
        ("working", False),
        ("input_required", False),
        ("completed", True),
        ("failed", True),
        ("canceled", True),
        ("rejected", True),
        ("killed", True),
    ],
)
def test_remote_task_state_is_terminal(status: str, expected: bool) -> None:
    assert _make_state(status).is_terminal is expected


def test_remote_task_state_default_poll_failures_is_zero() -> None:
    state = _make_state("submitted")
    assert state.poll_failures == 0


def test_token_route_has_no_asyncio_event() -> None:
    route = TokenRoute(
        context_id="ctx-AAA",
        agent_name="B",
        task_id=None,
        registered_at=datetime.now(),
    )
    assert not hasattr(route, "awaiting_task_id")
