import pytest

from obelix.adapters.outbound.a2a.notification import (
    build_remote_task_update_message,
)
from obelix.core.model.human_message import HumanMessage


def test_completed_notification_contains_status_and_result() -> None:
    msg = build_remote_task_update_message(
        task_id="t-001",
        agent_name="B",
        status="completed",
        result_text="Inventory has 42 SKUs.",
    )
    assert isinstance(msg, HumanMessage)
    assert "<remote_task_update>" in msg.content
    assert "<task_id>t-001</task_id>" in msg.content
    assert "<agent>B</agent>" in msg.content
    assert "<status>completed</status>" in msg.content
    assert "<result>Inventory has 42 SKUs.</result>" in msg.content


def test_failed_notification_contains_error_not_result() -> None:
    msg = build_remote_task_update_message(
        task_id="t-002",
        agent_name="C",
        status="failed",
        error_text="Database connection lost",
    )
    assert "<status>failed</status>" in msg.content
    assert "<error>Database connection lost</error>" in msg.content
    assert "<result>" not in msg.content


def test_input_required_notification_contains_deferred_calls() -> None:
    deferred = [{"tool_name": "bash", "arguments": {"command": "ls"}, "id": "c-1"}]
    msg = build_remote_task_update_message(
        task_id="t-003",
        agent_name="B",
        status="input_required",
        deferred_calls=deferred,
    )
    assert "<status>input_required</status>" in msg.content
    assert "<deferred_tool_calls>" in msg.content
    assert "bash" in msg.content
    assert "c-1" in msg.content


def test_xml_special_chars_escaped() -> None:
    msg = build_remote_task_update_message(
        task_id="t-004",
        agent_name="X",
        status="completed",
        result_text="value < 5 & status = 'ok'",
    )
    # The raw chars must not appear unescaped inside <result>
    assert "&lt;" in msg.content or "<result>value &lt; 5" in msg.content
    assert "&amp;" in msg.content


def test_canceled_status_is_supported() -> None:
    msg = build_remote_task_update_message(
        task_id="t-005",
        agent_name="B",
        status="canceled",
        error_text="User canceled",
    )
    assert "<status>canceled</status>" in msg.content


def test_rejected_status_is_supported() -> None:
    msg = build_remote_task_update_message(
        task_id="t-006",
        agent_name="B",
        status="rejected",
        error_text="Hook rejected",
    )
    assert "<status>rejected</status>" in msg.content


def test_unknown_status_raises() -> None:
    with pytest.raises(ValueError, match="unknown status"):
        build_remote_task_update_message(
            task_id="t-007",
            agent_name="B",
            status="weird",
        )
