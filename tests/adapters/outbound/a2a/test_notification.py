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
    # Exact structural assertion: escaping inside <result>, no raw < or &
    assert "<result>value &lt; 5 &amp; status = &#x27;ok&#x27;</result>" in msg.content


def test_xml_escape_in_agent_name() -> None:
    msg = build_remote_task_update_message(
        task_id="t-008",
        agent_name="<Bad>",
        status="completed",
        result_text="ok",
    )
    assert "<agent>&lt;Bad&gt;</agent>" in msg.content


def test_xml_escape_in_deferred_calls() -> None:
    deferred = [{"command": 'echo "<hello>"', "id": "c-1"}]
    msg = build_remote_task_update_message(
        task_id="t-009",
        agent_name="B",
        status="input_required",
        deferred_calls=deferred,
    )
    # The JSON inside <deferred_tool_calls> must be HTML-escaped, so the raw
    # angle brackets in the original command string must not survive verbatim.
    assert "<hello>" not in msg.content.replace("<deferred_tool_calls>", "").replace(
        "</deferred_tool_calls>", ""
    )
    assert "&lt;hello&gt;" in msg.content


def test_payload_status_mismatch_result_text_with_failed() -> None:
    with pytest.raises(ValueError, match="result_text is only valid"):
        build_remote_task_update_message(
            task_id="t-010",
            agent_name="B",
            status="failed",
            result_text="oops",
        )


def test_payload_status_mismatch_error_text_with_completed() -> None:
    with pytest.raises(ValueError, match="error_text is only valid"):
        build_remote_task_update_message(
            task_id="t-011",
            agent_name="B",
            status="completed",
            error_text="boom",
        )


def test_payload_status_mismatch_deferred_calls_with_completed() -> None:
    with pytest.raises(ValueError, match="deferred_calls is only valid"):
        build_remote_task_update_message(
            task_id="t-012",
            agent_name="B",
            status="completed",
            deferred_calls=[{"x": 1}],
        )


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
