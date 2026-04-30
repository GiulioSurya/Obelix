"""Build the XML-wrapped HumanMessage that surfaces remote task updates
into the parent agent's conversation history."""

from __future__ import annotations

import json
from html import escape

from obelix.core.model.human_message import HumanMessage

_VALID_STATUSES = {
    "completed",
    "failed",
    "canceled",
    "rejected",
    "input_required",
}


def build_remote_task_update_message(
    *,
    task_id: str,
    agent_name: str,
    status: str,
    result_text: str | None = None,
    error_text: str | None = None,
    deferred_calls: list[dict] | None = None,
) -> HumanMessage:
    """Build the user-role HumanMessage that the executor drains into the
    agent's conversation history at the next request boundary.

    Body shape (XML inside content):
        <remote_task_update>
          <task_id>...</task_id>
          <agent>...</agent>
          <status>...</status>
          <result>...</result>          # only when status == "completed"
          <error>...</error>            # only when status in failed/rejected/canceled
          <deferred_tool_calls>...</deferred_tool_calls>  # only when status == input_required
        </remote_task_update>

    Special characters in result_text / error_text are HTML-escaped to
    keep the wrapper unambiguous (e.g. result text containing '<' won't
    break the XML).
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"unknown status: {status}")

    parts: list[str] = [
        "<remote_task_update>",
        f"  <task_id>{escape(task_id)}</task_id>",
        f"  <agent>{escape(agent_name)}</agent>",
        f"  <status>{escape(status)}</status>",
    ]
    if status == "completed" and result_text is not None:
        parts.append(f"  <result>{escape(result_text)}</result>")
    elif status in ("failed", "rejected", "canceled") and error_text is not None:
        parts.append(f"  <error>{escape(error_text)}</error>")
    elif status == "input_required" and deferred_calls is not None:
        parts.append(
            f"  <deferred_tool_calls>{escape(json.dumps(deferred_calls))}</deferred_tool_calls>"
        )

    parts.append("</remote_task_update>")

    return HumanMessage(content="\n".join(parts))
