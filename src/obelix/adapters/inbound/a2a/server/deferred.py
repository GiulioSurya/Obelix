"""Deferred tool protocol: inject client responses and validate with OutputSchema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.types import DataPart, Message

from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.adapters.inbound.a2a.server.context import ContextEntry
    from obelix.core.model.tool_message import ToolCall

logger = get_logger(__name__)


def inject_deferred_response(entry: ContextEntry, message: Message) -> None:
    """Inject the client's response as ToolMessage for deferred tools.

    Extracts structured data from the DataPart in the message and
    validates it against the tool's OutputSchema if present.
    """
    # Extract structured data from DataPart
    structured_data: dict | None = None
    for part in message.parts:
        if isinstance(part.root, DataPart):
            structured_data = part.root.data
            break

    tool_results = []
    for call in entry.deferred_tool_calls:
        result_data = parse_deferred_result(
            call, entry.deferred_tools, structured_data or {}
        )
        tool_results.append(
            ToolResult(
                tool_name=call.name,
                tool_call_id=call.id,
                result=result_data,
                status=ToolStatus.SUCCESS,
            )
        )
    entry.history.append(ToolMessage(tool_results=tool_results))
    entry.deferred_tool_calls = None
    entry.deferred_tools = None
    logger.info(f"[A2A] Injected deferred response | tool_count={len(tool_results)}")


def parse_deferred_result(
    call: ToolCall,
    tools: list | None,
    data: dict,
) -> dict:
    """Parse a structured DataPart response for a deferred tool.

    If the tool has an OutputSchema, validates the data against it.
    Otherwise returns the data dict directly.
    """
    output_schema = None
    for tool in tools or []:
        if getattr(tool, "tool_name", None) == call.name:
            output_schema = getattr(tool, "_output_schema", None)
            break

    if output_schema:
        try:
            validated = output_schema(**data)
            return validated.model_dump()
        except Exception:
            return data

    return data
