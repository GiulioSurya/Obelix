"""Bidirectional conversion between A2A Part types and Obelix ContentPart types.

A2A -> Obelix (inbound): client message parts to HumanMessage text + attachments.
Obelix -> A2A (outbound): AssistantResponse to artifact Part list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.types import (
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Part,
    TextPart,
)

from obelix.core.model.content import DataContent, FileContent

if TYPE_CHECKING:
    from obelix.core.model.assistant_message import AssistantResponse
    from obelix.core.model.content import ContentPart
    from obelix.core.model.tool_message import ToolCall


def a2a_parts_to_obelix(a2a_parts: list[Part]) -> tuple[str, list[ContentPart]]:
    """Extract text + attachments from A2A Part list.

    Returns:
        (text, attachments) — text is concatenated TextPart content,
        attachments are FileContent/DataContent from FilePart/DataPart.
    """
    text_pieces: list[str] = []
    attachments: list[ContentPart] = []

    for part in a2a_parts:
        root = part.root
        if isinstance(root, TextPart):
            if root.text:
                text_pieces.append(root.text)

        elif isinstance(root, FilePart):
            file = root.file
            if isinstance(file, FileWithBytes):
                # bytes is already base64 in the SDK
                attachments.append(
                    FileContent(
                        data=file.bytes,
                        mime_type=file.mime_type or "application/octet-stream",
                        filename=file.name,
                        is_url=False,
                    )
                )
            elif isinstance(file, FileWithUri):
                attachments.append(
                    FileContent(
                        data=file.uri,
                        mime_type=file.mime_type or "application/octet-stream",
                        filename=file.name,
                        is_url=True,
                    )
                )

        elif isinstance(root, DataPart):
            attachments.append(
                DataContent(
                    data=root.data,
                    metadata=root.metadata,
                )
            )

    return "".join(text_pieces), attachments


def obelix_response_to_a2a_parts(response: AssistantResponse) -> list[Part]:
    """Convert AssistantResponse to A2A Part list for the artifact.

    - response.content -> TextPart
    - response.tool_results with dict results -> DataPart per result
    """
    parts: list[Part] = []

    if response.content:
        parts.append(Part(root=TextPart(text=response.content)))

    if response.tool_results:
        for result in response.tool_results:
            if isinstance(result.result, dict):
                parts.append(
                    Part(
                        root=DataPart(
                            data=result.result,
                            metadata={
                                "type": "tool_result",
                                "tool_name": result.tool_name,
                            },
                        )
                    )
                )

    # Ensure at least one part (A2A requires non-empty parts)
    if not parts:
        parts.append(Part(root=TextPart(text="")))

    return parts


def deferred_calls_to_a2a_parts(calls: list[ToolCall]) -> list[Part]:
    """Convert deferred tool calls to DataPart (instead of JSON string in TextPart).

    Each call becomes a DataPart with tool_name and arguments.
    """
    payload = [{"tool_name": call.name, "arguments": call.arguments} for call in calls]
    return [Part(root=DataPart(data={"deferred_tool_calls": payload}))]
