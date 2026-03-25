from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from obelix.core.model.roles import MessageRole
from obelix.core.model.tool_message import ToolCall, ToolResult
from obelix.core.model.usage import Usage


class AssistantMessage(BaseModel):
    """LLM assistant response message"""

    role: MessageRole = Field(default=MessageRole.ASSISTANT)
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls requested by the assistant"
    )
    usage: Usage | None = Field(
        default=None, description="Information about token usage for this LLM call"
    )


class AssistantResponse(BaseModel):
    """Structured agent response with complete execution information"""

    agent_name: str = Field(
        ..., description="Name of the agent that generated the response"
    )
    content: str = Field(..., description="Textual content of the final response")
    tool_results: list[ToolResult] | None = Field(
        default=None, description="Results of tools executed during the conversation"
    )
    error: str | None = Field(default=None, description="Any error during execution")

    def __init__(self, **kwargs):
        # If tool_results is an empty list, set it to None for cleanliness
        if "tool_results" in kwargs and not kwargs["tool_results"]:
            kwargs["tool_results"] = None
        super().__init__(**kwargs)


@dataclass
class StreamEvent:
    """A single event in a streaming LLM response.

    During streaming, text chunks arrive as StreamEvent(token="...").
    The final event has is_final=True and carries the complete
    AssistantMessage and AssistantResponse with all metadata
    (content, tool_results, usage, error).

    When deferred_tool_calls is set, the agent loop has stopped because
    one or more deferred tools returned None — their results must come
    from the client. The caller (e.g. A2A executor) should signal
    input-required and resume the agent with the tool responses.
    """

    token: str | None = None
    assistant_message: AssistantMessage | None = None
    assistant_response: AssistantResponse | None = None
    is_final: bool = field(default=False)
    deferred_tool_calls: list[ToolCall] | None = field(default=None)
    canceled: bool = field(default=False)
