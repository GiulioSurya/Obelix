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
