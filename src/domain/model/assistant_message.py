
from pydantic import Field, BaseModel
from typing import List, Optional, Dict, Any
from src.domain.model.roles import MessageRole
from src.domain.model.tool_message import ToolCall, ToolResult
from src.domain.model.usage import Usage
from datetime import datetime



class AssistantMessage(BaseModel):
    """LLM assistant response message"""
    role: MessageRole = Field(default=MessageRole.ASSISTANT)
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls requested by the assistant")
    usage: Optional[Usage] = Field(default=None, description="Information about token usage for this LLM call")


class AssistantResponse(BaseModel):
    """Structured agent response with complete execution information"""

    agent_name: str = Field(..., description="Name of the agent that generated the response")
    content: str = Field(..., description="Textual content of the final response")
    tool_results: Optional[List[ToolResult]] = Field(
        default=None,
        description="Results of tools executed during the conversation"
    )
    error: Optional[str] = Field(default=None, description="Any error during execution")

    def __init__(self, **kwargs):
        # If tool_results is an empty list, set it to None for cleanliness
        if 'tool_results' in kwargs and not kwargs['tool_results']:
            kwargs['tool_results'] = None
        super().__init__(**kwargs)