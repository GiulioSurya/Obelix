# src/messages/tool_messages.py
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict
from enum import Enum
from datetime import datetime

from src.messages.roles import  MessageRole

class ToolCall(BaseModel):
    id: str = Field(..., description="Unique ID of the tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="Name of the tool")
    tool_call_id: str = Field(..., description="ID of the tool call")
    result: Any = Field(..., description="Result of the tool")
    status: ToolStatus = Field(default=ToolStatus.SUCCESS)
    error: Optional[str] = Field(None)
    execution_time: Optional[float] = Field(None)

    def __init__(self, **kwargs):
        # If there is an error, truncate it automatically
        if 'error' in kwargs and kwargs['error']:
            kwargs['error'] = self._truncate_error_if_needed(kwargs['error'])

        super().__init__(**kwargs)

    @staticmethod
    def _truncate_error_if_needed(error_msg: str, max_length: int = 2000) -> str:
        """
        Truncate error messages that are too long while preserving beginning and end

        Args:
            error_msg: The original error message
            max_length: Maximum length beyond which to truncate (default: 2000)

        Returns:
            Truncated message if necessary
        """
        if not error_msg or len(error_msg) <= max_length:
            return error_msg

        head_chars = 800
        tail_chars = 800
        separator = " ... [truncated] ... "

        # Ensure that head + tail + separator do not exceed max_length
        available_space = max_length - len(separator)
        if head_chars + tail_chars > available_space:
            # Reduce proportionally
            ratio = available_space / (head_chars + tail_chars)
            head_chars = int(head_chars * ratio)
            tail_chars = int(tail_chars * ratio)

        truncated = error_msg[:head_chars] + separator + error_msg[-tail_chars:]
        return truncated

    class Config:
        arbitrary_types_allowed = True


class ToolMessage(BaseModel):
    role: MessageRole = Field(default=MessageRole.TOOL)
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tool_results: List[ToolResult] = Field(...)

    def __init__(self, tool_results: List[ToolResult], **kwargs):
        # *** DEPRECATED *** content auto-generation is redundant, see TODO.md
        # if 'content' not in kwargs:
        #     kwargs['content'] = self._generate_content_summary(tool_results)
        super().__init__(tool_results=tool_results, **kwargs)
    #
    # # *** DEPRECATED *** - kept for backward compatibility, see TODO.md
    # @staticmethod
    # def _generate_content_summary(tool_results: List[ToolResult]) -> str:
    #     if not tool_results:
    #         return "No tool results"
    #
    #     summaries = []
    #     for result in tool_results:
    #         if result.status == ToolStatus.SUCCESS:
    #             summaries.append(f"{result.tool_name}: {result.result}")
    #         else:
    #             summaries.append(f"{result.tool_name}: {result.status.value} - {result.error}")
    #
    #     return "; ".join(summaries)
    #

class MCPToolSchema(BaseModel):
    """Tool schema in MCP standard format - NOT a message"""

    # MCP Standard Fields
    name: str = Field(..., description="Unique identifier for the tool")
    description: str = Field(..., description="Description of the tool")
    title: Optional[str] = Field(None, description="Friendly name for UI")
    inputSchema: Dict[str, Any] = Field(..., description="JSON Schema for parameters")
    outputSchema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for response")
    annotations: Optional[Dict[str, Any]] = Field(None, description="Behavioral hints")

    class Config:
        # Strict validation
        validate_assignment = True
        # Field name as it appears in JSON (maintains MCP camelCase)
        populate_by_name = True