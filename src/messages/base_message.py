# src/messages/base_messages.py
from pydantic import BaseModel, Field
from typing import Dict, Any
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Roles of standardized messages"""
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"


class BaseMessage(BaseModel):
    """Base class for all standardized messages"""
    role: MessageRole = Field(..., description="Message role")
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        # Strict validation for assignment
        validate_assignment = True