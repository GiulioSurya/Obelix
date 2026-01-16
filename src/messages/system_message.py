
from pydantic import BaseModel, Field
from typing import Dict, Any

from datetime import datetime

from src.messages.roles import MessageRole


class SystemMessage(BaseModel):
    """System message for LLM instructions"""
    role: MessageRole = Field(default=MessageRole.SYSTEM)
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
