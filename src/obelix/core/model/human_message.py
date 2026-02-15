# src/core/model/human_message.py
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from obelix.core.model.roles import MessageRole


class HumanMessage(BaseModel):
    """Message from human user"""

    role: MessageRole = Field(default=MessageRole.HUMAN)
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
