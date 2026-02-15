from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from obelix.core.model.roles import MessageRole


class SystemMessage(BaseModel):
    """System message for LLM instructions"""

    role: MessageRole = Field(default=MessageRole.SYSTEM)
    content: str = Field(default="", description="Textual content of the message")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Creation timestamp"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
