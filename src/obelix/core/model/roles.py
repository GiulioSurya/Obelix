# src/core/model/roles.py
from enum import StrEnum


class MessageRole(StrEnum):
    """Roles of standardized messages"""

    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"
