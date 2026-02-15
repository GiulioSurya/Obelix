# src/core/model/roles.py
from enum import Enum


class MessageRole(str, Enum):
    """Roles of standardized messages"""

    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"
