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

