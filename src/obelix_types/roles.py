# src/obelix_types/base_messages.py
from pydantic import BaseModel, Field
from typing import Dict, Any
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Roles of standardized obelix_types"""
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"

