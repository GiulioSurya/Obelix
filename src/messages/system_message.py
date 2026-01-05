# src/messages/system_messages.py
from pydantic import Field

from src.messages.base_message import BaseMessage, MessageRole


class SystemMessage(BaseMessage):
    """System message for LLM instructions"""
    role: MessageRole = Field(default=MessageRole.SYSTEM)