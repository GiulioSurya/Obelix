# src/messages/system_messages.py
from pydantic import Field

from src.messages.base_message import BaseMessage, MessageRole


class SystemMessage(BaseMessage):
    """Messaggio di sistema per istruzioni LLM"""
    role: MessageRole = Field(default=MessageRole.SYSTEM)