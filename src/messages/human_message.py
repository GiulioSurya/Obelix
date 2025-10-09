# src/messages/human_messages.py
from pydantic import Field

from src.messages.base_message import BaseMessage, MessageRole


class HumanMessage(BaseMessage):
    """Messaggio dall'utente umano"""
    role: MessageRole = Field(default=MessageRole.HUMAN)