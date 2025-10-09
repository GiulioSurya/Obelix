# src/messages/base_messages.py
from pydantic import BaseModel, Field
from typing import Dict, Any
from enum import Enum
from datetime import datetime


class MessageRole(str, Enum):
    """Ruoli dei messaggi standardizzati"""
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "assistant"
    TOOL = "tool"


class BaseMessage(BaseModel):
    """Classe base per tutti i messaggi standardizzati"""
    role: MessageRole = Field(..., description="Ruolo del messaggio")
    content: str = Field(default="", description="Contenuto testuale del messaggio")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp di creazione")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadati aggiuntivi")

    class Config:
        # Validazione strict per assignment
        validate_assignment = True