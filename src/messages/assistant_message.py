
from pydantic import Field, BaseModel
from typing import List, Optional,Dict,Any

from src.messages.base_message import BaseMessage, MessageRole
from src.messages.tool_message import ToolCall



class AssistantMessage(BaseMessage):
    """Messaggio di risposta dell'assistant LLM"""
    role: MessageRole = Field(default=MessageRole.ASSISTANT)
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls richiesti dall'assistant")


class AssistantResponse(BaseModel):
    """Risposta strutturata dell'agent con informazioni complete sull'esecuzione"""

    agent_name: str = Field(..., description="Nome dell'agent che ha generato la risposta")
    content: str = Field(..., description="Contenuto testuale della risposta finale")
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Risultati dei tool eseguiti durante la conversazione"
    )
    error: Optional[str] = Field(default=None, description="Eventuale errore durante l'esecuzione")

    def __init__(self, **kwargs):
        # Se tool_results è una lista vuota, la impostiamo a None per pulizia
        if 'tool_results' in kwargs and not kwargs['tool_results']:
            kwargs['tool_results'] = None
        super().__init__(**kwargs)