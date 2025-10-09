# src/messages/tool_messages.py
from pydantic import BaseModel, Field
from typing import List, Any, Optional, Dict
from enum import Enum

from src.messages.base_message import BaseMessage, MessageRole

class ToolCall(BaseModel):
    id: str = Field(..., description="ID univoco della chiamata tool")
    name: str = Field(..., description="Nome del tool da chiamare")
    arguments: Dict[str, Any] = Field(..., description="Argomenti per il tool")


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="Nome del tool")
    tool_call_id: str = Field(..., description="ID della chiamata tool")
    result: Any = Field(..., description="Risultato del tool")
    status: ToolStatus = Field(default=ToolStatus.SUCCESS)
    error: Optional[str] = Field(None)
    execution_time: Optional[float] = Field(None)

    def __init__(self, **kwargs):
        # Se c'è un errore, troncalo automaticamente
        if 'error' in kwargs and kwargs['error']:
            kwargs['error'] = self._truncate_error_if_needed(kwargs['error'])

        super().__init__(**kwargs)

    @staticmethod
    def _truncate_error_if_needed(error_msg: str, max_length: int = 500) -> str:
        """
        Tronca messaggi di errore troppo lunghi mantenendo inizio e fine

        Args:
            error_msg: Il messaggio di errore originale
            max_length: Lunghezza massima oltre la quale troncare

        Returns:
            Messaggio troncato se necessario
        """
        if not error_msg or len(error_msg) <= max_length:
            return error_msg

        head_chars = 200
        tail_chars = 200
        separator = " ... [truncated] ... "

        # Assicurati che head + tail + separator non superino max_length
        available_space = max_length - len(separator)
        if head_chars + tail_chars > available_space:
            # Riduci proporzionalmente
            ratio = available_space / (head_chars + tail_chars)
            head_chars = int(head_chars * ratio)
            tail_chars = int(tail_chars * ratio)

        truncated = error_msg[:head_chars] + separator + error_msg[-tail_chars:]
        return truncated

    class Config:
        arbitrary_types_allowed = True


class ToolMessage(BaseMessage):
    role: MessageRole = Field(default=MessageRole.TOOL)
    tool_results: List[ToolResult] = Field(...)

    def __init__(self, tool_results: List[ToolResult], **kwargs):
        if 'content' not in kwargs:
            kwargs['content'] = self._generate_content_summary(tool_results)
        super().__init__(tool_results=tool_results, **kwargs)

    @staticmethod
    def _generate_content_summary(tool_results: List[ToolResult]) -> str:
        if not tool_results:
            return "No tool results"

        summaries = []
        for result in tool_results:
            if result.status == ToolStatus.SUCCESS:
                summaries.append(f"{result.tool_name}: {result.result}")
            else:
                summaries.append(f"{result.tool_name}: {result.status.value} - {result.error}")

        return "; ".join(summaries)


class MCPToolSchema(BaseModel):
    """Schema tool in formato MCP standard - NON è un messaggio"""

    # MCP Standard Fields
    name: str = Field(..., description="Identificatore univoco del tool")
    description: Optional[str] = Field(None, description="Descrizione del tool")
    title: Optional[str] = Field(None, description="Nome friendly per UI")
    inputSchema: Dict[str, Any] = Field(..., description="JSON Schema parametri")
    outputSchema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema risposta")
    annotations: Optional[Dict[str, Any]] = Field(None, description="Suggerimenti comportamentali")

    class Config:
        # Validazione strict
        validate_assignment = True
        # Nome del campo come appare in JSON (mantiene camelCase MCP)
        populate_by_name = True