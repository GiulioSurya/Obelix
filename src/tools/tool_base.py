# src/tools/tool_base.py
"""
Classe base per i tool - versione semplificata.

Usare insieme al decoratore @tool:
    from src.tools.tool_decorator import tool
    from src.tools.tool_base import ToolBase

    @tool(name="my_tool", description="Descrizione")
    class MyTool(ToolBase):
        param: str = Field(...)

        async def execute(self) -> dict:
            return {"result": self.param}
"""
from abc import ABC, abstractmethod
from typing import Any


class ToolBase(ABC):
    """
    Classe base astratta per tutti i tool.

    Attributi popolati dal decoratore @tool:
    - tool_name: str - Nome univoco del tool
    - tool_description: str - Descrizione per LLM
    - _input_schema: Type[BaseModel] - Schema Pydantic per validazione

    Il decoratore @tool si occupa di:
    - Validare name/description obbligatori
    - Creare lo schema Pydantic dai Field
    - Wrappare execute() per gestione automatica errori
    - Popolare gli attributi prima di execute()
    """

    # Attributi popolati dal decoratore @tool
    tool_name: str = None
    tool_description: str = None

    @classmethod
    def create_schema(cls):
        """
        Genera schema MCP per il tool.

        Questo metodo viene sovrascritto dal decoratore @tool.
        Se chiamato senza decoratore, solleva errore.
        """
        raise NotImplementedError(
            f"Classe {cls.__name__} deve usare il decoratore @tool. "
            f"Esempio: @tool(name='...', description='...')"
        )

    @abstractmethod
    async def execute(self) -> Any:
        """
        Esegue la logica del tool.

        Gli attributi Field sono già popolati dal decoratore prima della chiamata.
        Non è necessario gestire ToolCall o ToolResult - il decoratore wrappa tutto.

        Returns:
            Any: Risultato dell'esecuzione (wrappato automaticamente in ToolResult)

        Raises:
            Exception: Qualsiasi eccezione viene catturata e convertita in ToolResult con status=ERROR
        """
        pass
