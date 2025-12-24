# src/mcp/mcp_tool.py
import time
from typing import Any

from src.tools.tool_base import ToolBase
from src.messages.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema
from src.tools.mcp.run_time_manager import MCPRuntimeManager


class MCPTool(ToolBase):
    """
    Tool che wrappa un tool MCP esterno usando Runtime Manager.

    MODIFICHE FASE 3 - Validazione Centralizzata:
    - RIMOSSA validazione manuale (_convert_args_using_mcp_schema)
    - RIMOSSA conversione tipi custom (_convert_value_by_json_schema)
    - execute() ora passa argomenti direttamente (già validati dal manager)
    - Interfaccia pubblica INVARIATA (execute, create_schema)
    - Codice molto più semplice: da 150+ a 80 righe
    - Responsabilità unica: orchestrazione e formattazione output

    La validazione e conversione tipi ora avviene in MCPClientManager tramite
    MCPValidator, sfruttando Pydantic per conversione automatica.

    Note:
        MCPTool NON usa il decoratore @tool perché lo schema è dinamico
        (proviene dal server MCP). Implementa direttamente tool_name,
        tool_description e create_schema().
    """

    def __init__(self, tool_name: str, manager: MCPRuntimeManager):
        """
        Inizializza MCPTool per un tool specifico.

        Args:
            tool_name: Nome del tool MCP da wrappare
            manager: Runtime manager per comunicazione sincrona
        """
        self.manager = manager
        self._init_tool_metadata(tool_name)

    def _init_tool_metadata(self, tool_name: str) -> None:
        """
        Inizializza tool_name e tool_description dal server MCP.

        Args:
            tool_name: Nome del tool da cercare

        Raises:
            ValueError: Se il tool non esiste nel server MCP
        """
        tool = self.manager.find_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Popola attributi direttamente (compatibile con nuova API)
        self.tool_name = tool.name
        self.tool_description = tool.description or f"MCP tool: {tool.name}"

    def create_schema(self) -> MCPToolSchema:
        """
        Crea schema usando direttamente definizione MCP dal server.

        Interfaccia INVARIATA - stesso signature e comportamento.

        Returns:
            MCPToolSchema: Schema completo del tool MCP

        Raises:
            ValueError: Se tool non trovato
        """
        tool = self.manager.find_tool(self.tool_name)
        if not tool:
            raise ValueError(f"Tool {self.tool_name} not found")

        return MCPToolSchema(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.inputSchema,
            title=getattr(tool, 'title', None),
            outputSchema=getattr(tool, 'outputSchema', None),
            annotations=getattr(tool, 'annotations', None)
        )

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Esegue tool MCP passando argomenti direttamente al manager.

        MODIFICHE FASE 3:
        - RIMOSSA _convert_args_using_mcp_schema() completamente
        - RIMOSSA logica conversione tipi manuale
        - Argomenti passati direttamente (già validati in MCPClientManager)
        - Interfaccia INVARIATA (stesso input/output)
        - Focus su orchestrazione e gestione risultati

        Il flusso ora è:
        1. LLM → MCPTool.execute() con argomenti raw
        2. MCPTool → MCPRuntimeManager.call_tool() (sincrono)
        3. MCPRuntimeManager → MCPClientManager.call_tool() (async interno)
        4. MCPClientManager valida con MCPValidator (Pydantic automatico)
        5. Invio al server MCP con argomenti validati

        Args:
            tool_call: ToolCall con nome tool e argomenti

        Returns:
            ToolResult: Risultato esecuzione con status, timing, contenuto
        """
        start_time = time.time()

        try:
            # SEMPLIFICATO: Nessuna conversione argomenti, già validati dal manager
            # La validazione/conversione avviene in:
            # MCPRuntimeManager → MCPClientManager → MCPValidator (Pydantic)
            mcp_result = self.manager.call_tool(tool_call.name, tool_call.arguments)
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=self._extract_content(mcp_result),
                status=ToolStatus.SUCCESS,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time

            return ToolResult(
                tool_name=tool_call.name,
                tool_call_id=tool_call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=str(e),
                execution_time=execution_time
            )

    def _extract_content(self, mcp_result) -> Any:
        """
        Estrae contenuto da risultato MCP.

        Unica responsabilità rimasta del tool: formattare output MCP
        in formato comprensibile per il sistema chiamante.

        Args:
            mcp_result: Risultato grezzo dal server MCP

        Returns:
            Any: Contenuto estratto e formattato
        """
        if hasattr(mcp_result, 'content'):
            if isinstance(mcp_result.content, list) and len(mcp_result.content) > 0:
                first_item = mcp_result.content[0]
                if hasattr(first_item, 'text'):
                    content = first_item.text

                    # Arricchimento errori per debug - comportamento invariato
                    if "Internal error" in content and "400" in content:
                        return f"{content} | Check parameters format and values"

                    return content
        return str(mcp_result)

