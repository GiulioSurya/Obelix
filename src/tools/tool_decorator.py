# src/tools/tool_decorator.py
"""
Decoratore @tool per definire tool in modo dichiarativo.

Esempio d'uso:
    @tool(name="sql_query_executor", description="Esegue query SQL")
    class SqlQueryExecutor(ToolBase):
        sql_query: str = Field(..., description="Query SQL valida")

        def __init__(self, oracle_conn: OracleConnection):
            self.oracle_conn = oracle_conn

        async def execute(self) -> dict:
            # self.sql_query già popolato!
            return self.oracle_conn.execute_query(self.sql_query)
"""
import time
from typing import Type, Any, get_type_hints

from pydantic import create_model
from pydantic.fields import FieldInfo

from src.messages.tool_message import ToolCall, ToolResult, ToolStatus, MCPToolSchema


def tool(name: str = None, description: str = None):
    """
    Decoratore per definire tool in modo dichiarativo.

    Args:
        name: Nome del tool (OBBLIGATORIO)
        description: Descrizione del tool (OBBLIGATORIO)

    Raises:
        ValueError: Se name o description mancano (a import time)

    Returns:
        Decoratore che trasforma la classe in un tool completo
    """
    def decorator(cls: Type) -> Type:
        # 1. Validazione obbligatoria - fail-fast a import time
        if not name:
            raise ValueError(
                f"Tool {cls.__name__}: 'name' è obbligatorio nel decoratore @tool. "
                f"Uso: @tool(name='my_tool', description='...')"
            )
        if not description:
            raise ValueError(
                f"Tool {cls.__name__}: 'description' è obbligatorio nel decoratore @tool. "
                f"Uso: @tool(name='my_tool', description='...')"
            )

        # 2. Aggiungi attributi di classe per accesso diretto
        cls.tool_name = name
        cls.tool_description = description

        # 3. Estrai i Field dalla classe per creare schema Pydantic
        cls._input_schema = _create_input_schema(cls, name)

        # 4. Wrap del metodo execute originale
        original_execute = cls.execute

        async def wrapped_execute(self, tool_call: ToolCall) -> ToolResult:
            """Execute wrappato che gestisce validazione e errori automaticamente"""
            start_time = time.time()
            try:
                # Valida argomenti e popola attributi sull'istanza
                validated = self._input_schema(**tool_call.arguments)
                for field_name in validated.model_fields:
                    setattr(self, field_name, getattr(validated, field_name))

                # Chiama execute originale (senza parametri, attributi già popolati)
                result = await original_execute(self)

                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=result,
                    status=ToolStatus.SUCCESS,
                    execution_time=time.time() - start_time
                )
            except Exception as e:
                return ToolResult(
                    tool_name=tool_call.name,
                    tool_call_id=tool_call.id,
                    result=None,
                    status=ToolStatus.ERROR,
                    error=str(e),
                    execution_time=time.time() - start_time
                )

        cls.execute = wrapped_execute

        # 5. Aggiungi create_schema() come metodo di classe
        @classmethod
        def create_schema(cls_inner) -> MCPToolSchema:
            """Genera schema MCP dal modello Pydantic interno"""
            return MCPToolSchema(
                name=cls_inner.tool_name,
                description=cls_inner.tool_description,
                inputSchema=cls_inner._input_schema.model_json_schema(),
                outputSchema={"type": "object", "additionalProperties": True}
            )

        cls.create_schema = create_schema

        return cls

    return decorator


def _create_input_schema(cls: Type, tool_name: str) -> Type:
    """
    Crea modello Pydantic dinamico dai Field annotati sulla classe.

    Estrae tutti gli attributi di classe che hanno:
    - Type annotation (es. sql_query: str)
    - Default che è un FieldInfo (es. = Field(...))

    Args:
        cls: La classe tool da analizzare
        tool_name: Nome del tool per naming del modello

    Returns:
        Modello Pydantic dinamico con i campi estratti
    """
    fields = {}

    # Ottieni type hints dalla classe
    try:
        hints = get_type_hints(cls)
    except Exception:
        # Fallback se get_type_hints fallisce (es. forward references)
        hints = getattr(cls, '__annotations__', {})

    # Estrai campi con FieldInfo
    for attr_name, attr_type in hints.items():
        # Salta attributi interni e di classe base
        if attr_name.startswith('_'):
            continue

        # Ottieni il default dalla classe
        default = getattr(cls, attr_name, ...)

        # Se è un FieldInfo, includilo nello schema
        if isinstance(default, FieldInfo):
            fields[attr_name] = (attr_type, default)

    # Crea modello Pydantic dinamico
    schema_class_name = f"{tool_name.title().replace('_', '')}Schema"
    return create_model(schema_class_name, **fields)
