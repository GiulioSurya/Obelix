from abc import ABC, abstractmethod
from typing import Type

from src.tools.tool_schema import ToolSchema
from src.messages.tool_message import ToolCall, ToolResult
from src.messages.tool_message import MCPToolSchema


class ToolBase(ABC):
    """Classe base per l'esecuzione dei tool - Async Interface"""

    schema_class: Type[ToolSchema] = None

    @classmethod
    def create_schema(cls) -> MCPToolSchema:
        """Crea schema MCP standard per il tool"""
        if not cls.schema_class:
            raise ValueError("Devi definire schema_class")

        name = cls.schema_class.get_tool_name()
        description = cls.schema_class.get_tool_description()
        input_schema = cls.schema_class.model_json_schema()

        return MCPToolSchema(
            name=name,
            description=description,
            inputSchema=input_schema,
            outputSchema={"type": "object", "additionalProperties": True}
        )

    @abstractmethod
    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Esegue il tool con parametri standardizzati - ASYNC"""
        pass