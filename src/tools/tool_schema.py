from pydantic import BaseModel
from typing import ClassVar



class ToolSchema(BaseModel):
    """Classe base per definire schema e metadati dei tool"""

    # Metadati del tool (attributi di classe)
    tool_name: ClassVar[str] = None
    tool_description: ClassVar[str] = None

    @classmethod
    def get_tool_name(cls) -> str:
        """Ottiene il nome del tool"""
        if cls.tool_name:
            return cls.tool_name
        # Fallback: deriva dal nome della classe
        return cls.__name__.lower().replace('schema', '').replace('tool', '')

    @classmethod
    def get_tool_description(cls) -> str:
        """Ottiene la descrizione del tool"""
        if cls.tool_description:
            return cls.tool_description
        # Fallback: usa il docstring
        return cls.__doc__ or f"Tool {cls.get_tool_name()}"


