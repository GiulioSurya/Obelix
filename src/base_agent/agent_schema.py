# src/agents/agent_schema.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ToolDescription(BaseModel):
    """Descrizione semplificata di un tool per il manager"""
    name: str = Field(..., description="Nome del tool")
    description: str = Field(..., description="Descrizione del tool")
    parameters: List[str] = Field(..., description="Lista delle chiavi dei parametri richiesti")


class AgentInput(BaseModel):
    """Schema di input standard per l'agent"""
    query: str = Field(..., description="Query da eseguire")


class AgentOutput(BaseModel):
    """Schema di output standard per l'agent"""
    result: str = Field(..., description="Risultato dell'esecuzione")


class AgentSchema(BaseModel):
    name: str = Field(default="base-agent", description="Nome dell'agent")
    description: str = Field(default="Agent base", description="Descrizione delle capacità dell'agent")
    capabilities: Dict[str, Any] = Field(default={
        "tools": False,
        "available_tools": []
    })
    input_schema: Dict[str, Any] = Field(default_factory=lambda: AgentInput.model_json_schema())
    output_schema: Dict[str, Any] = Field(..., description="Schema di output dell'agent")
