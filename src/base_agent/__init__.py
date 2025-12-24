"""
Base Agent module - Classi base per la creazione di agenti.
"""

from src.base_agent.base_agent import BaseAgent
from src.base_agent.agent_schema import AgentSchema, ToolDescription

__all__ = [
    "BaseAgent",
    "AgentSchema",
    "ToolDescription",
]