"""
Base Agent module - Core classes for agent creation and composition.
"""

from src.domain.agent.base_agent import BaseAgent
from src.domain.agent.shared_memory import SharedMemoryGraph
from src.domain.agent.subagent_wrapper import SubAgentWrapper

__all__ = [
    "BaseAgent",
    "SharedMemoryGraph",
    "SubAgentWrapper",
]