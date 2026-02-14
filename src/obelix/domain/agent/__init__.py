"""
Base Agent module - Core classes for agent creation and composition.
"""

from obelix.domain.agent.base_agent import BaseAgent
from obelix.domain.agent.subagent_wrapper import SubAgentWrapper

__all__ = [
    "BaseAgent",
    "SubAgentWrapper",
]