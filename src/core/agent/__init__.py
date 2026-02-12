"""
Base Agent module - Core classes for agent creation and composition.
"""

from src.core.agent.base_agent import BaseAgent
from src.core.agent.shared_memory import SharedMemoryGraph
from src.core.agent.subagent_wrapper import SubAgentWrapper
from src.core.agent.hooks import AgentEvent, AgentStatus, Hook, HookDecision, Outcome
from src.core.agent.event_contracts import EventContract, get_event_contracts
from src.core.agent.agent_factory import AgentFactory, AgentSpec

__all__ = [
    "BaseAgent",
    "SharedMemoryGraph",
    "SubAgentWrapper",
    "AgentEvent",
    "AgentStatus",
    "Hook",
    "HookDecision",
    "Outcome",
    "EventContract",
    "get_event_contracts",
    "AgentFactory",
    "AgentSpec",
]