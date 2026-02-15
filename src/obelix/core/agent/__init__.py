"""
Base Agent module - Core classes for agent creation and composition.
"""

from obelix.core.agent.agent_factory import AgentFactory, AgentSpec
from obelix.core.agent.base_agent import BaseAgent
from obelix.core.agent.event_contracts import EventContract, get_event_contracts
from obelix.core.agent.hooks import AgentEvent, AgentStatus, Hook, HookDecision, Outcome
from obelix.core.agent.shared_memory import SharedMemoryGraph
from obelix.core.agent.subagent_wrapper import SubAgentWrapper

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
