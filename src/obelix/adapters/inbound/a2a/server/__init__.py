"""A2A server package — bridges a2a-sdk to Obelix BaseAgent."""

from obelix.adapters.inbound.a2a.server.context import ContextEntry, ContextStore
from obelix.adapters.inbound.a2a.server.deferred import (
    inject_deferred_response,
    parse_deferred_result,
)
from obelix.adapters.inbound.a2a.server.executor import ObelixAgentExecutor
from obelix.adapters.inbound.a2a.server.helpers import (
    DEFAULT_MAX_CONTEXTS,
    agent_message,
)

__all__ = [
    "ObelixAgentExecutor",
    "ContextEntry",
    "ContextStore",
    "inject_deferred_response",
    "parse_deferred_result",
    "agent_message",
    "DEFAULT_MAX_CONTEXTS",
]
