"""Per-context state management for the A2A server.

Each A2A context_id maps to a _ContextEntry that holds conversation
history, idle gate (for serialization), and deferred tool state.
The ContextStore wraps the LRU-eviction OrderedDict.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import TYPE_CHECKING

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import StandardMessage
    from obelix.core.model.tool_message import ToolCall

logger = get_logger(__name__)


class ContextEntry:
    """Holds the state for a single conversation context."""

    __slots__ = (
        "history",
        "idle",
        "deferred_tool_calls",
        "deferred_tools",
        "trace_session",
        "trace_span",
        "active_agent",
        "client_info",
    )

    def __init__(self) -> None:
        self.history: list[StandardMessage] = []
        self.idle = asyncio.Event()
        self.idle.set()  # starts as idle (ready for new executions)
        self.deferred_tool_calls: list[ToolCall] | None = None
        self.deferred_tools: list | None = None  # tool snapshot for OutputSchema lookup
        self.trace_session = None  # TraceSession saved when loop stops for deferred
        self.trace_span = None  # Current span saved when loop stops for deferred
        self.active_agent: BaseAgent | None = None  # ref to running agent for cancel
        self.client_info: dict | None = None  # client shell environment for BashTool


class ContextStore:
    """LRU context store with async-safe access.

    Thread safety: get_or_create() must be called under the caller's lock.
    """

    def __init__(self, max_contexts: int) -> None:
        self._max_contexts = max_contexts
        self._contexts: OrderedDict[str, ContextEntry] = OrderedDict()

    def get_or_create(self, context_id: str) -> ContextEntry:
        """Get or create a context entry, evicting oldest if over limit."""
        if context_id in self._contexts:
            self._contexts.move_to_end(context_id)
            return self._contexts[context_id]

        # Evict oldest contexts if we're at capacity
        while len(self._contexts) >= self._max_contexts:
            evicted_id, _evicted = self._contexts.popitem(last=False)
            logger.debug(f"[A2A] Evicted context | context_id={evicted_id}")

        entry = ContextEntry()
        self._contexts[context_id] = entry
        return entry
