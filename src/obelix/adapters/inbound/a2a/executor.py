"""Bridge between the a2a-sdk AgentExecutor and Obelix BaseAgent.

Thread safety: each A2A context_id gets its own BaseAgent instance
(created via the agent_factory callable). Conversation history is
persisted per context_id so multi-turn conversations work correctly.
Concurrent requests on *different* contexts run in parallel without
interference. Concurrent requests on the *same* context are serialized
via a per-context asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext

    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import StandardMessage

logger = get_logger(__name__)

DEFAULT_MAX_CONTEXTS = 1024


class _ContextEntry:
    """Holds the conversation history and lock for a single context."""

    __slots__ = ("history", "lock")

    def __init__(self) -> None:
        self.history: list[StandardMessage] = []
        self.lock = asyncio.Lock()


class ObelixAgentExecutor(AgentExecutor):
    """Executes an Obelix BaseAgent in response to A2A requests.

    Bridges the a2a-sdk execution model (RequestContext + EventQueue)
    to the Obelix agent model (execute_query_async).

    Each request creates a fresh BaseAgent via the factory callable.
    Conversation history is stored per context_id and injected into
    the fresh agent before execution, then saved back afterwards.
    """

    def __init__(
        self,
        agent_factory: Callable[[], BaseAgent],
        *,
        max_contexts: int = DEFAULT_MAX_CONTEXTS,
    ) -> None:
        self._agent_factory = agent_factory
        self._max_contexts = max_contexts
        self._contexts: OrderedDict[str, _ContextEntry] = OrderedDict()
        self._contexts_lock = asyncio.Lock()

    def _get_or_create_context(self, context_id: str) -> _ContextEntry:
        """Get or create a context entry, evicting oldest if over limit."""
        if context_id in self._contexts:
            self._contexts.move_to_end(context_id)
            return self._contexts[context_id]

        # Evict oldest contexts if we're at capacity
        while len(self._contexts) >= self._max_contexts:
            evicted_id, _ = self._contexts.popitem(last=False)
            logger.debug(f"[A2A] Evicted context | context_id={evicted_id}")

        entry = _ContextEntry()
        self._contexts[context_id] = entry
        return entry

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        # Extract user text from the incoming message
        user_text = context.get_user_input()
        if not user_text:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.failed, message=None),
                    final=True,
                )
            )
            return

        # Get or create the context entry (LRU eviction under global lock)
        async with self._contexts_lock:
            entry = self._get_or_create_context(context_id)

        # Serialize requests on the same context_id
        async with entry.lock:
            await self._execute_with_context(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                entry=entry,
                event_queue=event_queue,
            )

    async def _execute_with_context(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        entry: _ContextEntry,
        event_queue: EventQueue,
    ) -> None:
        """Run the agent with isolated context and persist history."""
        logger.info(
            f"[A2A] Executing agent | task_id={task_id} context_id={context_id} "
            f"text_len={len(user_text)} history_len={len(entry.history)}"
        )

        # Create a fresh agent for this request
        agent = self._agent_factory()

        # Inject conversation history from this context
        if entry.history:
            agent.conversation_history = [agent.system_message, *entry.history]

        # Signal that the agent is working
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            response = await agent.execute_query_async(user_text)

            # Persist updated history (everything after system message)
            entry.history = agent.conversation_history[1:]

            # Publish artifact with the response content
            artifact = Artifact(
                artifact_id=str(uuid.uuid4()),
                parts=[Part(root=TextPart(text=response.content or ""))],
            )
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    artifact=artifact,
                )
            )

            # Signal completion
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.completed),
                    final=True,
                )
            )

            logger.info(f"[A2A] Agent completed | task_id={task_id}")

        except asyncio.CancelledError:
            logger.info(f"[A2A] Agent canceled | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.canceled),
                    final=True,
                )
            )
            raise

        except Exception as e:
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.failed, message=None),
                    final=True,
                    metadata={"error": str(e)},
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id

        logger.info(f"[A2A] Cancel requested | task_id={task_id}")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )
