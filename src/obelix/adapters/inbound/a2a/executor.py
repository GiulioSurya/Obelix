"""Bridge between the a2a-sdk AgentExecutor and Obelix BaseAgent."""

from __future__ import annotations

import uuid
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

logger = get_logger(__name__)


class ObelixAgentExecutor(AgentExecutor):
    """Executes an Obelix BaseAgent in response to A2A requests.

    Bridges the a2a-sdk execution model (RequestContext + EventQueue)
    to the Obelix agent model (execute_query_async).
    """

    def __init__(self, agent: BaseAgent) -> None:
        self._agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id

        # Extract user text from the incoming message
        user_text = context.get_user_input()
        if not user_text:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=None,
                    ),
                    final=True,
                )
            )
            return

        logger.info(
            f"[A2A] Executing agent | task_id={task_id} text_len={len(user_text)}"
        )

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
            response = await self._agent.execute_query_async(user_text)

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

        except Exception as e:
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=None,
                    ),
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
