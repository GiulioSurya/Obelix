"""Bridge between the a2a-sdk AgentExecutor and Obelix BaseAgent.

Thread safety: each A2A context_id gets its own BaseAgent instance
(created via the agent_factory callable). Conversation history is
persisted per context_id so multi-turn conversations work correctly.
Concurrent requests on *different* contexts run in parallel without
interference. Concurrent requests on the *same* context are serialized
via a per-context asyncio.Event (idle gate).

Input-required flow: when the agent invokes RequestUserInputTool,
the tool suspends on an asyncio.Future. The executor detects this
via InputChannel.wait_for_request(), emits TaskState.input_required,
and returns. The next message/send on the same contextId delivers
the client's response to the suspended tool, which resumes the
agent loop normally.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from obelix.adapters.inbound.a2a.input_channel import InputChannel, input_channel_var
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext

    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import StandardMessage

logger = get_logger(__name__)

DEFAULT_MAX_CONTEXTS = 1024


def _agent_message(text: str) -> Message:
    """Build an A2A Message with agent role."""
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid.uuid4()),
    )


# Keep the old name as alias for backward compat in tests
_error_message = _agent_message


class _RequestRef:
    """Mutable reference to the current A2A request state.

    Allows swapping the target queue, task_id, and context_id
    mid-stream so that events emitted after an input-required
    resume go to the correct (second) A2A request.
    """

    __slots__ = ("queue", "task_id", "context_id")

    def __init__(self, queue: EventQueue, task_id: str, context_id: str) -> None:
        self.queue = queue
        self.task_id = task_id
        self.context_id = context_id

    async def enqueue_event(self, event: Any) -> None:
        await self.queue.enqueue_event(event)


class _ContextEntry:
    """Holds the state for a single conversation context."""

    __slots__ = (
        "history",
        "idle",
        "input_channel",
        "pending_task",
        "pending_agent",
        "pending_ref",
    )

    def __init__(self) -> None:
        self.history: list[StandardMessage] = []
        self.idle = asyncio.Event()
        self.idle.set()  # starts as idle (ready for new executions)
        self.input_channel: InputChannel | None = None
        self.pending_task: asyncio.Task | None = None
        self.pending_agent: BaseAgent | None = None
        self.pending_ref: _RequestRef | None = None


class ObelixAgentExecutor(AgentExecutor):
    """Executes an Obelix BaseAgent in response to A2A requests.

    Bridges the a2a-sdk execution model (RequestContext + EventQueue)
    to the Obelix agent model (execute_query_stream).

    Each request creates a fresh BaseAgent via the factory callable.
    Conversation history is stored per context_id and injected into
    the fresh agent before execution, then saved back afterwards.

    Supports the input-required A2A flow: when the agent calls
    RequestUserInputTool, execution suspends until the client
    sends a follow-up message on the same contextId.
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
            evicted_id, evicted = self._contexts.popitem(last=False)
            # Cancel any suspended execution on evicted context
            if evicted.pending_task and not evicted.pending_task.done():
                evicted.pending_task.cancel()
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
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_error_message("No user input provided"),
                    ),
                    final=True,
                )
            )
            return

        # Get or create the context entry (LRU eviction under global lock)
        async with self._contexts_lock:
            entry = self._get_or_create_context(context_id)

        # ── RESUME PATH: deliver response to suspended tool ──
        logger.debug(
            f"[A2A] Checking resume path | context_id={context_id} "
            f"has_channel={entry.input_channel is not None} "
            f"is_waiting={entry.input_channel.is_waiting() if entry.input_channel else 'N/A'} "
            f"idle={entry.idle.is_set()}"
        )
        if entry.input_channel and entry.input_channel.is_waiting():
            await self._resume_with_input(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                entry=entry,
                event_queue=event_queue,
            )
            return

        # ── NORMAL PATH: new execution ──
        await entry.idle.wait()
        entry.idle.clear()

        try:
            await self._execute_with_context(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                entry=entry,
                event_queue=event_queue,
            )
        finally:
            # Only release idle if we're NOT suspended for input
            if not (entry.input_channel and entry.input_channel.is_waiting()):
                self._cleanup_context(entry)

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

        # Set up InputChannel via ContextVar
        channel = InputChannel()
        entry.input_channel = channel
        input_channel_var.set(channel)

        # Mutable request ref (allows swapping queue/task_id/context_id for resume)
        req_ref = _RequestRef(event_queue, task_id, context_id)
        entry.pending_ref = req_ref

        # Signal that the agent is working
        await req_ref.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        try:
            # Run the agent stream as a task so we can multiplex
            # with the input-required detection.
            # _consume_stream reads task_id/context_id from req_ref
            # so they stay current after a resume swaps them.
            stream_task = asyncio.create_task(
                self._consume_stream(
                    agent=agent,
                    user_text=user_text,
                    entry=entry,
                    req_ref=req_ref,
                )
            )
            entry.pending_task = stream_task
            entry.pending_agent = agent

            # Wait for either: stream completes OR tool needs input
            input_wait = asyncio.create_task(channel.wait_for_request())

            done, pending = await asyncio.wait(
                {stream_task, input_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if stream_task in done:
                # Normal completion — stream already emitted all events
                input_wait.cancel()
                # Propagate exceptions from the stream task
                stream_task.result()

            elif input_wait in done:
                # Tool needs input — emit input-required and return
                question = input_wait.result()
                logger.info(
                    f"[A2A] Input required | task_id={task_id} "
                    f"context_id={context_id} question_len={len(question)}"
                )

                await req_ref.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=_agent_message(question),
                        ),
                        final=True,
                    )
                )
                # Don't clean up — the pending_task and channel stay alive
                # for the resume path. idle stays cleared.
                return

        except asyncio.CancelledError:
            logger.info(f"[A2A] Agent canceled | task_id={task_id}")
            await req_ref.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=_error_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            raise

        except Exception as e:
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await req_ref.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_error_message(f"Execution failed: {e}"),
                    ),
                    final=True,
                )
            )

        finally:
            input_channel_var.set(None)

    async def _consume_stream(
        self,
        *,
        agent: BaseAgent,
        user_text: str,
        entry: _ContextEntry,
        req_ref: _RequestRef,
    ) -> None:
        """Consume the agent's streaming output and emit A2A events.

        This runs as an asyncio.Task so the executor can multiplex
        between the stream and the input-required signal.

        IMPORTANT: reads task_id/context_id from req_ref (not parameters)
        so that after an input-required resume, events use the new
        request's task_id.
        """
        artifact_id = str(uuid.uuid4())
        first_chunk = True

        async for event in agent.execute_query_stream(user_text):
            if event.token:
                await req_ref.enqueue_event(
                    TaskArtifactUpdateEvent(
                        task_id=req_ref.task_id,
                        context_id=req_ref.context_id,
                        artifact=Artifact(
                            artifact_id=artifact_id,
                            parts=[Part(root=TextPart(text=event.token))],
                        ),
                        append=not first_chunk,
                        last_chunk=False,
                    )
                )
                first_chunk = False

            if event.is_final:
                response = event.assistant_response

                # Persist updated history (everything after system message)
                entry.history = agent.conversation_history[1:]

                if first_chunk:
                    # Non-streaming fallback: single complete artifact
                    await req_ref.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=req_ref.task_id,
                            context_id=req_ref.context_id,
                            artifact=Artifact(
                                artifact_id=artifact_id,
                                parts=[
                                    Part(root=TextPart(text=response.content or ""))
                                ],
                            ),
                            append=False,
                            last_chunk=True,
                        )
                    )
                else:
                    # Close the streamed artifact
                    await req_ref.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=req_ref.task_id,
                            context_id=req_ref.context_id,
                            artifact=Artifact(
                                artifact_id=artifact_id,
                                parts=[Part(root=TextPart(text=""))],
                            ),
                            append=True,
                            last_chunk=True,
                        )
                    )

                # Signal completion
                await req_ref.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=req_ref.task_id,
                        context_id=req_ref.context_id,
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                    )
                )

                logger.info(f"[A2A] Agent completed | task_id={req_ref.task_id}")
                break

    async def _resume_with_input(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        entry: _ContextEntry,
        event_queue: EventQueue,
    ) -> None:
        """Handle the response to an input-required state.

        Delivers the client's response to the suspended tool,
        waits for the agent to complete, and emits the result
        events to the new event queue.
        """
        logger.info(
            f"[A2A] Resuming with input | task_id={task_id} "
            f"context_id={context_id} input_len={len(user_text)}"
        )

        # Swap the request ref so post-resume events use the new
        # request's queue, task_id, and context_id
        if entry.pending_ref:
            entry.pending_ref.queue = event_queue
            entry.pending_ref.task_id = task_id
            entry.pending_ref.context_id = context_id

        # Signal working on the new event queue
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.working),
                final=False,
            )
        )

        # Deliver the response — this resolves the Future and resumes the tool
        entry.input_channel.provide_input(user_text)

        # Wait for the agent stream to complete
        try:
            await entry.pending_task

        except asyncio.CancelledError:
            logger.info(f"[A2A] Agent canceled during resume | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=_error_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            raise

        except TimeoutError:
            logger.error(f"[A2A] Input timeout | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_error_message(
                            "Input timeout — client did not respond in time"
                        ),
                    ),
                    final=True,
                )
            )

        except Exception as e:
            logger.error(
                f"[A2A] Agent failed during resume | task_id={task_id} error={e}"
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_error_message(f"Execution failed: {e}"),
                    ),
                    final=True,
                )
            )

        finally:
            self._cleanup_context(entry)

    def _cleanup_context(self, entry: _ContextEntry) -> None:
        """Reset context state after execution completes."""
        entry.input_channel = None
        entry.pending_task = None
        entry.pending_agent = None
        entry.pending_ref = None
        entry.idle.set()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id

        logger.info(f"[A2A] Cancel requested | task_id={task_id}")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=_error_message("Task canceled by client"),
                ),
                final=True,
            )
        )
