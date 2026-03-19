"""Bridge between the a2a-sdk AgentExecutor and Obelix BaseAgent.

Thread safety: each A2A context_id gets its own BaseAgent instance
(created via the agent_factory callable). Conversation history is
persisted per context_id so multi-turn conversations work correctly.
Concurrent requests on *different* contexts run in parallel without
interference. Concurrent requests on the *same* context are serialized
via a per-context asyncio.Event (idle gate).

Input-required flow (deferred tools): when the agent encounters a
tool with is_deferred=True that returns None, the loop stops and
yields a StreamEvent with deferred_tool_calls. The executor emits
TaskState.input_required with DataPart. On the next message/send for
the same contextId, the executor extracts the DataPart response,
injects it as a ToolMessage, and restarts the agent loop.
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
    DataPart,
    Message,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

from obelix.adapters.inbound.a2a.part_converter import (
    a2a_parts_to_obelix,
    deferred_calls_to_a2a_parts,
    obelix_response_to_a2a_parts,
)
from obelix.core.agent.exceptions import TaskRejectedError
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from obelix.core.tracer.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext

    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.model import StandardMessage
    from obelix.core.model.tool_message import ToolCall

logger = get_logger(__name__)

DEFAULT_MAX_CONTEXTS = 1024


def _agent_message(text: str) -> Message:
    """Build an A2A Message with agent role."""
    return Message(
        role=Role.agent,
        parts=[Part(root=TextPart(text=text))],
        message_id=str(uuid.uuid4()),
    )


class _ContextEntry:
    """Holds the state for a single conversation context."""

    __slots__ = (
        "history",
        "idle",
        "deferred_tool_calls",
        "deferred_tools",
        "trace_session",
        "trace_span",
    )

    def __init__(self) -> None:
        self.history: list[StandardMessage] = []
        self.idle = asyncio.Event()
        self.idle.set()  # starts as idle (ready for new executions)
        self.deferred_tool_calls: list[ToolCall] | None = None
        self.deferred_tools: list | None = None  # tool snapshot for OutputSchema lookup
        self.trace_session = None  # TraceSession saved when loop stops for deferred
        self.trace_span = None  # Current span saved when loop stops for deferred


class ObelixAgentExecutor(AgentExecutor):
    """Executes an Obelix BaseAgent in response to A2A requests.

    Bridges the a2a-sdk execution model (RequestContext + EventQueue)
    to the Obelix agent model (execute_query_stream).

    Each request creates a fresh BaseAgent via the factory callable.
    Conversation history is stored per context_id and injected into
    the fresh agent before execution, then saved back afterwards.

    Supports the input-required A2A flow via deferred tools: when the
    agent yields deferred_tool_calls, the executor emits input-required.
    The next request on the same contextId injects the response as a
    ToolMessage and restarts the agent.
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
            evicted_id, _evicted = self._contexts.popitem(last=False)
            logger.debug(f"[A2A] Evicted context | context_id={evicted_id}")

        entry = _ContextEntry()
        self._contexts[context_id] = entry
        return entry

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        # Extract content from the incoming message (multi-part)
        message = context.message
        user_text, attachments = a2a_parts_to_obelix(message.parts)

        if not user_text and not attachments:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_agent_message("No user input provided"),
                    ),
                    final=True,
                )
            )
            return

        # Get or create the context entry (LRU eviction under global lock)
        async with self._contexts_lock:
            entry = self._get_or_create_context(context_id)

        # Serialize requests on the same context
        await entry.idle.wait()
        entry.idle.clear()

        try:
            # RESUME PATH: if we have deferred tool calls, inject response
            if entry.deferred_tool_calls:
                self._inject_deferred_response(entry, message)

            await self._run_agent(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                attachments=attachments,
                entry=entry,
                event_queue=event_queue,
            )
        finally:
            entry.idle.set()

    def _inject_deferred_response(
        self,
        entry: _ContextEntry,
        message: Message,
    ) -> None:
        """Inject the client's response as ToolMessage for deferred tools.

        Extracts structured data from the DataPart in the message and
        validates it against the tool's OutputSchema if present.
        """
        # Extract structured data from DataPart
        structured_data: dict | None = None
        for part in message.parts:
            if isinstance(part.root, DataPart):
                structured_data = part.root.data
                break

        tool_results = []
        for call in entry.deferred_tool_calls:
            result_data = _parse_deferred_result(
                call, entry.deferred_tools, structured_data or {}
            )
            tool_results.append(
                ToolResult(
                    tool_name=call.name,
                    tool_call_id=call.id,
                    result=result_data,
                    status=ToolStatus.SUCCESS,
                )
            )
        entry.history.append(ToolMessage(tool_results=tool_results))
        entry.deferred_tool_calls = None
        entry.deferred_tools = None
        logger.info(
            f"[A2A] Injected deferred response | tool_count={len(tool_results)}"
        )

    async def _run_agent(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        attachments: list,
        entry: _ContextEntry,
        event_queue: EventQueue,
    ) -> None:
        """Run the agent with isolated context and persist history."""
        is_resume = bool(entry.history and isinstance(entry.history[-1], ToolMessage))

        logger.info(
            f"[A2A] Executing agent | task_id={task_id} context_id={context_id} "
            f"text_len={len(user_text)} history_len={len(entry.history)} "
            f"is_resume={is_resume}"
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
            # For resume: restore the trace context from the first invocation
            # so the resume appears under the same trace in the tracer UI.
            if is_resume:
                if entry.trace_session:
                    set_current_trace(entry.trace_session)
                    set_current_span(entry.trace_span)
                    entry.trace_session = None
                    entry.trace_span = None
                stream = agent.resume_after_deferred()
            elif attachments:
                # Pass as HumanMessage with attachments for multimodal
                query = HumanMessage(content=user_text, attachments=attachments)
                stream = agent.execute_query_stream(query)
            else:
                stream = agent.execute_query_stream(user_text)

            artifact_id = str(uuid.uuid4())
            first_chunk = True

            async for event in stream:
                # === Deferred tool detected: emit input-required ===
                if event.deferred_tool_calls:
                    entry.history = agent.conversation_history[1:]
                    entry.deferred_tool_calls = event.deferred_tool_calls
                    entry.deferred_tools = list(agent.registered_tools)
                    # Save trace context so the resume continues the same trace
                    entry.trace_session = get_current_trace()
                    entry.trace_span = get_current_span()

                    # Build DataPart message from deferred tool calls
                    deferred_parts = deferred_calls_to_a2a_parts(
                        event.deferred_tool_calls
                    )

                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.input_required,
                                message=Message(
                                    role=Role.agent,
                                    parts=deferred_parts,
                                    message_id=str(uuid.uuid4()),
                                ),
                            ),
                            final=True,
                        )
                    )
                    logger.info(
                        f"[A2A] Input required | task_id={task_id} "
                        f"context_id={context_id} "
                        f"deferred_count={len(event.deferred_tool_calls)}"
                    )
                    return

                # === Streaming token ===
                if event.token:
                    await event_queue.enqueue_event(
                        TaskArtifactUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            artifact=Artifact(
                                artifact_id=artifact_id,
                                parts=[Part(root=TextPart(text=event.token))],
                            ),
                            append=not first_chunk,
                            last_chunk=False,
                        )
                    )
                    first_chunk = False

                # === Final response ===
                if event.is_final and not event.deferred_tool_calls:
                    response = event.assistant_response
                    entry.history = agent.conversation_history[1:]

                    if first_chunk:
                        # Non-streaming: emit full response with multi-part
                        parts = (
                            obelix_response_to_a2a_parts(response)
                            if response
                            else [Part(root=TextPart(text=""))]
                        )
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=artifact_id,
                                    parts=parts,
                                ),
                                append=False,
                                last_chunk=True,
                            )
                        )
                    else:
                        # Streaming: text already sent, emit DataParts for tool results
                        final_parts = []
                        if response and response.tool_results:
                            for result in response.tool_results:
                                if isinstance(result.result, dict):
                                    final_parts.append(
                                        Part(
                                            root=DataPart(
                                                data=result.result,
                                                metadata={
                                                    "type": "tool_result",
                                                    "tool_name": result.tool_name,
                                                },
                                            )
                                        )
                                    )
                        if not final_parts:
                            final_parts = [Part(root=TextPart(text=""))]
                        await event_queue.enqueue_event(
                            TaskArtifactUpdateEvent(
                                task_id=task_id,
                                context_id=context_id,
                                artifact=Artifact(
                                    artifact_id=artifact_id,
                                    parts=final_parts,
                                ),
                                append=True,
                                last_chunk=True,
                            )
                        )

                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(state=TaskState.completed),
                            final=True,
                        )
                    )
                    logger.info(f"[A2A] Agent completed | task_id={task_id}")
                    break

        except asyncio.CancelledError:
            logger.info(f"[A2A] Agent canceled | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=_agent_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            raise

        except TaskRejectedError as e:
            logger.info(
                f"[A2A] Agent rejected task | task_id={task_id} reason={e.reason}"
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.rejected,
                        message=_agent_message(e.reason),
                    ),
                    final=True,
                )
            )

        except Exception as e:
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=_agent_message(f"Execution failed: {e}"),
                    ),
                    final=True,
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
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=_agent_message("Task canceled by client"),
                ),
                final=True,
            )
        )


# ── Module-level helpers for deferred tool protocol ─────────────────


def _parse_deferred_result(
    call: ToolCall,
    tools: list | None,
    data: dict,
) -> dict:
    """Parse a structured DataPart response for a deferred tool.

    If the tool has an OutputSchema, validates the data against it.
    Otherwise returns the data dict directly.
    """
    output_schema = None
    for tool in tools or []:
        if getattr(tool, "tool_name", None) == call.name:
            output_schema = getattr(tool, "_output_schema", None)
            break

    if output_schema:
        try:
            validated = output_schema(**data)
            return validated.model_dump()
        except Exception:
            return data

    return data
