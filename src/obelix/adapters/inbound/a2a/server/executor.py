"""ObelixAgentExecutor: bridges the a2a-sdk AgentExecutor to Obelix BaseAgent.

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
from obelix.adapters.inbound.a2a.server.context import ContextStore
from obelix.adapters.inbound.a2a.server.deferred import inject_deferred_response
from obelix.adapters.inbound.a2a.server.helpers import (
    DEFAULT_MAX_CONTEXTS,
    agent_message,
)
from obelix.core.agent.exceptions import TaskRejectedError
from obelix.core.model.assistant_message import AssistantResponse
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.tool_message import ToolMessage, ToolResult, ToolStatus
from obelix.core.tracer.context import (
    get_current_span,
    get_current_trace,
    set_current_span,
    set_current_trace,
)
from obelix.core.tracer.models import SpanType
from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from a2a.server.agent_execution.context import RequestContext

    from obelix.core.agent.base_agent import BaseAgent
    from obelix.core.tracer.tracer import Tracer

logger = get_logger(__name__)


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
        tracer: Tracer | None = None,
    ) -> None:
        self._agent_factory = agent_factory
        self._store = ContextStore(max_contexts)
        self._store_lock = asyncio.Lock()
        self._tracer = tracer

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        # Extract content from the incoming message (multi-part)
        message = context.message
        user_text, attachments = a2a_parts_to_obelix(message.parts)

        # Extract client metadata (shell environment, etc.) if present.
        # The client sends this via Message.metadata on the first message.
        # Stored in the ContextEntry so it persists across the conversation.
        if message.metadata and "client_info" in message.metadata:
            async with self._store_lock:
                entry = self._store.get_or_create(context_id)
            entry.client_info = message.metadata["client_info"]

        if not user_text and not attachments:
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=agent_message("No user input provided"),
                    ),
                    final=True,
                )
            )
            return

        # Get or create the context entry (LRU eviction under global lock)
        async with self._store_lock:
            entry = self._store.get_or_create(context_id)

        # Serialize requests on the same context
        await entry.idle.wait()
        entry.idle.clear()

        try:
            # RESUME PATH: if we have deferred tool calls, inject response
            is_resume = bool(entry.deferred_tool_calls)
            if is_resume:
                inject_deferred_response(entry, message)

            await self._run_agent(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                attachments=attachments,
                entry=entry,
                event_queue=event_queue,
                is_resume=is_resume,
            )
        finally:
            entry.active_agent = None
            entry.idle.set()

    async def _run_agent(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        attachments: list,
        entry,
        event_queue: EventQueue,
        is_resume: bool = False,
    ) -> None:
        """Run the agent with isolated context and persist history.

        When a tracer is configured, opens an ``a2a_task`` span as the trace
        root before delegating to the agent. The agent's own span becomes a
        child of it. On deferred-tool suspension the trace and ``a2a_task``
        span are left open so ``resume_after_deferred`` can continue inside
        them; they are closed on any terminal outcome (completed / failed /
        rejected / canceled) or on the resume invocation that finishes them.
        """

        tracer = self._tracer
        # Open a2a_task root span on first invocation; on resume we reuse the
        # trace + a2a_task span that are already restored by ``_run_agent_impl``
        # via ``set_current_trace`` / ``set_current_span``.
        a2a_task_span = None
        trace_opened_here = False
        if tracer and not is_resume:
            await tracer.start_trace(
                name="a2a.task",
                metadata={"task_id": task_id, "context_id": context_id},
            )
            a2a_task_span = await tracer.start_span(
                SpanType.a2a_task,
                name=f"task {task_id[:8] if task_id else 'unknown'}",
                input={"context_id": context_id},
                metadata={"task_id": task_id, "context_id": context_id},
            )
            trace_opened_here = True

        # Tracks whether the executor suspended for a deferred tool. When
        # True, the finally block leaves the trace + a2a_task span open so
        # ``resume_after_deferred`` can continue inside them.
        deferred_suspended = False

        try:
            deferred_suspended = await self._run_agent_impl(
                task_id=task_id,
                context_id=context_id,
                user_text=user_text,
                attachments=attachments,
                entry=entry,
                event_queue=event_queue,
                is_resume=is_resume,
            )
        finally:
            if tracer and not deferred_suspended and (trace_opened_here or is_resume):
                # Re-pin the a2a_task span as current before closing. On the
                # initial invocation this is defensive (agent's generator
                # finally should have already restored it); on resume we
                # need to walk the trace to find it.
                task_span = a2a_task_span
                if task_span is None:
                    trace = get_current_trace()
                    if trace is not None:
                        task_span = next(
                            (
                                s
                                for s in trace.spans
                                if s.span_type == SpanType.a2a_task
                                and s.end_time is None
                            ),
                            None,
                        )
                if task_span is not None:
                    set_current_span(task_span)
                await tracer.end_span()
                await tracer.end_trace()

    async def _run_agent_impl(
        self,
        *,
        task_id: str,
        context_id: str,
        user_text: str,
        attachments: list,
        entry,
        event_queue: EventQueue,
        is_resume: bool = False,
    ) -> bool:
        """Inner agent runner. Returns True if suspended for a deferred tool."""

        logger.info(
            f"[A2A] Executing agent | task_id={task_id} context_id={context_id} "
            f"text_len={len(user_text)} history_len={len(entry.history)} "
            f"is_resume={is_resume}"
        )

        # Create a fresh agent for this request
        agent = self._agent_factory()
        entry.active_agent = agent

        # Inject client shell info into BashTool's ClientShellExecutor
        if entry.client_info:
            self._inject_client_info(agent, entry.client_info)

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

        stream = None
        # Captured on the success path to emit an ``assistant`` span as a
        # sibling of ``agent`` under ``a2a_task``. Stays ``None`` on
        # cancel / reject / failure / deferred suspension so those paths
        # do NOT emit an assistant span. Initialized BEFORE the try block so
        # that if any call below (e.g. tracer ``start_span`` on the human
        # span) raises, the ``except`` / ``finally`` / post-block
        # ``final_response is not None`` check does not hit
        # ``UnboundLocalError``.
        final_response: AssistantResponse | None = None
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

            # Emit human span as direct child of a2a_task on initial
            # invocation only. On resume the input is a DataPart (deferred
            # tool response) conceptually continuing the same turn, not a
            # new user query, so we skip emitting a second human span.
            if self._tracer and not is_resume:
                await self._tracer.start_span(
                    SpanType.human,
                    "human.input",
                    input=user_text,
                )
                await self._tracer.end_span(output=user_text)

            artifact_id = str(uuid.uuid4())
            first_chunk = True

            async for event in stream:
                # === Agent canceled by user ===
                if event.canceled:
                    entry.history = agent.conversation_history[1:]
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.canceled,
                                message=agent_message("Task canceled by client"),
                            ),
                            final=True,
                        )
                    )
                    logger.info(f"[A2A] Agent canceled by user | task_id={task_id}")
                    return False

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
                    return True

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
                    final_response = response
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

                    # Place the agent response in a working status so
                    # the SDK TaskManager appends it to task.history.
                    history_parts = (
                        obelix_response_to_a2a_parts(response)
                        if response
                        else [Part(root=TextPart(text=""))]
                    )
                    await event_queue.enqueue_event(
                        TaskStatusUpdateEvent(
                            task_id=task_id,
                            context_id=context_id,
                            status=TaskStatus(
                                state=TaskState.working,
                                message=Message(
                                    role=Role.agent,
                                    parts=history_parts,
                                    message_id=str(uuid.uuid4()),
                                ),
                            ),
                            final=False,
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
            entry.history = agent.conversation_history[1:]
            logger.info(f"[A2A] Agent canceled | task_id={task_id}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=agent_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
            raise

        except TaskRejectedError as e:
            entry.history = agent.conversation_history[1:]
            logger.info(
                f"[A2A] Agent rejected task | task_id={task_id} reason={e.reason}"
            )
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.rejected,
                        message=agent_message(e.reason),
                    ),
                    final=True,
                )
            )

        except Exception as e:
            entry.history = agent.conversation_history[1:]
            logger.error(f"[A2A] Agent failed | task_id={task_id} error={e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=agent_message(f"Execution failed: {e}"),
                    ),
                    final=True,
                )
            )

        finally:
            # Always close the async generator to trigger cleanup
            # (tracer span closing, resource release) in base_agent's
            # _execute_loop finally block.
            if stream is not None:
                aclose = getattr(stream, "aclose", None)
                if aclose and callable(aclose):
                    try:
                        await aclose()
                    except TypeError:
                        pass  # not a real async generator (e.g. mock)

        # Emit assistant span on the success path only. ``final_response`` is
        # only set when the stream produced a terminal ``is_final`` event
        # without deferred tool calls, so rejection / cancellation / failure /
        # deferred suspension naturally skip this. We emit AFTER the stream's
        # ``aclose()`` so the BaseAgent ``agent`` span has already ended —
        # that way the ``assistant`` span becomes a direct child of
        # ``a2a_task`` (sibling of ``agent``), per spec §6.
        if self._tracer and final_response is not None:
            content = getattr(final_response, "content", None)
            await self._tracer.start_span(
                SpanType.assistant,
                "assistant.response",
                input={"has_tool_calls": False},
            )
            await self._tracer.end_span(output={"content": content})

        # Normal termination (completed / rejected / failed): not suspended
        # for deferred input, so the caller should close the trace.
        return False

    @staticmethod
    def _inject_client_info(agent: BaseAgent, client_info: dict) -> None:
        """Inject client shell environment into the agent's system message.

        Finds any BashTool with a ClientShellExecutor among the agent's
        registered tools, populates its shell_info, and appends the
        system_prompt_fragment to the agent's system message.
        """
        from obelix.adapters.outbound.shell.client_executor import ClientShellExecutor

        for tool in agent.registered_tools:
            executor = getattr(tool, "_executor", None)
            if isinstance(executor, ClientShellExecutor):
                if not executor.shell_info:
                    executor.set_shell_info(client_info)
                fragment = tool.system_prompt_fragment()
                if fragment and fragment not in agent.system_message.content:
                    agent.system_message.content += fragment
                    logger.info("[A2A] Injected client shell info into system message")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id or "default"

        logger.info(f"[A2A] Cancel requested | task_id={task_id}")

        # Try to find the active agent for this context and signal cancel
        async with self._store_lock:
            entry = self._store.get_or_create(context_id)

        if entry.active_agent:
            # Agent is running — signal cooperative cancellation.
            # The _execute_loop will detect the flag, yield a canceled
            # StreamEvent, and _run_agent will emit TaskState.canceled.
            entry.active_agent.cancel()
            logger.info(
                f"[A2A] Cancel signal sent to active agent | "
                f"task_id={task_id} context_id={context_id}"
            )
        else:
            # No active agent — task was likely in input_required (deferred).
            # Clean up: replace the null ToolMessage with a cancel message
            # so the LLM knows the tool was not executed.
            if entry.deferred_tool_calls and entry.history:
                # Deferred tools have NO ToolMessage in the history yet —
                # only the AssistantMessage with tool_calls was saved.
                # Inject a ToolMessage with cancel results so the LLM
                # sees a proper tool_use → tool_result sequence.
                cancel_results = [
                    ToolResult(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        result="Execution canceled by user",
                        status=ToolStatus.ERROR,
                    )
                    for tc in entry.deferred_tool_calls
                ]
                entry.history.append(ToolMessage(tool_results=cancel_results))
                entry.deferred_tool_calls = None
                entry.deferred_tools = None
                entry.trace_session = None
                entry.trace_span = None

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.canceled,
                        message=agent_message("Task canceled by client"),
                    ),
                    final=True,
                )
            )
