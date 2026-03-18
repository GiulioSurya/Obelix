# src/obelix/core/agent/base_agent.py
import asyncio
import inspect
import time
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from obelix.infrastructure.logging import get_logger

if TYPE_CHECKING:
    from obelix.core.agent.shared_memory import SharedMemoryGraph
    from obelix.core.tracer.tracer import Tracer

from obelix.core.agent.agent_tracing import (
    emit_assistant_span,
    emit_human_span,
    end_agent_trace,
    end_llm_span,
    end_tool_span,
    start_agent_trace,
    start_llm_span,
    start_tool_span,
)
from obelix.core.agent.event_contracts import EventContract, get_event_contracts
from obelix.core.agent.hooks import AgentEvent, AgentStatus, Hook, HookDecision, Outcome
from obelix.core.agent.memory_hooks import register_memory_hooks
from obelix.core.model.assistant_message import (
    AssistantMessage,
    AssistantResponse,
    StreamEvent,
)
from obelix.core.model.human_message import HumanMessage
from obelix.core.model.standard_message import StandardMessage
from obelix.core.model.system_message import SystemMessage
from obelix.core.model.tool_message import (
    ToolCall,
    ToolMessage,
    ToolRequirement,
    ToolResult,
    ToolStatus,
)
from obelix.core.model.usage import AgentUsage
from obelix.core.tool.tool_base import Tool
from obelix.ports.outbound.llm_provider import AbstractLLMProvider

logger = get_logger(__name__)


class BaseAgent:
    def __init__(
        self,
        system_message: str,
        provider: AbstractLLMProvider,
        max_iterations: int = 15,
        tools: Tool | type[Tool] | list[type[Tool] | Tool] | None = None,
        tool_policy: list[ToolRequirement] | None = None,
        exit_on_success: list[str] | None = None,
        response_schema: type[BaseModel] | None = None,
        tracer: "Tracer | None" = None,
        planning: bool = False,
    ):
        self.system_message = SystemMessage(content=system_message)
        self.max_iterations = max_iterations
        self._exit_on_success: set[str] = (
            set(exit_on_success) if exit_on_success else set()
        )
        self.response_schema = response_schema
        self._planning = planning

        self.provider = provider

        self.registered_tools: list[Tool] = []
        self.conversation_history: list[StandardMessage] = [self.system_message]
        self.agent_usage = AgentUsage(model_id=self.provider.model_id)
        self._tool_policy = tool_policy or []

        # Hook system
        self._hooks: dict[AgentEvent, list[Hook]] = {event: [] for event in AgentEvent}
        self._event_contracts: dict[AgentEvent, EventContract] = get_event_contracts()

        # Register tools from constructor parameter
        if tools:
            if not isinstance(tools, list):
                tools = [tools]
            for tool in tools:
                tool_instance = tool() if inspect.isclass(tool) else tool
                self.register_tool(tool_instance)

        if self._tool_policy:
            from obelix.core.agent.tool_policy_hook import ToolPolicyHook

            ToolPolicyHook(self._tool_policy).register(self)

        # Planning protocol: append instructions to system message
        if self._planning:
            from obelix.core.agent.planning import get_planning_instruction

            self.system_message.content += get_planning_instruction()
            logger.info(
                f"[{self.__class__.__name__}] Planning mode enabled. "
                "For best results, set reasoning_effort='high' (or thinking_mode=True "
                "for Anthropic) on the provider."
            )

        # Tracer (optional, zero overhead when None)
        self._tracer: Tracer | None = tracer

        # Shared memory (set by factory or manually, None if not used)
        self.memory_graph: SharedMemoryGraph | None = None
        self.agent_id: str | None = None
        register_memory_hooks(self)

    def register_tool(self, tool: Tool):
        """
        Registers a tool for the agent
        """
        try:
            from obelix.plugins.mcp.mcp_tool import MCPTool

            if isinstance(tool, MCPTool):
                if not tool.manager.is_connected():
                    logger.error(
                        f"[{self.__class__.__name__}] Tool registration failed — MCP manager not connected | tool={tool.tool_name}"
                    )
                    raise RuntimeError(
                        f"MCP Tool {tool.tool_name} is not connected. "
                        "The manager must be connected before registration."
                    )
        except ImportError:
            logger.debug(
                "[BaseAgent] MCP plugin unavailable — skipping MCP connectivity check"
            )

        if tool not in self.registered_tools:
            self.registered_tools.append(tool)
            tool_name = getattr(tool, "tool_name", None) or tool.__class__.__name__
            logger.info(
                f"[{self.__class__.__name__}] Tool registered | tool={tool_name}"
            )

            # Auto-inject system prompt fragment if the tool provides one
            fragment_fn = getattr(tool, "system_prompt_fragment", None)
            if callable(fragment_fn):
                fragment = fragment_fn()
                if fragment:
                    self.system_message.content += fragment
                    logger.info(
                        f"[{self.__class__.__name__}] System prompt enriched by tool={tool_name}"
                    )

    def on(self, event: AgentEvent) -> Hook:
        """
        Fluent API for registering hooks.
        """
        hook = Hook(event)
        self._hooks[event].append(hook)
        return hook

    def _is_valid_value(self, value: Any, expected: Any) -> bool:
        if expected is None:
            return value is None
        if isinstance(expected, tuple):
            return isinstance(value, expected)
        return isinstance(value, expected)

    def _retryable_events(self) -> list[AgentEvent]:
        return [evt for evt, c in self._event_contracts.items() if c.retryable]

    async def _run_hooks(
        self, event: AgentEvent, current_value: Any = None, **ctx_kwargs
    ) -> Outcome:
        """
        Executes all hooks registered for an event.
        """
        agent_status = AgentStatus(event=event, agent=self, **ctx_kwargs)

        contract = self._event_contracts[event]
        if not self._is_valid_value(current_value, contract.input_type):
            raise RuntimeError(
                f"Input value invalid for event {event.value}. "
                f"Expected: {contract.input_type}, got: {type(current_value)}"
            )

        result_value = current_value

        for hook in self._hooks[event]:
            outcome = await hook.execute(agent_status, result_value)

            if outcome.decision == HookDecision.RETRY:
                if not contract.retryable:
                    retryable = ", ".join(e.value for e in self._retryable_events())
                    raise RuntimeError(
                        f"RETRY not allowed for event {event.value}. "
                        f"Retryable events: {retryable}"
                    )
                return outcome

            if outcome.decision == HookDecision.STOP:
                if contract.stop_output is None:
                    raise RuntimeError(f"STOP not allowed for event {event.value}.")
                if outcome.value is None or not isinstance(
                    outcome.value, contract.stop_output
                ):
                    raise RuntimeError(
                        f"STOP requires value of type {contract.stop_output} for event {event.value}."
                    )
                return outcome

            if outcome.decision == HookDecision.FAIL:
                return outcome

            if not self._is_valid_value(outcome.value, contract.output_type):
                raise RuntimeError(
                    f"Value invalid for event {event.value}. "
                    f"Expected: {contract.output_type}, got: {type(outcome.value)}"
                )
            result_value = outcome.value

        return Outcome(HookDecision.CONTINUE, result_value)

    # ─── Public API ───────────────────────────────────────────────────────────

    async def execute_query_async(
        self, query: str | list[StandardMessage]
    ) -> AssistantResponse:
        """
        Execute query asynchronously (for FastAPI).
        """
        self._validate_query_input(query)
        async with aclosing(self._execute_loop(query, stream=False)) as stream:
            async for event in stream:
                if event.is_final:
                    return event.assistant_response
        raise RuntimeError("Execution loop ended without a final event")

    def execute_query(self, query: str | list[StandardMessage]) -> AssistantResponse:
        """
        Execute query synchronously (for CLI).
        """
        return asyncio.run(self.execute_query_async(query))

    async def execute_query_stream(
        self, query: str | list[StandardMessage]
    ) -> AsyncIterator[StreamEvent]:
        """
        Execute query with streaming. Yields StreamEvent chunks.

        Same semantics as execute_query_async() but text tokens are yielded
        in real-time as they arrive from the LLM. The final StreamEvent
        (is_final=True) carries the complete AssistantResponse.

        Intermediate iterations (tool calls) use streaming internally
        but do not yield tokens to the consumer.

        Requires the provider to implement invoke_stream().
        Raises NotImplementedError if the provider does not support streaming.

        Usage:
            async for event in agent.execute_query_stream("question"):
                if event.token:
                    print(event.token, end="", flush=True)
                if event.is_final:
                    response = event.assistant_response
        """
        self._validate_query_input(query)
        async for event in self._execute_loop(query, stream=True):
            yield event

    async def resume_after_deferred(self) -> AsyncIterator[StreamEvent]:
        """Resume the agent loop after a deferred tool response was injected.

        Call this after injecting the deferred tool's ToolMessage into
        conversation_history. The loop restarts from the current history
        without adding a new HumanMessage — the LLM sees the pending
        tool_call + tool_result and continues normally.
        """
        async for event in self._execute_loop(None, stream=True, resume=True):
            yield event

    # ─── Unified Execution Loop ───────────────────────────────────────────────

    async def _execute_loop(
        self,
        query: str | list[StandardMessage] | None,
        *,
        stream: bool = False,
        resume: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        """
        Unified execution loop. Always an async generator yielding StreamEvent.

        When stream=False, only yields the final StreamEvent (is_final=True).
        When stream=True, also yields intermediate token events.
        When resume=True, skips adding a HumanMessage — the conversation_history
        already contains everything needed (used after deferred tool response).
        """
        if resume:
            # Resume after deferred: trace and agent span are still open
            # from the first invocation. Don't create new ones.
            # We must close the root trace at the end.
            is_root_trace = True
        else:
            is_root_trace = await start_agent_trace(
                self._tracer,
                self.__class__.__name__,
                query if query is not None else "",
            )

        collected_tool_results: list[ToolResult] = []
        execution_error: str | None = None
        _trace_error: str | None = None
        _stopped_for_deferred = False
        self._invocation_start = len(self.conversation_history)

        if resume:
            # Resume after deferred: history already has everything
            pass
        elif isinstance(query, str):
            user_message = HumanMessage(content=query)
            self.conversation_history.append(user_message)
        elif isinstance(query, list):
            human_messages = [msg for msg in query if isinstance(msg, HumanMessage)]
            if len(human_messages) != 1:
                raise ValueError(
                    f"Message list must contain exactly 1 HumanMessage, "
                    f"found {len(human_messages)}"
                )
            self.conversation_history.extend(query)
        else:
            raise TypeError(
                f"query must be str or List[StandardMessage], "
                f"received {type(query).__name__}"
            )

        try:
            if not resume:
                query_text = (
                    query
                    if isinstance(query, str)
                    else str([m.content for m in query if isinstance(m, HumanMessage)])
                )
                await emit_human_span(self._tracer, query_text)

            for iteration in range(1, self.max_iterations + 1):
                # === BEFORE_LLM_CALL hooks ===
                outcome = await self._run_hooks(
                    AgentEvent.BEFORE_LLM_CALL, iteration=iteration
                )
                if outcome.decision == HookDecision.STOP:
                    assistant_msg = outcome.value
                    await emit_assistant_span(self._tracer, assistant_msg)
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._run_hooks(
                        AgentEvent.QUERY_END,
                        current_value=response,
                        iteration=iteration,
                    )
                    yield StreamEvent(
                        assistant_message=assistant_msg,
                        assistant_response=response,
                        is_final=True,
                    )
                    return
                if outcome.decision == HookDecision.FAIL:
                    raise RuntimeError("Hook BEFORE_LLM_CALL requested FAIL")

                # === LLM call ===
                await start_llm_span(
                    self._tracer,
                    self.provider,
                    self.conversation_history,
                    self.registered_tools,
                )

                assistant_msg: AssistantMessage | None = None
                streamed_tokens = False

                if stream:
                    try:
                        llm_stream = self.provider.invoke_stream(
                            self.conversation_history,
                            self.registered_tools,
                            self.response_schema,
                        )
                        async for event in llm_stream:
                            if event.is_final:
                                assistant_msg = event.assistant_message
                            elif event.token is not None:
                                streamed_tokens = True
                                yield event
                    except NotImplementedError:
                        logger.info(
                            f"[{self.__class__.__name__}] Provider does not support "
                            f"streaming, falling back to invoke()"
                        )
                        assistant_msg = await self.provider.invoke(
                            self.conversation_history,
                            self.registered_tools,
                            self.response_schema,
                        )
                        if assistant_msg.content:
                            yield StreamEvent(token=assistant_msg.content)
                            streamed_tokens = True
                else:
                    assistant_msg = await self.provider.invoke(
                        self.conversation_history,
                        self.registered_tools,
                        self.response_schema,
                    )

                if assistant_msg is None:
                    raise RuntimeError(
                        "invoke_stream() did not yield a final StreamEvent"
                    )

                await end_llm_span(self._tracer, assistant_msg)

                # === AFTER_LLM_CALL hooks ===
                outcome = await self._run_hooks(
                    AgentEvent.AFTER_LLM_CALL,
                    current_value=assistant_msg,
                    iteration=iteration,
                    assistant_message=assistant_msg,
                )
                if outcome.decision == HookDecision.RETRY:
                    if stream and streamed_tokens:
                        raise RuntimeError(
                            "Hook AFTER_LLM_CALL requested RETRY after tokens were "
                            "already streamed to the consumer. This is not supported "
                            "in streaming mode."
                        )
                    continue
                if outcome.decision == HookDecision.FAIL:
                    raise RuntimeError("Hook AFTER_LLM_CALL requested FAIL")
                if outcome.decision == HookDecision.STOP:
                    assistant_msg = outcome.value
                    await emit_assistant_span(self._tracer, assistant_msg)
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._run_hooks(
                        AgentEvent.QUERY_END,
                        current_value=response,
                        iteration=iteration,
                    )
                    yield StreamEvent(
                        assistant_message=assistant_msg,
                        assistant_response=response,
                        is_final=True,
                    )
                    return
                assistant_msg = outcome.value

                if assistant_msg.usage:
                    self.agent_usage.add_usage(assistant_msg.usage)

                # === Empty response ===
                if not assistant_msg.tool_calls and not assistant_msg.content:
                    logger.info(
                        f"[{self.__class__.__name__}] LLM returned empty response | iteration={iteration}"
                    )
                    self.conversation_history.append(assistant_msg)
                    outcome = await self._run_hooks(
                        AgentEvent.BEFORE_FINAL_RESPONSE,
                        current_value=assistant_msg,
                        iteration=iteration,
                        assistant_message=assistant_msg,
                    )
                    if outcome.decision == HookDecision.RETRY:
                        continue
                    if outcome.decision == HookDecision.FAIL:
                        raise RuntimeError(
                            "Required tool call missing before final response"
                        )
                    assistant_msg = outcome.value
                    await emit_assistant_span(self._tracer, assistant_msg)
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._run_hooks(
                        AgentEvent.QUERY_END,
                        current_value=response,
                        iteration=iteration,
                    )
                    yield StreamEvent(
                        assistant_message=assistant_msg,
                        assistant_response=response,
                        is_final=True,
                    )
                    return

                # === Tool calls ===
                if assistant_msg.tool_calls:
                    (
                        execution_error,
                        iteration_results,
                        deferred_calls,
                    ) = await self._process_tool_calls(
                        assistant_msg,
                        collected_tool_results,
                        execution_error,
                        iteration,
                    )

                    # === Deferred tools: stop loop, signal caller ===
                    if deferred_calls:
                        _stopped_for_deferred = True
                        yield StreamEvent(
                            deferred_tool_calls=deferred_calls,
                            is_final=True,
                        )
                        return

                    if (
                        self._should_exit_on_success(iteration_results)
                        and execution_error is None
                    ):
                        outcome = await self._run_hooks(
                            AgentEvent.BEFORE_FINAL_RESPONSE,
                            current_value=assistant_msg,
                            iteration=iteration,
                            assistant_message=assistant_msg,
                        )
                        if outcome.decision == HookDecision.RETRY:
                            continue
                        if outcome.decision == HookDecision.FAIL:
                            raise RuntimeError(
                                "Required tool call missing before final response"
                            )
                        assistant_msg = outcome.value
                        await emit_assistant_span(self._tracer, assistant_msg)
                        response = self._build_final_response(
                            assistant_msg, collected_tool_results, execution_error
                        )
                        await self._run_hooks(
                            AgentEvent.QUERY_END,
                            current_value=response,
                            iteration=iteration,
                        )
                        yield StreamEvent(
                            assistant_message=assistant_msg,
                            assistant_response=response,
                            is_final=True,
                        )
                        return

                    continue

                # === Text response (final) ===
                self.conversation_history.append(assistant_msg)

                outcome = await self._run_hooks(
                    AgentEvent.BEFORE_FINAL_RESPONSE,
                    current_value=assistant_msg,
                    iteration=iteration,
                    assistant_message=assistant_msg,
                )
                if outcome.decision == HookDecision.RETRY:
                    if stream and streamed_tokens:
                        raise RuntimeError(
                            "Hook BEFORE_FINAL_RESPONSE requested RETRY after tokens "
                            "were already streamed to the consumer. This is not "
                            "supported in streaming mode."
                        )
                    continue
                if outcome.decision == HookDecision.FAIL:
                    raise RuntimeError(
                        "Required tool call missing before final response"
                    )
                assistant_msg = outcome.value
                await emit_assistant_span(self._tracer, assistant_msg)
                response = self._build_final_response(
                    assistant_msg, collected_tool_results, execution_error
                )
                await self._run_hooks(
                    AgentEvent.QUERY_END,
                    current_value=response,
                    iteration=iteration,
                )
                yield StreamEvent(
                    assistant_message=assistant_msg,
                    assistant_response=response,
                    is_final=True,
                )
                return

            # Max iterations reached
            _trace_error = f"Max iterations ({self.max_iterations}) reached"
            timeout_response = self._build_timeout_response(
                self.max_iterations, collected_tool_results, execution_error
            )
            yield StreamEvent(
                assistant_response=timeout_response,
                is_final=True,
            )
        except Exception as e:
            _trace_error = str(e)
            raise
        finally:
            if not _stopped_for_deferred:
                # Normal completion or error — close agent span and
                # (if root) the trace. When stopped for deferred, leave
                # everything open for resume_after_deferred().
                await end_agent_trace(
                    self._tracer,
                    is_root_trace,
                    self.conversation_history,
                    self.system_message,
                    error=_trace_error,
                )

    # ─── Input Validation ─────────────────────────────────────────────────────

    def _validate_query_input(self, query: str | list[StandardMessage]) -> None:
        """
        Validates the query input.
        """
        if isinstance(query, list):
            human_messages = [msg for msg in query if isinstance(msg, HumanMessage)]
            if len(human_messages) != 1:
                raise ValueError(
                    f"Message list must contain exactly 1 HumanMessage, "
                    f"found {len(human_messages)}"
                )
        elif not isinstance(query, str):
            raise TypeError(
                f"query must be str or List[StandardMessage], "
                f"received {type(query).__name__}"
            )

    # ─── Tool Dispatch ────────────────────────────────────────────────────────

    async def _process_tool_calls(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: list[ToolResult],
        execution_error: str | None,
        iteration: int,
    ) -> tuple[str | None, list[ToolResult], list[ToolCall] | None]:
        tool_names = [tc.name for tc in assistant_msg.tool_calls]
        logger.debug(
            f"[{self.__class__.__name__}] Dispatching tool batch | iteration={iteration} count={len(tool_names)} tools={tool_names}"
        )

        batch_start_time = time.perf_counter()

        tasks = [
            self._execute_single_tool_with_hooks(call, iteration, batch_start_time)
            for call in assistant_msg.tool_calls
        ]
        tool_results = await asyncio.gather(*tasks)

        batch_duration = time.perf_counter() - batch_start_time

        # Detect deferred tools: is_deferred=True AND result is None
        deferred_calls: list[ToolCall] = []
        resolved_results: list[ToolResult] = []
        for call, result in zip(assistant_msg.tool_calls, tool_results, strict=True):
            tool = self._find_tool(call.name)
            if tool and getattr(tool, "is_deferred", False) and result.result is None:
                deferred_calls.append(call)
            else:
                resolved_results.append(result)

        if deferred_calls:
            # Save assistant message (with tool_calls) to history
            self.conversation_history.append(assistant_msg)
            # Save resolved (non-deferred) results if any
            if resolved_results:
                self.conversation_history.append(
                    ToolMessage(tool_results=resolved_results)
                )
            logger.info(
                f"[{self.__class__.__name__}] Deferred tool detected — stopping loop | "
                f"iteration={iteration} deferred={[c.name for c in deferred_calls]}"
            )
            return execution_error, tool_results, deferred_calls

        collected_tool_results.extend(tool_results)

        current_iteration_errors = [
            f"Error in {r.tool_name} (ID: {r.tool_call_id}): {r.error}"
            for r in tool_results
            if r.error
        ]

        if current_iteration_errors:
            execution_error = "; ".join(current_iteration_errors)
            logger.warning(
                f"[{self.__class__.__name__}] Tool batch completed with errors | iteration={iteration} duration_s={batch_duration:.3f} errors={current_iteration_errors}"
            )
        else:
            execution_error = None
            logger.debug(
                f"[{self.__class__.__name__}] Tool batch completed successfully | iteration={iteration} count={len(tool_results)} duration_s={batch_duration:.3f}"
            )

        tool_message = ToolMessage(tool_results=tool_results)
        self.conversation_history.extend([assistant_msg, tool_message])

        return execution_error, tool_results, None

    def _should_exit_on_success(self, iteration_results: list[ToolResult]) -> bool:
        if not self._exit_on_success:
            return False
        called_tools = {r.tool_name for r in iteration_results}
        return called_tools.issubset(self._exit_on_success)

    async def _execute_single_tool_with_hooks(
        self, call: ToolCall, iteration: int, batch_start_time: float = None
    ) -> ToolResult:
        tool_start_time = time.perf_counter()

        await start_tool_span(self._tracer, call, self.registered_tools)

        outcome = await self._run_hooks(
            AgentEvent.BEFORE_TOOL_EXECUTION,
            current_value=call,
            iteration=iteration,
            tool_call=call,
        )
        if outcome.decision == HookDecision.FAIL:
            raise RuntimeError("Hook BEFORE_TOOL_EXECUTION requested FAIL")
        if outcome.decision == HookDecision.RETRY:
            error_msg = (
                str(outcome.value)
                if outcome.value
                else "Tool execution blocked by hook"
            )
            logger.warning(
                f"[{self.__class__.__name__}] Hook RETRY — tool skipped | tool={call.name} call_id={call.id[-12:]}"
            )
            return ToolResult(
                tool_name=call.name,
                tool_call_id=call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=error_msg,
            )
        call = outcome.value

        result = await self._async_execute_tool(call)

        if not result:
            result = ToolResult(
                tool_name=call.name,
                tool_call_id=call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=f"Tool {call.name} not found or not executable",
            )

        outcome = await self._run_hooks(
            AgentEvent.AFTER_TOOL_EXECUTION,
            current_value=result,
            iteration=iteration,
            tool_result=result,
        )
        if outcome.decision == HookDecision.FAIL:
            raise RuntimeError("Hook AFTER_TOOL_EXECUTION requested FAIL")
        result = outcome.value

        if result.status == ToolStatus.ERROR:
            outcome = await self._run_hooks(
                AgentEvent.ON_TOOL_ERROR,
                current_value=result,
                iteration=iteration,
                tool_result=result,
                error=result.error,
            )
            if outcome.decision == HookDecision.FAIL:
                raise RuntimeError("Hook ON_TOOL_ERROR requested FAIL")
            result = outcome.value

        await end_tool_span(self._tracer, result)

        tool_duration = time.perf_counter() - tool_start_time
        log_level = "warning" if result.status == ToolStatus.ERROR else "debug"
        getattr(logger, log_level)(
            f"[{self.__class__.__name__}] Tool execution completed | tool={result.tool_name} status={result.status} duration_s={tool_duration:.3f} call_id={result.tool_call_id[-12:]}"
        )

        return result

    def _find_tool(self, name: str) -> Tool | None:
        """Find a registered tool by name."""
        for tool in self.registered_tools:
            if getattr(tool, "tool_name", None) == name:
                return tool
        return None

    async def _async_execute_tool(self, tool_call: ToolCall) -> ToolResult | None:
        for tool in self.registered_tools:
            tool_name = getattr(tool, "tool_name", None)
            if tool_name == tool_call.name:
                try:
                    result = await tool.execute(tool_call)
                    return result
                except Exception as e:
                    return ToolResult(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        result=None,
                        status="error",
                        error=f"Tool execution error: {e}",
                    )

        registered_names = [
            getattr(t, "tool_name", t.__class__.__name__) for t in self.registered_tools
        ]
        logger.error(
            f"[{self.__class__.__name__}] Tool not found — LLM requested an unregistered tool | requested_tool={tool_call.name} registered={registered_names}"
        )
        return None

    # ─── Response Building ────────────────────────────────────────────────────

    def _build_final_response(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: list[ToolResult],
        execution_error: str | None,
    ) -> AssistantResponse:
        final_content = (
            assistant_msg.content if assistant_msg.content else "Execution completed."
        )

        logger.info(
            f"[{self.__class__.__name__}] Final response built | tool_results={len(collected_tool_results) if collected_tool_results else 0} has_error={execution_error is not None}"
        )

        # Avoid duplicates - message may already be in history from BEFORE_FINAL_RESPONSE
        # Use identity check (is) because Pydantic __eq__ may have edge cases
        if not any(msg is assistant_msg for msg in self.conversation_history):
            self.conversation_history.append(assistant_msg)

        return AssistantResponse(
            agent_name=self.__class__.__name__,
            content=final_content,
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error,
        )

    def _build_timeout_response(
        self,
        max_iterations: int,
        collected_tool_results: list[ToolResult],
        execution_error: str | None,
    ) -> AssistantResponse:
        logger.warning(
            f"[{self.__class__.__name__}] Max iterations reached — execution stopped | max_iterations={max_iterations} tool_results={len(collected_tool_results) if collected_tool_results else 0}"
        )

        return AssistantResponse(
            agent_name=self.__class__.__name__,
            content=f"Execution stopped after {max_iterations} iterations.",
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error or f"Max iterations ({max_iterations}) reached",
        )

    # ─── Sub-Agent Registration ───────────────────────────────────────────────

    def register_agent(
        self,
        agent: "BaseAgent",
        *,
        name: str,
        description: str,
        stateless: bool = False,
    ) -> None:
        """
        Register a sub-agent as a tool on this agent.

        Any agent can register sub-agents - no decorator needed.

        Args:
            agent: BaseAgent instance to register as sub-agent.
            name: Tool name the LLM will use to call this sub-agent.
            description: Description of what the sub-agent does (shown to LLM).
            stateless: If True, each call executes on an isolated copy (parallel-safe).
                If False (default), calls are serialized and conversation history is shared.
        """
        from obelix.core.agent.subagent_wrapper import SubAgentWrapper

        wrapper = SubAgentWrapper(
            agent,
            name=name,
            description=description,
            stateless=stateless,
        )
        self.registered_tools.append(wrapper)
        logger.info(
            f"[{self.__class__.__name__}] Sub-agent registered as tool | sub_agent={name} stateless={stateless}"
        )

    # ─── Conversation History ─────────────────────────────────────────────────

    @property
    def get_conversation_history(self) -> list[StandardMessage]:
        """
        Return the conversation history
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self, keep_system_message: bool = True):
        """
        Clear the conversation history
        """
        if keep_system_message:
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []
