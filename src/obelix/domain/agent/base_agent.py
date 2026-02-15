# src/base_agent/base_agent.py
import asyncio
import inspect
import time
from typing import List, Optional, Dict, Any, Union, Type, Tuple, Set

from obelix.infrastructure.logging import get_logger
from obelix.domain.model.system_message import SystemMessage
from obelix.domain.model.human_message import HumanMessage
from obelix.domain.model.assistant_message import AssistantMessage, AssistantResponse
from obelix.domain.model.tool_message import ToolMessage, ToolCall, ToolResult, ToolStatus, ToolRequirement
from obelix.domain.agent.hooks import AgentEvent, Hook, AgentStatus, HookDecision, Outcome
from obelix.domain.agent.event_contracts import EventContract, get_event_contracts
from obelix.domain.model.standard_message import StandardMessage
from obelix.domain.model.usage import AgentUsage
from obelix.domain.tool.tool_base import ToolBase
from obelix.ports.outbound.llm_provider import AbstractLLMProvider
from obelix.infrastructure.config import GlobalConfig

logger = get_logger(__name__)


class BaseAgent:
    def __init__(
        self,
        system_message: str,
        provider: Optional[AbstractLLMProvider] = None,
        max_iterations: int = 15,
        tools: Optional[Union[ToolBase, Type[ToolBase], List[Union[Type[ToolBase], ToolBase]]]] = None,
        tool_policy: Optional[List[ToolRequirement]] = None,
        exit_on_success: Optional[List[str]] = None,
    ):
        self.system_message = SystemMessage(content=system_message)
        self.max_iterations = max_iterations
        self._exit_on_success: Set[str] = set(exit_on_success) if exit_on_success else set()

        self.provider = provider or GlobalConfig().get_current_provider_instance()

        self.registered_tools: List[ToolBase] = []
        self.conversation_history: List[StandardMessage] = [self.system_message]
        self.agent_usage = AgentUsage(model_id=self.provider.model_id)
        self._tool_policy = tool_policy or []

        # Hook system
        self._hooks: Dict[AgentEvent, List[Hook]] = {event: [] for event in AgentEvent}
        self._event_contracts: Dict[AgentEvent, EventContract] = get_event_contracts()

        # Register tools from constructor parameter
        if tools:
            if not isinstance(tools, list):
                tools = [tools]
            for tool in tools:
                tool_instance = tool() if inspect.isclass(tool) else tool
                self.register_tool(tool_instance)

        if self._tool_policy:
            self.on(AgentEvent.BEFORE_FINAL_RESPONSE) \
                .when(self._tool_policy_should_fail) \
                .handle(
                    decision=HookDecision.FAIL,
                    effects=[self._tool_policy_inject_message],
                )
            self.on(AgentEvent.BEFORE_FINAL_RESPONSE) \
                .when(self._tool_policy_should_retry) \
                .handle(
                    decision=HookDecision.RETRY,
                    effects=[self._tool_policy_inject_message],
                )

    def register_tool(self, tool: ToolBase):
        """
        Registers a tool for the agent
        """
        try:
            from obelix.plugins.mcp.mcp_tool import MCPTool
            if isinstance(tool, MCPTool):
                if not tool.manager.is_connected():
                    logger.error(
                        f"Agent {self.__class__.__name__}: MCP Tool registration attempt for {tool.tool_name} "
                        "failed - manager not connected"
                    )
                    raise RuntimeError(
                        f"MCP Tool {tool.tool_name} is not connected. "
                        "The manager must be connected before registration."
                    )
        except ImportError:
            logger.debug("MCP tools not available (mcp library import failed)")

        if tool not in self.registered_tools:
            self.registered_tools.append(tool)
            tool_name = getattr(tool, 'tool_name', None) or tool.__class__.__name__
            logger.info(f"Agent {self.__class__.__name__}: tool '{tool_name}' registered")

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

    def _retryable_events(self) -> List[AgentEvent]:
        return [evt for evt, c in self._event_contracts.items() if c.retryable]

    async def _run_hooks(
        self,
        event: AgentEvent,
        current_value: Any = None,
        **ctx_kwargs
    ) -> Outcome:
        """
        Executes all hooks registered for an event.
        """
        agent_status = AgentStatus(
            event=event,
            agent=self,
            **ctx_kwargs
        )

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
                if outcome.value is None or not isinstance(outcome.value, contract.stop_output):
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

    async def execute_query_async(
        self,
        query: Union[str, List[StandardMessage]]
    ) -> AssistantResponse:
        """
        Execute query asynchronously (for FastAPI).
        """
        self._validate_query_input(query)
        return await self._async_execute_query(query)

    def execute_query(
        self,
        query: Union[str, List[StandardMessage]]
    ) -> AssistantResponse:
        """
        Execute query synchronously (for CLI).
        """
        return asyncio.run(self.execute_query_async(query))

    def _validate_query_input(self, query: Union[str, List[StandardMessage]]) -> None:
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

    async def _async_execute_query(self, query: Union[str, List[StandardMessage]]) -> AssistantResponse:
        """
        Asynchronous query execution (core execution engine)
        """
        collected_tool_results = []
        execution_error = None

        if isinstance(query, str):
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

        for iteration in range(1, self.max_iterations + 1):
            outcome = await self._run_hooks(AgentEvent.BEFORE_LLM_CALL, iteration=iteration)
            if outcome.decision == HookDecision.STOP:
                assistant_msg = outcome.value
                response = self._build_final_response(
                    assistant_msg,
                    collected_tool_results,
                    execution_error
                )
                await self._run_hooks(AgentEvent.QUERY_END, current_value=response, iteration=iteration)
                return response
            if outcome.decision == HookDecision.FAIL:
                raise RuntimeError("Hook BEFORE_LLM_CALL requested FAIL")

            assistant_msg = await self.provider.invoke(self.conversation_history, self.registered_tools)

            outcome = await self._run_hooks(
                AgentEvent.AFTER_LLM_CALL,
                current_value=assistant_msg,
                iteration=iteration,
                assistant_message=assistant_msg
            )
            if outcome.decision == HookDecision.RETRY:
                continue
            if outcome.decision == HookDecision.FAIL:
                raise RuntimeError("Hook AFTER_LLM_CALL requested FAIL")
            if outcome.decision == HookDecision.STOP:
                assistant_msg = outcome.value
                response = self._build_final_response(
                    assistant_msg,
                    collected_tool_results,
                    execution_error
                )
                await self._run_hooks(AgentEvent.QUERY_END, current_value=response, iteration=iteration)
                return response
            assistant_msg = outcome.value

            if assistant_msg.usage:
                self.agent_usage.add_usage(assistant_msg.usage)

            if not assistant_msg.tool_calls and not assistant_msg.content:
                logger.info("Agent completed execution without final text response")
                self.conversation_history.append(assistant_msg)
                outcome = await self._run_hooks(
                    AgentEvent.BEFORE_FINAL_RESPONSE,
                    current_value=assistant_msg,
                    iteration=iteration,
                    assistant_message=assistant_msg
                )
                if outcome.decision == HookDecision.RETRY:
                    continue
                if outcome.decision == HookDecision.FAIL:
                    raise RuntimeError("Required tool call missing before final response")
                assistant_msg = outcome.value
                response = self._build_final_response(
                    assistant_msg,
                    collected_tool_results,
                    execution_error
                )
                await self._run_hooks(AgentEvent.QUERY_END, current_value=response, iteration=iteration)
                return response

            if assistant_msg.tool_calls:
                execution_error, iteration_results = await self._process_tool_calls(
                    assistant_msg, collected_tool_results, execution_error, iteration
                )

                if self._should_exit_on_success(iteration_results) and execution_error is None:
                    outcome = await self._run_hooks(
                        AgentEvent.BEFORE_FINAL_RESPONSE,
                        current_value=assistant_msg,
                        iteration=iteration,
                        assistant_message=assistant_msg
                    )
                    if outcome.decision == HookDecision.RETRY:
                        continue
                    if outcome.decision == HookDecision.FAIL:
                        raise RuntimeError("Required tool call missing before final response")
                    assistant_msg = outcome.value
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._run_hooks(AgentEvent.QUERY_END, current_value=response, iteration=iteration)
                    return response

                continue

            # Add assistant message to history BEFORE validation hooks
            # This allows hooks to see (and LLM to correct) the message on RETRY
            self.conversation_history.append(assistant_msg)

            outcome = await self._run_hooks(
                AgentEvent.BEFORE_FINAL_RESPONSE,
                current_value=assistant_msg,
                iteration=iteration,
                assistant_message=assistant_msg
            )
            if outcome.decision == HookDecision.RETRY:
                continue
            if outcome.decision == HookDecision.FAIL:
                raise RuntimeError("Required tool call missing before final response")
            assistant_msg = outcome.value
            response = self._build_final_response(
                assistant_msg, collected_tool_results, execution_error
            )
            await self._run_hooks(AgentEvent.QUERY_END, current_value=response, iteration=iteration)
            return response

        return self._build_timeout_response(
            self.max_iterations, collected_tool_results, execution_error
        )

    async def _process_tool_calls(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[ToolResult],
        execution_error: Optional[str],
        iteration: int
    ) -> Tuple[Optional[str], List[ToolResult]]:
        logger.debug(f"Processing {len(assistant_msg.tool_calls)} tool call(s)")
        logger.debug(f"Tool names: {[tc.name for tc in assistant_msg.tool_calls]}")

        batch_start_time = time.perf_counter()
        logger.debug(f"START ELABORATING TOOLS: n {len(assistant_msg.tool_calls)} tools @ t=0.000s")

        tasks = [
            self._execute_single_tool_with_hooks(call, iteration, batch_start_time)
            for call in assistant_msg.tool_calls
        ]
        tool_results = await asyncio.gather(*tasks)

        batch_duration = time.perf_counter() - batch_start_time
        logger.debug(f"PARALLEL END: {len(tool_results)} tools completed in {batch_duration:.3f}s")

        collected_tool_results.extend(tool_results)

        current_iteration_errors = [
            f"Error in {r.tool_name} (ID: {r.tool_call_id}): {r.error}"
            for r in tool_results
            if r.error
        ]

        if current_iteration_errors:
            execution_error = "; ".join(current_iteration_errors)
            logger.warning(f"Tool execution errors: {execution_error}")
        else:
            execution_error = None
            logger.debug("All tools executed successfully")

        tool_message = ToolMessage(tool_results=tool_results)
        self.conversation_history.extend([assistant_msg, tool_message])

        logger.debug(f"Added {len(tool_results)} tool result(s) to conversation history")

        return execution_error, tool_results

    def _should_exit_on_success(self, iteration_results: List[ToolResult]) -> bool:
        if not self._exit_on_success:
            return False
        called_tools = {r.tool_name for r in iteration_results}
        return called_tools.issubset(self._exit_on_success)

    def _build_final_response(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[ToolResult],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        final_content = assistant_msg.content if assistant_msg.content else "Execution completed."

        logger.info(f"Assistant response generated for agent {self.__class__.__name__}")
        logger.debug(f"Final response: tool_results count={len(collected_tool_results) if collected_tool_results else 0}")

        # Avoid duplicates - message may already be in history from BEFORE_FINAL_RESPONSE
        # Use identity check (is) because Pydantic __eq__ may have edge cases
        if not any(msg is assistant_msg for msg in self.conversation_history):
            self.conversation_history.append(assistant_msg)

        return AssistantResponse(
            agent_name=self.__class__.__name__,
            content=final_content,
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error
        )

    def _build_timeout_response(
        self,
        max_iterations: int,
        collected_tool_results: List[ToolResult],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        warning_msg = f"Reached iteration limit of {max_iterations}. Execution stopped."
        print(warning_msg)
        logger.warning(warning_msg)

        return AssistantResponse(
            agent_name=self.__class__.__name__,
            content=f"Execution stopped after {max_iterations} iterations.",
            tool_results=collected_tool_results if collected_tool_results else None,
            error=execution_error or warning_msg
        )

    async def _execute_single_tool_with_hooks(
        self,
        call: ToolCall,
        iteration: int,
        batch_start_time: float = None
    ) -> ToolResult:
        tool_start_time = time.perf_counter()
        if batch_start_time:
            relative_start = tool_start_time - batch_start_time
            logger.debug(f"TOOL START: {call.name} (ID: {call.id[-12:]}) @ t={relative_start:.3f}s")

        outcome = await self._run_hooks(
            AgentEvent.BEFORE_TOOL_EXECUTION,
            current_value=call,
            iteration=iteration,
            tool_call=call
        )
        if outcome.decision == HookDecision.FAIL:
            raise RuntimeError("Hook BEFORE_TOOL_EXECUTION requested FAIL")
        call = outcome.value

        result = await self._async_execute_tool(call)

        if not result:
            result = ToolResult(
                tool_name=call.name,
                tool_call_id=call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=f"Tool {call.name} not found or not executable"
            )

        outcome = await self._run_hooks(
            AgentEvent.AFTER_TOOL_EXECUTION,
            current_value=result,
            iteration=iteration,
            tool_result=result
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
                error=result.error
            )
            if outcome.decision == HookDecision.FAIL:
                raise RuntimeError("Hook ON_TOOL_ERROR requested FAIL")
            result = outcome.value

        if batch_start_time:
            tool_duration = time.perf_counter() - tool_start_time
            relative_end = time.perf_counter() - batch_start_time
            logger.debug(
                f"TOOL END: {result.tool_name} (ID: {result.tool_call_id[-12:]}) "
                f"@ t={relative_end:.3f}s (duration: {tool_duration:.3f}s) status={result.status}"
            )
        else:
            logger.debug(
                f"Tool executed: {result.tool_name} "
                f"(ID: {result.tool_call_id}), status={result.status}"
            )

        return result

    async def _async_execute_tool(self, tool_call: ToolCall) -> Optional[ToolResult]:
        for tool in self.registered_tools:
            tool_name = getattr(tool, 'tool_name', None)
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
                        error=f"Tool execution error: {e}"
                    )

        logger.error(f"Tool {tool_call.name} not found among registered tools")
        logger.error(f"Registered tools: {[getattr(t, 'tool_name', t.__class__.__name__) for t in self.registered_tools]}")
        return None

    def _collect_tool_results(self, history: Optional[List[StandardMessage]] = None) -> List[ToolResult]:
        results: List[ToolResult] = []
        target_history = history if history is not None else self.conversation_history
        for message in target_history:
            if isinstance(message, ToolMessage):
                results.extend(message.tool_results)
        return results

    def _get_tool_policy_violation(
        self,
        status: AgentStatus,
    ) -> Tuple[Optional[HookDecision], Optional[str]]:
        if not self._tool_policy:
            return None, None

        cached = getattr(status, "_tool_policy_cache", None)
        if cached is not None:
            return cached

        # Use status.agent to support stateless subagent copies
        results = self._collect_tool_results(status.agent.conversation_history)
        logger.debug(
            f"Tool policy check: required={[r.tool_name for r in self._tool_policy]}, "
            f"results={[r.tool_name for r in results]}"
        )

        for req in self._tool_policy:
            matching = [r for r in results if r.tool_name == req.tool_name]
            logger.debug(
                f"Tool policy match: tool={req.tool_name}, "
                f"count={len(matching)}, min_calls={req.min_calls}, require_success={req.require_success}"
            )
            if len(matching) < req.min_calls:
                msg = req.error_message or (
                    f"You must call tool '{req.tool_name}' at least {req.min_calls} time(s) "
                    "before responding."
                )
                decision = (
                    HookDecision.FAIL
                    if status.iteration >= status.agent.max_iterations
                    else HookDecision.RETRY
                )
                logger.warning(
                    "Tool policy violation: tool={}, count={}, min_calls={}, require_success={}, decision={}",
                    req.tool_name,
                    len(matching),
                    req.min_calls,
                    req.require_success,
                    decision.value,
                )
                status._tool_policy_cache = (decision, msg)
                return decision, msg

            if req.require_success:
                success_count = sum(1 for r in matching if r.status == ToolStatus.SUCCESS)
                if success_count < req.min_calls:
                    msg = req.error_message or (
                        f"Tool '{req.tool_name}' failed. "
                        "Retry and ensure the result is SUCCESS."
                    )
                    decision = (
                        HookDecision.FAIL
                        if status.iteration >= status.agent.max_iterations
                        else HookDecision.RETRY
                    )
                    logger.warning(
                        "Tool policy violation: tool={}, success_count={}, min_calls={}, require_success={}, decision={}",
                        req.tool_name,
                        success_count,
                        req.min_calls,
                        req.require_success,
                        decision.value,
                    )
                    status._tool_policy_cache = (decision, msg)
                    return decision, msg

        status._tool_policy_cache = (None, None)
        return None, None

    def _tool_policy_should_retry(self, status: AgentStatus) -> bool:
        decision, _ = self._get_tool_policy_violation(status)
        return decision == HookDecision.RETRY

    def _tool_policy_should_fail(self, status: AgentStatus) -> bool:
        decision, _ = self._get_tool_policy_violation(status)
        return decision == HookDecision.FAIL

    def _tool_policy_inject_message(self, status: AgentStatus) -> None:
        _, msg = self._get_tool_policy_violation(status)
        if msg:
            status.agent.conversation_history.append(SystemMessage(content=msg))

    @property
    def get_conversation_history(self) -> List[StandardMessage]:
        """
        Return the conversation history
        """
        return self.conversation_history.copy()


    def register_agent(
        self,
        agent: 'BaseAgent',
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
        from obelix.domain.agent.subagent_wrapper import SubAgentWrapper

        wrapper = SubAgentWrapper(
            agent,
            name=name,
            description=description,
            stateless=stateless,
        )
        self.registered_tools.append(wrapper)
        logger.info(
            f"Agent {self.__class__.__name__}: sub-agent '{name}' registered"
        )

    def clear_conversation_history(self, keep_system_message: bool = True):
        """
        Clear the conversation history
        """
        if keep_system_message:
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []
