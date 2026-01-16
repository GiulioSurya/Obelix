# src/base_agent/base_agent.py
import asyncio
import inspect
import time
from typing import List, Optional, Dict, Any, Union, Type
from src.logging_config import get_logger
from src.messages.system_message import SystemMessage
from src.messages.human_message import HumanMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage, ToolCall, ToolResult, ToolStatus
from src.base_agent.hooks import AgentEvent, Hook, AgentStatus
from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantResponse
from src.messages.usage import AgentUsage
from src.tools.tool_base import ToolBase
from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.config import GlobalConfig

logger = get_logger(__name__)


class BaseAgent:
    def __init__(self,
                 system_message: str,
                 provider: Optional[AbstractLLMProvider] = None,
                 agent_comment: bool = True,
                 max_iterations: int = 15,
                 max_attempts: int = 5,
                 tools: Optional[Union[ToolBase, Type[ToolBase], List[Union[Type[ToolBase], ToolBase]]]] = None):
        self.system_message = SystemMessage(content=system_message)
        self.agent_comment = agent_comment
        self.max_iterations = max_iterations
        self.max_attempts = max_attempts

        self.provider = provider or GlobalConfig().get_current_provider_instance()

        self.registered_tools: List[ToolBase] = []
        self.conversation_history: List[StandardMessage] = [self.system_message]
        self.agent_usage = AgentUsage(model_id=self.provider.model_id)

        # Hook system
        self._hooks: Dict[AgentEvent, List[Hook]] = {
            event: [] for event in AgentEvent
        }

        # Register tools from constructor parameter
        if tools:
            if not isinstance(tools, list):
                tools = [tools]
            for tool in tools:
                tool_instance = tool() if inspect.isclass(tool) else tool
                self.register_tool(tool_instance)

    def register_tool(self, tool: ToolBase):
        """
        Registers a tool for the agent

        Args:
            tool: Tool to register (ToolBase or MCPTool)

        Raises:
            RuntimeError: If the MCP tool is not connected
        """
        # Lazy import of MCPTool to avoid Windows dependencies on Linux/Docker
        try:
            from src.tools.mcp.mcp_tool import MCPTool
            # If it's an MCPTool, check that the manager is connected
            if isinstance(tool, MCPTool):
                if not tool.manager.is_connected():
                    logger.error(f"Agent {self.__class__.__name__}: MCP Tool registration attempt for {tool.tool_name} failed - manager not connected")
                    raise RuntimeError(f"MCP Tool {tool.tool_name} is not connected. "
                                       f"The manager must be connected before registration.")


        except ImportError:
            # MCPTool not available (e.g. missing pywin32 on Windows or Docker environment)
            logger.debug("MCP tools not available (mcp library import failed)")

        # Register the tool
        if tool not in self.registered_tools:
            self.registered_tools.append(tool)
            # Direct access to tool.tool_name (populated by @tool decorator or MCPTool)
            tool_name = getattr(tool, 'tool_name', None) or tool.__class__.__name__
            logger.info(f"Agent {self.__class__.__name__}: tool '{tool_name}' registered")

    def on(self, event: AgentEvent) -> Hook:
        """
        Fluent API for registering hooks.

        Examples:
            agent.on(AgentEvent.ON_TOOL_ERROR).when(...).inject(...)
            agent.on(AgentEvent.AFTER_LLM_CALL).transform(...)

        Args:
            event: The event to register the hook on

        Returns:
            Hook: Hook object for method chaining
        """
        hook = Hook(event)
        self._hooks[event].append(hook)
        return hook

    async def _trigger_hooks(
        self,
        event: AgentEvent,
        current_value: Any = None,
        **ctx_kwargs
    ) -> Any:
        """
        Executes all hooks registered for an event.

        Args:
            event: The event to trigger
            current_value: Current value to pass to hooks
            **ctx_kwargs: Additional parameters for AgentStatus

        Returns:
            Value transformed by hooks
        """
        agent_status = AgentStatus(
            event=event,
            agent=self,
            **ctx_kwargs
        )

        result = current_value
        for hook in self._hooks[event]:
            result = await hook.execute(agent_status, result)

        return result

    async def execute_query_async(
        self,
        query: Union[str, List[StandardMessage]]
    ) -> AssistantResponse:
        """
        Execute query asynchronously (for FastAPI).

        Native async version that uses the existing event loop.
        Includes retry logic for fatal errors (LLM crash, network timeout) and input validation.

        Args:
            query: Query to execute. Can be:
                   - str: string automatically converted to HumanMessage
                   - List[StandardMessage]: list of messages (must contain exactly 1 HumanMessage)

        Returns:
            AssistantResponse: Structured agent response

        Raises:
            ValueError: If message list does not contain exactly 1 HumanMessage
            TypeError: If query is not str or List[StandardMessage]
            RuntimeError: If all retry attempts fail
        """
        # 1. Input validation
        self._validate_query_input(query)

        # 2. Backup conversation history for clean retries
        original_history = self.conversation_history.copy()

        # 3. Retry loop for fatal errors
        for attempt in range(self.max_attempts):
            try:
                return await self._async_execute_query(query)
            except (ValueError, TypeError):
                # Validation errors should not be retried (already validated)
                raise
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    raise RuntimeError(
                        f"Execution failed after {self.max_attempts} attempts. "
                        f"Last error: {e}"
                    )
                # Restore history for clean retry
                self.conversation_history = original_history.copy()
                # WARNING: retry in progress - anomalous but handled situation
                logger.warning(f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

        raise RuntimeError("Execution failed for unknown reasons")

    def execute_query(
        self,
        query: Union[str, List[StandardMessage]]
    ) -> AssistantResponse:
        """
        Execute query synchronously (for CLI).

        Sync wrapper that creates a new event loop to run the async version.
        Includes retry logic for fatal errors and input validation.

        Args:
            query: Query to execute. Can be:
                   - str: string automatically converted to HumanMessage
                   - List[StandardMessage]: list of messages (must contain exactly 1 HumanMessage)

        Returns:
            AssistantResponse: Structured agent response

        Raises:
            ValueError: If message list does not contain exactly 1 HumanMessage
            TypeError: If query is not str or List[StandardMessage]
            RuntimeError: If all retry attempts fail
        """
        return asyncio.run(self.execute_query_async(query))

    def _validate_query_input(self, query: Union[str, List[StandardMessage]]) -> None:
        """
        Validates the query input.

        Args:
            query: Query to validate

        Raises:
            ValueError: If message list does not contain exactly 1 HumanMessage
            TypeError: If query is not str or List[StandardMessage]
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

        Args:
            query: Query to execute. Can be:
                   - str: string automatically converted to HumanMessage
                   - List[StandardMessage]: list of messages (must contain exactly 1 HumanMessage)

        Returns:
            AssistantResponse: Structured agent response

        Raises:
            ValueError: If message list does not contain exactly 1 HumanMessage
        """
        collected_tool_results = []
        execution_error = None

        # Handle input: string or list of messages
        if isinstance(query, str):
            # Case 1: string → create HumanMessage
            user_message = HumanMessage(content=query)
            self.conversation_history.append(user_message)
        elif isinstance(query, list):
            # Case 2: list of messages → validate and add
            # Count HumanMessages in the list
            human_messages = [msg for msg in query if isinstance(msg, HumanMessage)]

            if len(human_messages) != 1:
                raise ValueError(
                    f"Message list must contain exactly 1 HumanMessage, "
                    f"found {len(human_messages)}"
                )

            # Add all messages to conversation history
            self.conversation_history.extend(query)
        else:
            raise TypeError(
                f"query must be str or List[StandardMessage], "
                f"received {type(query).__name__}"
            )

        # >>> HOOKS: ON_QUERY_START <<<
        await self._trigger_hooks(AgentEvent.ON_QUERY_START, iteration=0)

        # Multi-turn loop to handle tool calls
        for iteration in range(1, self.max_iterations + 1):
            # >>> HOOKS: BEFORE_LLM_CALL <<<
            await self._trigger_hooks(AgentEvent.BEFORE_LLM_CALL, iteration=iteration)

            assistant_msg = self.provider.invoke(self.conversation_history, self.registered_tools)

            # >>> HOOKS: AFTER_LLM_CALL <<< (can transform assistant_msg)
            assistant_msg = await self._trigger_hooks(
                AgentEvent.AFTER_LLM_CALL,
                current_value=assistant_msg,
                iteration=iteration,
                assistant_message=assistant_msg
            )

            # Track usage from LLM call (if available)
            if assistant_msg.usage:
                self.agent_usage.add_usage(assistant_msg.usage)

            # Case 1: Empty response → valid loop exit
            if not assistant_msg.tool_calls and not assistant_msg.content:
                #print("[INFO] Agent completed execution without final text response.")
                # INFO: normal application event
                logger.info("Agent completed execution without final text response")
                self.conversation_history.append(assistant_msg)
                response = self._build_final_response(
                    assistant_msg,
                    collected_tool_results,
                    execution_error
                )
                # >>> HOOKS: ON_QUERY_END <<<
                await self._trigger_hooks(AgentEvent.ON_QUERY_END, iteration=iteration)
                return response

            # Case 2: Tool calls → process
            if assistant_msg.tool_calls:
                execution_error = await self._process_tool_calls(
                    assistant_msg, collected_tool_results, execution_error, iteration
                )

                # If agent_comment=False and no error, exit without LLM comment
                if not self.agent_comment and execution_error is None:
                    response = self._build_final_response(
                        assistant_msg, collected_tool_results, execution_error
                    )
                    await self._trigger_hooks(AgentEvent.ON_QUERY_END, iteration=iteration)
                    return response

                continue

            # Case 3: Text content → final response
            response = self._build_final_response(
                assistant_msg, collected_tool_results, execution_error
            )
            await self._trigger_hooks(AgentEvent.ON_QUERY_END, iteration=iteration)
            return response

        # >>> HOOKS: ON_MAX_ITERATIONS <<<
        await self._trigger_hooks(AgentEvent.ON_MAX_ITERATIONS, iteration=self.max_iterations)
        return self._build_timeout_response(
            self.max_iterations, collected_tool_results, execution_error
        )

    async def _process_tool_calls(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[ToolResult],
        execution_error: Optional[str],
        iteration: int
    ) -> Optional[str]:
        """
        Process assistant tool calls with parallel execution.

        Flow:
        1. Execute all tools in parallel with asyncio.gather()
           (if N=1, gather still executes correctly)
        2. Collect results and create summary for final response
        3. Track ALL errors from current iteration (with tool_call_id)
        4. Add AssistantMessage + ToolMessage to conversation history
        5. Return execution_error (None if all successful, string with errors otherwise)

        Args:
            assistant_msg: Assistant message with tool calls
            collected_tool_results: List to collect results
                                    (accumulated between iterations for final response)
            execution_error: Previous execution error (ignored,
                             overwritten with current iteration errors)
            iteration: Current iteration number

        Returns:
            Optional[str]: String with ALL errors from current iteration,
                          format: "Error in tool_name (ID: call_id): message; ..."
                          or None if all tools succeeded.

        Note:
            - asyncio.gather() executes tools in parallel automatically
            - If N=1, gather still executes correctly (no branch needed)
            - execution_error is OVERWRITTEN with current iteration errors
            - Does not track errors between different iterations (known limitation)
            - If a tool succeeds in iteration 2, iteration 1 error is lost
              (this is intentional: if LLM responds, it means it handled the error)
        """

        logger.debug(f"Processing {len(assistant_msg.tool_calls)} tool call(s)")
        logger.debug(f"Tool names: {[tc.name for tc in assistant_msg.tool_calls]}")

        batch_start_time = time.perf_counter()
        logger.debug(f"START ELABORATING TOOLS: n {len(assistant_msg.tool_calls)} tools @ t=0.000s")

        #parallel tools call
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
            if r.error  # Filter only tools with errors
        ]

        if current_iteration_errors:
            # Combine all errors in a string separated by "; "
            execution_error = "; ".join(current_iteration_errors)
            logger.warning(f"Tool execution errors: {execution_error}")
        else:
            # All tools succeeded → reset execution_error
            execution_error = None
            logger.debug("All tools executed successfully")


        tool_message = ToolMessage(tool_results=tool_results)
        self.conversation_history.extend([assistant_msg, tool_message])

        logger.debug(f"Added {len(tool_results)} tool result(s) to conversation history")

        return execution_error

    def _build_final_response(
        self,
        assistant_msg: AssistantMessage,
        collected_tool_results: List[ToolResult],
        execution_error: Optional[str]
    ) -> AssistantResponse:
        """
        Build the final agent response

        Args:
            assistant_msg: Final assistant message
            collected_tool_results: Results from executed tools
            execution_error: Execution error (if present)

        Returns:
            Structured AssistantResponse
        """
        # Fallback for empty content
        final_content = assistant_msg.content if assistant_msg.content else "Execution completed."

        # INFO: business operation completed
        logger.info(f"Assistant response generated for agent {self.__class__.__name__}")
        logger.debug(f"Final response: tool_results count={len(collected_tool_results) if collected_tool_results else 0}")
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
        """
        Build response when iteration limit is reached

        Args:
            max_iterations: Maximum iterations reached
            collected_tool_results: Results from executed tools
            execution_error: Execution error (if present)

        Returns:
            AssistantResponse with timeout warning
        """
        warning_msg = f"Reached iteration limit of {max_iterations}. Execution stopped."
        print(warning_msg)
        # WARNING: limit reached, anomalous situation
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
        """
        Execute a single tool with all its hooks.

        Handles the complete execution cycle for a tool:
        1. BEFORE_TOOL_EXECUTION hook (can transform ToolCall)
        2. Actual tool execution
        3. AFTER_TOOL_EXECUTION hook (can transform ToolResult)
        4. ON_TOOL_ERROR hook (only if error)

        Args:
            call: ToolCall to execute (contains id, name, arguments)
            iteration: Current iteration number of agent loop
            batch_start_time: Batch start timestamp for parallelism debugging

        Returns:
            ToolResult: Execution result with:
                - tool_name: tool name
                - tool_call_id: ID provided by provider (for tracking)
                - result: tool result (if success)
                - status: SUCCESS or ERROR
                - error: error message (if ERROR)

        Note:
            - This function is async-safe and can be executed in parallel
            - Hooks are executed sequentially for each tool
            - If tool not found, returns ToolResult with ERROR status
        """
        # === DEBUG PARALLELISM: log tool start ===
        tool_start_time = time.perf_counter()
        if batch_start_time:
            relative_start = tool_start_time - batch_start_time
            logger.debug(f"TOOL START: {call.name} (ID: {call.id[-12:]}) @ t={relative_start:.3f}s")

        call = await self._trigger_hooks(
            AgentEvent.BEFORE_TOOL_EXECUTION,
            current_value=call,
            iteration=iteration,
            tool_call=call
        )


        result = await self._async_execute_tool(call)

        # Fallback: if tool not found, create error ToolResult
        if not result:
            result = ToolResult(
                tool_name=call.name,
                tool_call_id=call.id,
                result=None,
                status=ToolStatus.ERROR,
                error=f"Tool {call.name} not found or not executable"
            )

        result = await self._trigger_hooks(
            AgentEvent.AFTER_TOOL_EXECUTION,
            current_value=result,
            iteration=iteration,
            tool_result=result
        )

        if result.status == ToolStatus.ERROR:
            await self._trigger_hooks(
                AgentEvent.ON_TOOL_ERROR,
                iteration=iteration,
                tool_result=result,
                error=result.error
            )

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
        """
        Helper for asynchronous tool execution

        Args:
            tool_call: Tool call to execute

        Returns:
            ToolResult: Execution result, None if tool not found
        """

        for tool in self.registered_tools:
            # Direct access to tool.tool_name (@tool decorator or MCPTool)
            tool_name = getattr(tool, 'tool_name', None)
            if tool_name == tool_call.name:
                try:
                    result = await tool.execute(tool_call)
                    return result
                except Exception as e:
                    # Create ToolResult with error instead of failing completely
                    return ToolResult(
                        tool_name=tool_call.name,
                        tool_call_id=tool_call.id,
                        result=None,
                        status="error",
                        error=f"Tool execution error: {e}"
                    )

        # print(f"[ERROR] Tool {tool_call.name} not found among registered tools")
        # print(f"[ERROR] Registered tools: {[getattr(t, 'tool_name', t.__class__.__name__) for t in self.registered_tools]}")
        # ERROR: operation failed, tool not found
        logger.error(f"Tool {tool_call.name} not found among registered tools")
        logger.error(f"Registered tools: {[getattr(t, 'tool_name', t.__class__.__name__) for t in self.registered_tools]}")
        return None

    def get_conversation_history(self) -> List[StandardMessage]:
        """
        Return the conversation history

        Returns:
            List[StandardMessage]: List of conversation messages
        """
        return self.conversation_history.copy()

    def clear_conversation_history(self, keep_system_message: bool = True):
        """
        Clear the conversation history

        Args:
            keep_system_message: Whether to keep the system message
        """
        if keep_system_message:
            self.conversation_history = [self.system_message]
        else:
            self.conversation_history = []
