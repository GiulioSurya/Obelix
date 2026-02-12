# src/adapters/outbound/ibm/provider.py
"""
IBM Watson X Provider.

Self-contained provider with inline message/tool conversion.
No external mapping dependencies.

Uses OpenAI-compatible format (dict-based responses).
IBM SDK has built-in retry for transport errors (429, 503, 504, 520),
so tenacity is NOT needed here - only ToolCallExtractionError retry loop.
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Type

from pydantic import ValidationError

from src.ports.outbound.llm_provider import AbstractLLMProvider
from src.core.model import SystemMessage, HumanMessage, AssistantMessage, ToolMessage, StandardMessage
from src.core.model.tool_message import ToolCall
from src.core.model.usage import Usage
from src.core.tool.tool_base import ToolBase
from src.infrastructure.providers import Providers
from src.adapters.outbound.ibm.connection import IBMConnection
from src.infrastructure.logging import get_logger, format_message_for_trace

try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
except ImportError:
    raise ImportError(
        "ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai"
    )

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""
    pass


class IBMWatsonXLLm(AbstractLLMProvider):
    """
    Provider for IBM Watson X with configurable parameters.

    Self-contained: all conversion logic is inline.
    Uses OpenAI-compatible dict format for messages and responses.
    """

    MAX_EXTRACTION_RETRIES = 3

    @property
    def provider_type(self) -> Providers:
        return Providers.IBM_WATSON

    def __init__(self,
                 connection: Optional[IBMConnection] = None,
                 model_id: str = "meta-llama/llama-3-3-70b-instruct",
                 max_tokens: int = 3000,
                 temperature: float = 0.3,
                 top_p: Optional[float] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 logprobs: Optional[bool] = None,
                 top_logprobs: Optional[int] = None,
                 n: Optional[int] = None,
                 logit_bias: Optional[Dict[int, float]] = None):
        """
        Initialize the IBM Watson X provider with dependency injection of connection.

        Args:
            connection: IBMConnection singleton (default: None, reuse from GlobalConfig if provider matches)
            model_id: Model ID (default: "meta-llama/llama-3-3-70b-instruct")
            max_tokens: Maximum number of tokens (default: 3000)
            temperature: Sampling temperature (default: 0.3)
            top_p: Top-p sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            frequency_penalty: Token frequency penalty (default: None)
            presence_penalty: Token presence penalty (default: None)
            logprobs: Return log probabilities (default: None)
            top_logprobs: Number of top log probabilities (default: None)
            n: Number of completions to generate (default: None)
            logit_bias: Bias for specific tokens (default: None)

        Raises:
            ValueError: If connection=None and GlobalConfig does not have IBM_WATSON set
        """
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.IBM_WATSON,
                "IBMWatsonXLLm"
            )

        self.connection = connection
        self.model_id = model_id

        # Build parameters
        params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if top_p is not None:
            params_dict["top_p"] = top_p
        if seed is not None:
            params_dict["seed"] = seed
        if stop is not None:
            params_dict["stop"] = stop
        if frequency_penalty is not None:
            params_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params_dict["presence_penalty"] = presence_penalty
        if logprobs is not None:
            params_dict["logprobs"] = logprobs
        if top_logprobs is not None:
            params_dict["top_logprobs"] = top_logprobs
        if n is not None:
            params_dict["n"] = n
        if logit_bias is not None:
            params_dict["logit_bias"] = logit_bias

        # Create ModelInference using credentials from connection
        credentials = self.connection.get_client()
        self.client = ModelInference(
            model_id=model_id,
            params=TextChatParameters(**params_dict),
            credentials=credentials,
            project_id=self.connection.get_project_id()
        )

    # ========== INVOKE ==========

    async def invoke(
        self,
        messages: List[StandardMessage],
        tools: List[ToolBase],
        response_schema: Optional[Type["BaseModel"]] = None,
    ) -> AssistantMessage:
        """
        Call the IBM Watson model with standardized messages and tools.

        Uses asyncio.to_thread() to run the sync IBM SDK client without
        blocking the event loop. IBM SDK handles transport retries internally.
        If tool call extraction fails, retries with error feedback to LLM.
        """
        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            logger.debug(f"IBM Watson invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}")

            converted_messages = self._convert_messages(working_messages)

            if converted_tools:
                response = await asyncio.to_thread(
                    self.client.chat,
                    messages=converted_messages,
                    tools=converted_tools,
                    tool_choice_option="auto"
                )
            else:
                response = await asyncio.to_thread(
                    self.client.chat,
                    messages=converted_messages
                )

            logger.info(f"IBM Watson chat completed: {self.model_id}")

            usage = response.get("usage", {})
            if usage:
                logger.debug(f"IBM Watson tokens: input={usage.get('prompt_tokens')}, output={usage.get('completion_tokens')}, total={usage.get('total_tokens')}")

            try:
                return self._convert_response_to_assistant_message(response)
            except ToolCallExtractionError as e:
                if attempt >= self.MAX_EXTRACTION_RETRIES:
                    logger.error(f"Tool call extraction failed after {attempt} attempts: {e}")
                    raise

                logger.warning(f"Tool call extraction failed (attempt {attempt}): {e}")

                error_feedback = HumanMessage(
                    content=f"ERROR: Your tool call was malformed.\n{e}\nPlease retry with valid JSON arguments."
                )
                working_messages.append(error_feedback)

        raise RuntimeError("Unexpected end of invoke loop")

    # ========== MESSAGE CONVERSION ==========

    def _convert_messages(self, messages: List[StandardMessage]) -> List[dict]:
        """
        Convert standardized messages to IBM Watson format (OpenAI-compatible dicts).
        """
        converted = []

        for i, msg in enumerate(messages):
            logger.trace(f"msg[{i}] {format_message_for_trace(msg)}")

            if isinstance(msg, SystemMessage):
                converted.append({
                    "role": "system",
                    "content": msg.content
                })

            elif isinstance(msg, HumanMessage):
                converted.append({
                    "role": "user",
                    "content": msg.content
                })

            elif isinstance(msg, AssistantMessage):
                assistant_msg: Dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content

                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "type": "function",
                            "id": tc.id,
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                            }
                        }
                        for tc in msg.tool_calls
                    ]

                if not msg.content and not msg.tool_calls:
                    assistant_msg["content"] = ""

                converted.append(assistant_msg)

            elif isinstance(msg, ToolMessage):
                for result in msg.tool_results:
                    converted.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": str(result.result) if result.result is not None else (result.error or "No result")
                    })

        logger.debug(f"Converted {len(messages)} messages to {len(converted)} IBM Watson messages")
        return converted

    # ========== TOOL CONVERSION ==========

    def _convert_tools(self, tools: List[ToolBase]) -> List[dict]:
        """Convert tool list to OpenAI-compatible function format."""
        if not tools:
            return []

        converted = []
        for tool in tools:
            schema = tool.create_schema()
            converted.append({
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.inputSchema
                }
            })

        logger.debug(f"Converted {len(tools)} tools to IBM Watson format")
        return converted

    # ========== TOOL CALL EXTRACTION ==========

    def _extract_tool_calls(self, response: dict) -> List[ToolCall]:
        """
        Extract and validate tool calls from IBM Watson response.

        Path: response["choices"][0]["message"]["tool_calls"]
        IBM returns dict-based responses (not objects like OpenAI SDK).

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails
        """
        choices = response.get("choices")
        if not choices:
            return []

        message = choices[0].get("message", {})
        raw_tool_calls = message.get("tool_calls")
        if not raw_tool_calls:
            return []

        tool_calls = []
        errors = []

        for call in raw_tool_calls:
            if call.get("type") != "function":
                continue

            try:
                function = call.get("function", {})
                arguments = function.get("arguments", "{}")

                if isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                # Handle double-encoding
                while isinstance(arguments, str):
                    arguments = json.loads(arguments, strict=False)

                tool_call = ToolCall(
                    id=call["id"],
                    name=function["name"],
                    arguments=arguments
                )
                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                errors.append(
                    f"Tool '{call.get('function', {}).get('name', 'unknown')}': Invalid JSON in arguments - {e.msg}"
                )
            except (ValidationError, KeyError) as e:
                errors.append(
                    f"Tool '{call.get('function', {}).get('name', 'unknown')}': {e}"
                )

        if errors:
            raise ToolCallExtractionError(
                "Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    # ========== RESPONSE EXTRACTION ==========

    def _extract_content(self, response: dict) -> str:
        """Extract text content from IBM Watson response."""
        choices = response.get("choices")
        if not choices:
            return ""

        return choices[0].get("message", {}).get("content", "") or ""

    def _extract_usage(self, response: dict) -> Optional[Usage]:
        """Extract usage information from IBM Watson response."""
        usage = response.get("usage")
        if not usage:
            return None

        return Usage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0)
        )

    def _convert_response_to_assistant_message(self, response: dict) -> AssistantMessage:
        """
        Convert IBM Watson response to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction fails
        """
        tool_calls = self._extract_tool_calls(response)
        content = self._extract_content(response)
        usage = self._extract_usage(response)

        logger.debug(
            f"IBM Watson response: content_length={len(content)}, tool_calls={len(tool_calls)}"
        )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )
