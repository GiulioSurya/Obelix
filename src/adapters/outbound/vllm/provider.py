# src/adapters/outbound/vllm/provider.py
"""
vLLM Provider.

Self-contained provider with inline message/tool conversion.
No external mapping dependencies.

Offline inference engine - no HTTP calls, no transport retry needed.
Uses asyncio.to_thread() to run the sync vLLM engine without blocking.
Tool calls are text-based: parsed from JSON in generated output text.
"""
import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional

from pydantic import ValidationError

from src.ports.outbound.llm_provider import AbstractLLMProvider
from src.domain.model import SystemMessage, HumanMessage, AssistantMessage, ToolMessage, StandardMessage
from src.domain.model.tool_message import ToolCall
from src.domain.tool.tool_base import ToolBase
from src.infrastructure.providers import Providers
from src.infrastructure.logging import get_logger, format_message_for_trace

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    raise ImportError(
        "vLLM is not installed. Install with: pip install vllm"
    )

logger = get_logger(__name__)


class ToolCallExtractionError(Exception):
    """Raised when tool call extraction or validation fails."""
    pass


class VLLMProvider(AbstractLLMProvider):
    """
    Provider for vLLM offline inference with configurable parameters.

    Self-contained: all conversion logic is inline.
    Uses OpenAI-compatible dict format for messages.
    Tool calls are extracted from generated text (JSON parsing).
    """

    MAX_EXTRACTION_RETRIES = 3

    @property
    def provider_type(self) -> Providers:
        return Providers.VLLM

    def __init__(self,
                 model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = 2000,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 repetition_penalty: Optional[float] = None,
                 min_p: Optional[float] = None,
                 # vLLM engine parameters
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 quantization: Optional[str] = None,
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 trust_remote_code: bool = False,
                 **kwargs):
        """
        Initialize the vLLM provider for offline inference.

        Args:
            model_id: Model ID to load (default: "meta-llama/Llama-3.2-1B-Instruct")
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum number of tokens to generate (default: 2000)
            top_p: Top-p (nucleus) sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            frequency_penalty: Penalty for token frequency (default: None)
            presence_penalty: Penalty for token presence (default: None)
            repetition_penalty: Penalty for repetitions (default: None)
            min_p: Minimum probability threshold (default: None)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
            dtype: Data type for model weights (default: "auto")
            quantization: Quantization method (e.g. "awq", "gptq") (default: None)
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
            max_model_len: Maximum context length (default: None, uses model config)
            trust_remote_code: Allow execution of remote code (default: False)
            **kwargs: Other parameters to pass to vLLM engine
        """
        self.model_id = model_id

        # Initialize the vLLM model
        llm_kwargs: Dict[str, Any] = {
            "model": model_id,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
        }

        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        llm_kwargs.update(kwargs)

        self.llm = LLM(**llm_kwargs)

        # Build sampling parameters
        sampling_params_dict: Dict[str, Any] = {}

        if temperature is not None:
            sampling_params_dict["temperature"] = temperature
        if max_tokens is not None:
            sampling_params_dict["max_tokens"] = max_tokens
        if top_p is not None:
            sampling_params_dict["top_p"] = top_p
        if top_k is not None:
            sampling_params_dict["top_k"] = top_k
        if seed is not None:
            sampling_params_dict["seed"] = seed
        if stop is not None:
            sampling_params_dict["stop"] = stop
        if frequency_penalty is not None:
            sampling_params_dict["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            sampling_params_dict["presence_penalty"] = presence_penalty
        if repetition_penalty is not None:
            sampling_params_dict["repetition_penalty"] = repetition_penalty
        if min_p is not None:
            sampling_params_dict["min_p"] = min_p

        self.sampling_params = SamplingParams(**sampling_params_dict)

    # ========== INVOKE ==========

    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Call the vLLM model with standardized messages and tools.

        Uses asyncio.to_thread() to run the sync vLLM engine without
        blocking the event loop. Offline inference - no transport retry needed.
        If tool call extraction fails, retries with error feedback to LLM.
        """
        working_messages = list(messages)
        converted_tools = self._convert_tools(tools)

        for attempt in range(1, self.MAX_EXTRACTION_RETRIES + 1):
            logger.debug(f"vLLM invoke: model={self.model_id}, messages={len(working_messages)}, tools={len(converted_tools)}, attempt={attempt}")

            converted_messages = self._convert_messages(working_messages)

            if converted_tools:
                outputs = await asyncio.to_thread(
                    self.llm.chat,
                    converted_messages,
                    sampling_params=self.sampling_params,
                    tools=converted_tools,
                    use_tqdm=False
                )
            else:
                outputs = await asyncio.to_thread(
                    self.llm.chat,
                    converted_messages,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )

            logger.info(f"vLLM chat completed: {self.model_id}")

            if outputs and hasattr(outputs[0], 'metrics'):
                logger.debug(f"vLLM metrics: {outputs[0].metrics}")

            try:
                return self._convert_response_to_assistant_message(outputs[0], has_tools=bool(converted_tools))
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
        Convert standardized messages to vLLM format (OpenAI-compatible dicts).
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

        logger.debug(f"Converted {len(messages)} messages to {len(converted)} vLLM messages")
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

        logger.debug(f"Converted {len(tools)} tools to vLLM format")
        return converted

    # ========== TOOL CALL EXTRACTION ==========

    def _extract_tool_calls(self, output, has_tools: bool) -> List[ToolCall]:
        """
        Extract tool calls from vLLM output by parsing JSON from generated text.

        vLLM outputs tool calls as JSON text in outputs[0].text.
        Supports: JSON array of tool calls, single tool call object.

        Raises:
            ToolCallExtractionError: If JSON parsing or validation fails
        """
        if not has_tools:
            return []

        content = self._extract_raw_text(output)
        if not content:
            return []

        # Try parsing as JSON (array or single object)
        try:
            parsed = json.loads(content, strict=False)
        except json.JSONDecodeError:
            # Try finding JSON within the text
            return self._extract_tool_calls_from_text(content)

        if isinstance(parsed, list):
            return self._parse_tool_call_list(parsed)

        if isinstance(parsed, dict) and "name" in parsed:
            return self._parse_single_tool_call(parsed)

        return []

    def _parse_tool_call_list(self, items: list) -> List[ToolCall]:
        """Parse a list of tool call dicts."""
        tool_calls = []
        errors = []

        for item in items:
            if not isinstance(item, dict) or "name" not in item:
                continue

            try:
                arguments = item.get("arguments", item.get("parameters", {}))
                tool_call = ToolCall(
                    id=item.get("id", str(uuid.uuid4())),
                    name=item["name"],
                    arguments=arguments
                )
                tool_calls.append(tool_call)
            except (ValidationError, KeyError) as e:
                errors.append(f"Tool '{item.get('name', 'unknown')}': {e}")

        if errors and not tool_calls:
            raise ToolCallExtractionError(
                "Failed to extract tool calls:\n" + "\n".join(f"  - {err}" for err in errors)
            )

        return tool_calls

    def _parse_single_tool_call(self, obj: dict) -> List[ToolCall]:
        """Parse a single tool call dict."""
        try:
            arguments = obj.get("arguments", obj.get("parameters", {}))
            tool_call = ToolCall(
                id=obj.get("id", str(uuid.uuid4())),
                name=obj["name"],
                arguments=arguments
            )
            return [tool_call]
        except (ValidationError, KeyError) as e:
            raise ToolCallExtractionError(
                f"Failed to extract tool call '{obj.get('name', 'unknown')}': {e}"
            )

    def _extract_tool_calls_from_text(self, content: str) -> List[ToolCall]:
        """Fallback: find and parse JSON object from text content."""
        start_pos = content.find('{')
        if start_pos == -1:
            return []

        json_content = content[start_pos:].replace("\\'", "'")

        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(json_content, 0)

            if isinstance(obj, dict) and "name" in obj:
                return self._parse_single_tool_call(obj)
        except json.JSONDecodeError:
            pass

        return []

    # ========== RESPONSE EXTRACTION ==========

    def _extract_raw_text(self, output) -> str:
        """Extract raw text from vLLM RequestOutput."""
        if hasattr(output, 'outputs') and output.outputs:
            first_output = output.outputs[0]
            if hasattr(first_output, 'text'):
                return first_output.text.strip()
        return ""

    def _convert_response_to_assistant_message(self, output, has_tools: bool) -> AssistantMessage:
        """
        Convert vLLM output to standardized AssistantMessage.

        Raises:
            ToolCallExtractionError: If tool call extraction fails
        """
        tool_calls = self._extract_tool_calls(output, has_tools)
        content = self._extract_raw_text(output) if not tool_calls else ""

        logger.debug(
            f"vLLM response: content_length={len(content)}, tool_calls={len(tool_calls)}"
        )

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )