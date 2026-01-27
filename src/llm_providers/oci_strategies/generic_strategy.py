# src/llm_providers/oci_strategies/generic_strategy.py
from typing import List, Any, Optional, Dict
from oci.generative_ai_inference.models import (
    GenericChatRequest,
    BaseChatRequest,
    Message,
    ToolChoiceAuto,
    ToolChoiceRequired,
    ToolChoiceNone
)

from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.messages.standard_message import StandardMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.tool_message import ToolMessage
from src.tools.tool_base import ToolBase
from src.mapping.provider_mapping import OCI_GENERATIVE_AI_GENERIC
from src.logging_config import get_logger, format_message_for_trace

# Logger for GenericRequestStrategy
logger = get_logger(__name__)


def _inline_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inline JSON Schema $ref using local $defs to avoid OCI Generic rejection.

    OCI Generic (Gemini) rejects $defs/$ref in tool parameters, so we resolve
    local references and drop $defs in the output.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    key = ref.split("/")[-1]
                    resolved = defs.get(key, {})
                    return resolve(resolved)
                return node
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


class GenericRequestStrategy(OCIRequestStrategy):
    """
    Strategy for models using the GENERIC API format.
    Supports: Meta Llama, Google Gemini, xAI Grok, OpenAI GPT-OSS
    """

    def convert_messages(self, messages: List[StandardMessage]) -> List[Message]:
        """
        Convert StandardMessage objects to Generic format (UserMessage, SystemMessage, etc.)
        """
        logger.debug(f"Converting {len(messages)} messages to OCI GENERIC format")

        mapping = self.get_mapping()
        message_converters = mapping["message_input"]

        converted_messages = []
        for i, message in enumerate(messages):
            msg_type = type(message).__name__

            # TRACE: full message content with tool_calls/tool_results
            logger.trace(f"msg[{i}] {format_message_for_trace(message)}")

            if isinstance(message, HumanMessage):
                converted_messages.append(message_converters["human_message"](message))
            elif isinstance(message, SystemMessage):
                converted_messages.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                converted_messages.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                converted_messages.extend(message_converters["tool_message"](message))

        logger.debug(f"Converted {len(converted_messages)} messages to OCI GENERIC format")
        return converted_messages

    def convert_tools(self, tools: List[ToolBase]) -> List[Any]:
        """
        Convert ToolBase objects to FunctionDefinition format
        """
        if not tools:
            logger.debug("No tools to convert for OCI GENERIC format")
            return []

        logger.debug(f"Converting {len(tools)} tools to OCI GENERIC format")

        mapping = self.get_mapping()
        tool_mapper = mapping["tool_input"]["tool_schema"]

        converted_tools = []
        for tool in tools:
            schema = tool.create_schema()
            if isinstance(schema.inputSchema, dict) and schema.inputSchema:
                schema.inputSchema = _inline_schema_refs(schema.inputSchema)
            converted_tools.append(tool_mapper(schema))
        logger.debug(f"Converted {len(converted_tools)} tools to OCI GENERIC format")

        return converted_tools

    def build_request(
        self,
        converted_messages: List[Message],
        converted_tools: List[Any],
        max_tokens: int,
        temperature: float,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        is_stream: bool = False,
        **kwargs
    ) -> BaseChatRequest:
        """
        Build a GenericChatRequest for Llama, Gemini, Grok, and OpenAI models.

        Args:
            converted_messages: List[Message] from convert_messages()
            converted_tools: List[FunctionDefinition] from convert_tools()

        Additional kwargs supported:
            - reasoning_effort: str (MINIMAL, LOW, MEDIUM, HIGH)
            - verbosity: str (LOW, MEDIUM, HIGH)
            - num_generations: int
            - log_probs: bool
            - logit_bias: dict
            - is_parallel_tool_calls: bool
            - seed: int
            - metadata: dict
            - tool_choice: str or ToolChoice object (AUTO, REQUIRED, NONE)
        """
        request_params = {
            "api_format": BaseChatRequest.API_FORMAT_GENERIC,
            "messages": converted_messages,
            "tools": converted_tools,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Add standard optional parameters
        if top_p is not None:
            request_params["top_p"] = top_p
        if top_k is not None:
            request_params["top_k"] = top_k
        if frequency_penalty is not None:
            request_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            request_params["presence_penalty"] = presence_penalty
        if stop_sequences is not None:
            request_params["stop"] = stop_sequences
        if is_stream:
            request_params["is_stream"] = is_stream

        # Add GENERIC-specific parameters from kwargs
        generic_specific_params = [
            "reasoning_effort",
            "verbosity",
            "num_generations",
            "log_probs",
            "logit_bias",
            "is_parallel_tool_calls",
            "seed",
            "metadata"
        ]

        for param in generic_specific_params:
            if param in kwargs and kwargs[param] is not None:
                request_params[param] = kwargs[param]

        # Handle tool_choice conversion from string to OCI ToolChoice object
        if "tool_choice" in kwargs and kwargs["tool_choice"] is not None:
            tool_choice_value = kwargs["tool_choice"]
            if isinstance(tool_choice_value, str):
                # Convert string to appropriate ToolChoice object
                tool_choice_map = {
                    "AUTO": ToolChoiceAuto(),
                    "REQUIRED": ToolChoiceRequired(),
                    "NONE": ToolChoiceNone()
                }
                if tool_choice_value.upper() in tool_choice_map:
                    request_params["tool_choice"] = tool_choice_map[tool_choice_value.upper()]
                else:
                    raise ValueError(
                        f"Invalid tool_choice value: '{tool_choice_value}'. "
                        f"Expected one of: AUTO, REQUIRED, NONE"
                    )
            else:
                # Already a ToolChoice object, use as-is
                request_params["tool_choice"] = tool_choice_value

        return GenericChatRequest(**request_params)

    def get_mapping(self) -> Dict[str, Any]:
        """Returns the OCI_GENERATIVE_AI_GENERIC mapping"""
        return OCI_GENERATIVE_AI_GENERIC

    def get_api_format(self) -> str:
        """Returns API_FORMAT_GENERIC"""
        return BaseChatRequest.API_FORMAT_GENERIC

    def get_supported_model_prefixes(self) -> List[str]:
        """
        Returns model ID prefixes supported by GENERIC format.

        Supported models:
        - meta.* (Llama 3, 3.1, 3.2, 3.3, 4)
        - google.* (Gemini 2.5 Pro, Flash, Flash-Lite)
        - xai.* (Grok 3, 3 Mini, 4, Code Fast)
        - openai.* (GPT-OSS models)
        """
        return ["meta.", "google.", "xai.", "openai."]
