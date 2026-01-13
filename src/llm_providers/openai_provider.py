# src/llm_providers/openai_provider.py
from typing import List, Dict, Any, Optional

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.messages.system_message import SystemMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection.openai_connection import OpenAIConnection
from src.logging_config import get_logger

# Logger for OpenAI provider
logger = get_logger(__name__)


class OpenAIProvider(AbstractLLMProvider):
    """
    Provider for OpenAI GPT models with configurable parameters.

    Supports:
    - System messages (as first message in array with role="system")
    - Tool calling with function format
    - Multi-turn conversations
    - Usage tracking
    """

    @property
    def provider_type(self) -> Providers:
        return Providers.OPENAI

    def __init__(self,
                 connection: Optional[OpenAIConnection] = None,
                 model_id: str = "gpt-4o",
                 max_tokens: int = 4096,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None):
        """
        Initialize the OpenAI provider with dependency injection of connection

        Args:
            connection: OpenAIConnection singleton (default: None, reuse from GlobalConfig if provider matches)
            model_id: OpenAI model ID (default: "gpt-4o")
            max_tokens: Maximum number of tokens (default: 4096)
            temperature: Sampling temperature (default: 0.1)
            top_p: Top-p sampling (default: None)

        Raises:
            ValueError: If connection=None and GlobalConfig does not have OPENAI set
        """
        # Dependency injection of connection with fallback to GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OPENAI,
                "OpenAIProvider"
            )

        self.connection = connection

        # Save parameters
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Calls the OpenAI model with standardized messages and tools
        """
        logger.debug(f"OpenAI invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}")

        # 1. Convert messages to OpenAI format (system message included in array)
        openai_messages = self._convert_messages_to_provider_format(messages)

        # 2. Convert tools to OpenAI format
        openai_tools = self._convert_tools_to_provider_format(tools) if tools else None

        # 3. Build API call parameters
        api_params: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": openai_messages,
        }

        if self.top_p is not None:
            api_params["top_p"] = self.top_p

        if openai_tools:
            api_params["tools"] = openai_tools

        # 4. Call OpenAI API using client from connection
        try:
            client = self.connection.get_client()
            response = client.chat.completions.create(**api_params)

            logger.info(f"OpenAI chat completed: {self.model_id}")

            # Log usage
            if hasattr(response, 'usage') and response.usage:
                logger.debug(f"OpenAI tokens: input={response.usage.prompt_tokens}, output={response.usage.completion_tokens}, total={response.usage.total_tokens}")

        except Exception as e:
            logger.error(f"OpenAI request failed: {e}")
            raise

        # 5. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Convert OpenAI response to standardized AssistantMessage
        """
        # Extract tool_calls using centralized method
        tool_calls = self._extract_tool_calls(response)
        text_content = self._extract_text_content(response)
        usage = self._extract_usage_info(response)

        logger.debug(f"OpenAI response: content_length={len(text_content) if text_content else 0}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=text_content,
            tool_calls=tool_calls if tool_calls else [],
            usage=usage
        )

    def _extract_text_content(self, response) -> str:
        """
        Extract text content from OpenAI response

        Args:
            response: Response object from OpenAI API

        Returns:
            Text content from the message
        """
        if not hasattr(response, 'choices') or not response.choices:
            return ""

        message = response.choices[0].message
        return message.content if message.content else ""

    def _extract_usage_info(self, response) -> Optional[Usage]:
        """
        Extract usage information from response

        Args:
            response: Response object from OpenAI API

        Returns:
            Usage object if available, None otherwise
        """
        if not hasattr(response, 'usage') or not response.usage:
            return None

        return Usage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens
        )
