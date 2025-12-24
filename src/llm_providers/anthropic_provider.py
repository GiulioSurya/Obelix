# src/llm_providers/anthropic_provider.py
from typing import List, Dict, Any, Optional

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.messages.system_message import SystemMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import AnthropicConnection
from src.logging_config import get_logger

# Logger per Anthropic provider
logger = get_logger(__name__)


class AnthropicProvider(AbstractLLMProvider):
    """
    Provider per Anthropic Claude con parametri configurabili.

    Supporta:
    - System messages (come parametro separato, non nell'array messages)
    - Tool calling con content blocks
    - Multi-turn conversations
    - Usage tracking
    """

    @property
    def provider_type(self) -> Providers:
        return Providers.ANTHROPIC

    def __init__(self,
                 connection: Optional[AnthropicConnection] = None,
                 model_id: str = "claude-haiku-4-5-20251001",
                 max_tokens: int = 3000,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 thinking_mode: bool = False,
                 thinking_params: Optional[Dict[str, Any]] = None):
        """
        Inizializza il provider Anthropic con dependency injection della connection

        Args:
            connection: AnthropicConnection singleton (default: None, riusa da GlobalConfig se provider match)
            model_id: ID del modello Claude (default: "claude-haiku-4-5-20251001")
            max_tokens: Numero massimo di token (default: 3000)
            temperature: Temperatura per sampling (default: 0.1)
            top_p: Top-p sampling (default: None)
            thinking_mode: Abilita extended thinking (default: False)
            thinking_params: Parametri per thinking mode (default: None)

        Raises:
            ValueError: Se connection=None e GlobalConfig non ha ANTHROPIC settato
        """
        # Dependency injection della connection con fallback a GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.ANTHROPIC,
                "AnthropicProvider"
            )

        self.connection = connection

        # Salva parametri
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = 1 if thinking_mode else temperature
        self.top_p = top_p
        self.thinking_mode = thinking_mode
        if thinking_mode:
            if thinking_params is None:
                print("default parameters: {'type': 'enabled', 'budget_tokens': 2000}")
                self.thinking_params = {"type": "enabled", "budget_tokens": 2000}
            else:
                self.thinking_params = thinking_params
        else:
            self.thinking_params = {"type": "disabled"}

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello Anthropic con messaggi e tool standardizzati
        """
        logger.debug(f"Anthropic invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}, thinking_mode={self.thinking_mode}")

        # 1. Separa SystemMessage dagli altri (Anthropic lo vuole come parametro, non in messages array)
        system_content = None
        other_messages = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # SystemMessage va come parametro system, converti singolarmente
                system_content = self._convert_messages_to_provider_format([msg])[0]
            else:
                other_messages.append(msg)

        # 2. Converte gli altri messaggi (usa metodo base class)
        conversation_messages = self._convert_messages_to_provider_format(other_messages)

        # 3. Converte tools nel formato Anthropic (usa metodo base class)
        anthropic_tools = self._convert_tools_to_provider_format(tools) if tools else None

        # 4. Costruisce parametri chiamata API
        api_params: Dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": conversation_messages,
            "thinking": self.thinking_params
        }

        if system_content:
            api_params["system"] = system_content

        if anthropic_tools:
            api_params["tools"] = anthropic_tools

        # 5. Chiama Anthropic API usando il client dalla connection
        try:
            client = self.connection.get_client()
            response = client.messages.create(**api_params)

            logger.info(f"Anthropic chat completed: {self.model_id}")

            # Log usage
            if hasattr(response, 'usage'):
                logger.debug(f"Anthropic tokens: input={response.usage.input_tokens}, output={response.usage.output_tokens}, total={response.usage.input_tokens + response.usage.output_tokens}")

        except Exception as e:
            logger.error(f"Anthropic request failed: {e}")
            raise

        # 6. Converte response in AssistantMessage standardizzato
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Converte risposta Anthropic in AssistantMessage standardizzato
        """
        # Estrae tool_calls usando il metodo centralizzato
        tool_calls = self._extract_tool_calls(response)
        text_content = self._extract_text_from_content_blocks(response)
        usage = self._extract_usage_info(response)

        logger.debug(f"Anthropic response: content_length={len(text_content)}, tool_calls={len(tool_calls)}, content_blocks={len(response.content) if hasattr(response, 'content') else 0}")

        return AssistantMessage(
            content=text_content,
            tool_calls=tool_calls if tool_calls else [],
            usage=usage
        )

    def _extract_text_from_content_blocks(self, response) -> str:
        """
        Estrae il contenuto testuale dai content blocks della response

        Gestisce sia dict che object notation per compatibilità.

        Args:
            response: Response object da Anthropic API

        Returns:
            Contenuto testuale concatenato dai blocks di tipo "text"
        """
        if not hasattr(response, "content"):
            return ""

        text_content = ""
        for block in response.content:
            block_type = self._get_block_attribute(block, "type")
            if block_type == "text":
                text = self._get_block_attribute(block, "text")
                text_content += text

        return text_content

    def _get_block_attribute(self, block, attribute: str) -> Any:
        """
        Helper per ottenere attributo da block gestendo dict e object notation

        Args:
            block: Content block (può essere dict o object)
            attribute: Nome dell'attributo da estrarre

        Returns:
            Valore dell'attributo o None se non trovato
        """
        if isinstance(block, dict):
            return block.get(attribute)
        return getattr(block, attribute, None)

    def _extract_usage_info(self, response) -> Optional[Usage]:
        """
        Estrae informazioni di usage dalla response

        Args:
            response: Response object da Anthropic API

        Returns:
            Usage object se disponibile, None altrimenti
        """
        if not hasattr(response, "usage"):
            return None

        return Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens
        )
