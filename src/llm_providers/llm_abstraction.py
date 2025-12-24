# src/llm_providers/llm_abstraction.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from src.messages.standard_message import StandardMessage
from src.messages.assistant_message import AssistantMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.tool_message import ToolMessage
from src.tools.tool_base import ToolBase
from src.logging_config import get_logger

if TYPE_CHECKING:
    from src.providers import Providers

# Logger per trace operazioni comuni provider
logger = get_logger(__name__)


class AbstractLLMProvider(ABC):
    """
    Classe base astratta per i provider LLM.

    Fornisce:
    - Metodi comuni per conversione messaggi e tools (usando ProviderRegistry)
    - Helper per estrazione tool_calls
    - Metodo centralizzato per ottenere connection da GlobalConfig
    - Interfaccia pubblica obbligatoria: invoke()

    Ogni provider deve:
    - Implementare `provider_type` property
    - Implementare `invoke()` method
    - (Opzionale) Override metodi di conversione se usa pattern diversi (es. strategy)
    """

    @property
    @abstractmethod
    def provider_type(self) -> "Providers":
        """
        Ritorna l'enum Providers di questo provider.
        Usato per ottenere il mapping corretto da ProviderRegistry.
        """
        pass

    @abstractmethod
    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Chiama il modello LLM con messaggi e tool standardizzati

        Args:
            messages: Lista di messaggi in formato StandardMessage
            tools: Lista di tool disponibili

        Returns:
            AssistantMessage con la risposta del modello (include campo usage popolato)
        """
        pass

    # ========== METODO CENTRALIZZATO PER CONNECTION ==========

    @staticmethod
    def _get_connection_from_global_config(
        expected_provider: "Providers",
        provider_class_name: str
    ) -> Any:
        """
        Logica comune per ottenere connection da GlobalConfig.

        Verifica che GlobalConfig abbia il provider corretto settato,
        altrimenti solleva ValueError con istruzioni chiare.

        Args:
            expected_provider: Il Providers enum atteso (es. Providers.OCI_GENERATIVE_AI)
            provider_class_name: Nome della classe per messaggi di errore (es. "OCILLm")

        Returns:
            Connection dal GlobalConfig

        Raises:
            ValueError: Se GlobalConfig non ha provider settato o ha provider diverso
        """
        from src.config import GlobalConfig

        logger.debug(f"Getting connection from GlobalConfig for {provider_class_name} (expected: {expected_provider.name})")

        config = GlobalConfig()

        # Verifica che ci sia un provider settato
        try:
            current_provider = config.get_current_provider()
        except ValueError:
            logger.error(f"GlobalConfig has no provider set, cannot create {provider_class_name}")
            raise ValueError(
                f"Impossibile creare {provider_class_name} senza connection. "
                "Opzioni: \n"
                f"1. Passa connection esplicitamente\n"
                f"2. Setta GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        # Verifica che sia il provider atteso
        if current_provider != expected_provider:
            logger.error(
                f"Provider mismatch: GlobalConfig has {current_provider.value}, "
                f"but creating {provider_class_name} (expected {expected_provider.value})"
            )
            raise ValueError(
                f"GlobalConfig ha provider '{current_provider.value}' settato, "
                f"ma stai creando {provider_class_name}.\n"
                f"Opzioni:\n"
                f"1. Passa connection esplicitamente\n"
                f"2. Cambia GlobalConfig: GlobalConfig().set_provider(Providers.{expected_provider.name})"
            )

        # Riusa connection dal GlobalConfig (lazy init se necessario)
        if current_provider not in config._connections:
            logger.debug(f"Creating new connection for {current_provider.value}")
            config._connections[current_provider] = config._create_connection(current_provider)
        else:
            logger.debug(f"Reusing existing connection for {current_provider.value}")

        return config._connections[current_provider]

    # ========== METODI COMUNI PER CONVERSIONE ==========

    def _convert_messages_to_provider_format(self, messages: List[StandardMessage]) -> List[Any]:
        """
        Converte StandardMessage nel formato provider-specific usando ProviderRegistry.

        Questo metodo implementa il pattern comune usato da IBM, Ollama, vLLM, Anthropic.
        Provider con logica diversa (es. OCI con strategy) possono fare override.

        Args:
            messages: Lista di StandardMessage

        Returns:
            Lista di messaggi nel formato del provider
        """
        from src.providers import ProviderRegistry

        logger.debug(f"Converting {len(messages)} messages to {self.provider_type.value} format")

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        message_converters = mapping["message_input"]

        converted_messages = []

        for i, message in enumerate(messages):
            msg_type = type(message).__name__

            # TRACE: preview del contenuto messaggio
            content_preview = ""
            if hasattr(message, 'content') and message.content:
                content_preview = str(message.content)[:100]
            logger.trace(f"msg[{i}] {msg_type}: {content_preview}")

            if isinstance(message, HumanMessage):
                converted_messages.append(message_converters["human_message"](message))
            elif isinstance(message, SystemMessage):
                converted_messages.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                converted_messages.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                # ToolMessage può generare multiple messages
                converted_messages.extend(message_converters["tool_message"](message))

        logger.debug(f"Converted {len(converted_messages)} messages for {self.provider_type.value}")
        return converted_messages

    def _convert_tools_to_provider_format(self, tools: List[ToolBase]) -> List[Any]:
        """
        Converte ToolBase nel formato provider-specific usando ProviderRegistry.

        Questo metodo implementa il pattern comune usato da IBM, Ollama, vLLM, Anthropic.
        Provider con logica diversa (es. OCI con strategy) possono fare override.

        Args:
            tools: Lista di ToolBase

        Returns:
            Lista di tool nel formato del provider (vuota se tools è vuoto)
        """
        if not tools:
            logger.debug("No tools to convert")
            return []

        logger.debug(f"Converting {len(tools)} tools to {self.provider_type.value} format")

        from src.providers import ProviderRegistry

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        tool_mapper = mapping["tool_input"]["tool_schema"]

        converted_tools = [tool_mapper(tool.create_schema()) for tool in tools]
        logger.debug(f"Converted {len(converted_tools)} tools for {self.provider_type.value}")

        return converted_tools

    def _extract_tool_calls(self, response: Any, **kwargs) -> List[Dict[str, Any]]:
        """
        Estrae tool_calls dalla response usando il mapping del provider.

        Args:
            response: Response object dal provider API
            **kwargs: Parametri aggiuntivi (es. tools per vLLM)

        Returns:
            Lista di tool calls estratti
        """
        logger.debug(f"Extracting tool_calls from {self.provider_type.value} response")

        from src.providers import ProviderRegistry

        mapping = ProviderRegistry.get_mapping(self.provider_type)
        extractor = mapping["tool_output"]["tool_calls"]

        # Alcuni extractor richiedono parametri aggiuntivi (es. vLLM richiede tools)
        if kwargs:
            tool_calls = extractor(response, **kwargs)
        else:
            tool_calls = extractor(response)

        logger.debug(f"Extracted {len(tool_calls)} tool_calls")
        return tool_calls
