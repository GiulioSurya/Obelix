# src/llm_providers/oci_provider.py
import logging
from typing import List, Optional

try:
    import oci
    from oci.generative_ai_inference.models import (
        ChatDetails,
        OnDemandServingMode,
    )
except ImportError:
    raise ImportError(
        "oci is not installed. Install with: pip install oci"
    )

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.messages.usage import Usage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import OCIConnection
from src.logging_config import get_logger

# Import strategies
from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.llm_providers.oci_strategies.generic_strategy import GenericRequestStrategy
from src.llm_providers.oci_strategies.cohere_strategy import CohereRequestStrategy

# Logger per trace delle chiamate OCI
logger = get_logger(__name__)

class OCILLm(AbstractLLMProvider):
    """Provider per OCI Generative AI con parametri configurabili"""

    # Available strategies
    _STRATEGIES = [
        GenericRequestStrategy(),
        CohereRequestStrategy()
    ]

    @property
    def provider_type(self) -> Providers:
        return Providers.OCI_GENERATIVE_AI

    def __init__(self,
                 connection: Optional[OCIConnection] = None,
                 model_id: str = "meta.llama-3.3-70b-instruct",
                 max_tokens: int = 3500,
                 temperature: float = 0,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 stop_sequences: Optional[List[str]] = None,
                 is_stream: bool = False,
                 strategy: Optional[OCIRequestStrategy] = None,
                 logger: bool = False,
                 **strategy_kwargs):
        """
        Inizializza il provider OCI con dependency injection della connection

        Args:
            connection: OCIConnection singleton (default: None, riusa da GlobalConfig se provider match)
            model_id: ID del modello OCI (default: "meta.llama-3.3-70b-instruct")
            max_tokens: Numero massimo di token (default: 2000)
            temperature: Temperatura per sampling (default: 0.1)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            frequency_penalty: Penalità frequenza (default: None)
            presence_penalty: Penalità presenza (default: None)
            stop_sequences: Sequenze di stop (default: None)
            is_stream: Abilita streaming (default: False)
            strategy: Strategy specifica da usare (default: auto-detect dal model_id)
            **strategy_kwargs: Parametri specifici per la strategia selezionata
                Generic: reasoning_effort, verbosity, num_generations, log_probs, etc.
                Cohere: preamble_override, safety_mode, documents, citation_quality, etc.

        Raises:
            ValueError: Se connection=None e GlobalConfig non ha OCI_GENERATIVE_AI settato
        """
        # Configura logging OCI SDK se abilitato in infrastructure.yaml


        if logger is True:
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger('oci').setLevel(logging.DEBUG)
            oci.base_client.is_http_log_enabled(True)
        else:
            # Disabilita logging OCI SDK se non richiesto
            logging.getLogger('oci').setLevel(logging.WARNING)
            oci.base_client.is_http_log_enabled(False)

        # Dependency injection della connection con fallback a GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.OCI_GENERATIVE_AI,
                "OCILLm"
            )

        self.connection = connection

        # Salva parametri per uso nelle chiamate
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences
        self.is_stream = is_stream
        self.strategy_kwargs = strategy_kwargs

        # Auto-detect or use provided strategy
        self.strategy = strategy if strategy else self._detect_strategy(model_id)

    @classmethod
    def _detect_strategy(cls, model_id: str) -> OCIRequestStrategy:
        """
        Auto-detect the appropriate strategy based on model_id prefix.

        Args:
            model_id: The OCI model identifier (e.g., "meta.llama-3.3-70b-instruct")

        Returns:
            OCIRequestStrategy: The appropriate strategy for the model

        Raises:
            ValueError: If no strategy supports the model_id prefix
        """
        for strategy in cls._STRATEGIES:
            for prefix in strategy.get_supported_model_prefixes():
                if model_id.startswith(prefix):
                    return strategy

        raise ValueError(
            f"No strategy found for model_id '{model_id}'. "
            f"Supported prefixes: {[p for s in cls._STRATEGIES for p in s.get_supported_model_prefixes()]}"
        )

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello OCI con messaggi e tool standardizzati.
        Usa la strategia appropriata (auto-detected o specificata) per tutto il flusso.
        """
        # 1. La strategy converte messaggi e tool nel formato provider-specific
        converted_messages = self.strategy.convert_messages(messages)
        converted_tools = self.strategy.convert_tools(tools)

        logger.debug(f"OCI invoke: model={self.model_id}, messages={len(converted_messages)}, tools={len(converted_tools)}")

        # 2. La strategy costruisce la richiesta specifica (Generic o Cohere)
        chat_request = self.strategy.build_request(
            converted_messages=converted_messages,
            converted_tools=converted_tools,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop_sequences=self.stop_sequences,
            is_stream=self.is_stream,
            **self.strategy_kwargs
        )

        # 3. Chiama OCI usando il client dalla connection
        from src.k8s_config import YamlConfig
        import os
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_config = infra_config.get("llm_providers.oci")

        client = self.connection.get_client()

        try:
            response = client.chat(
                chat_details=ChatDetails(
                    compartment_id=oci_config["compartment_id"],
                    serving_mode=OnDemandServingMode(model_id=self.model_id),
                    chat_request=chat_request
                )
            )
            logger.info(f"OCI chat completed: {response.data.model_id}")
            logger.debug(f"OCI response tokens: {getattr(response.data.chat_response.usage, 'total_tokens', 'N/A')}")
        except Exception as e:
            # Log errore con dump messaggi per debug "Unsafe Text detected"
            logger.error(f"OCI request failed: {e}")
            for i, msg in enumerate(converted_messages):
                logger.error(f"msg[{i}]: {getattr(msg, 'content', 'N/A')}")
            raise

        # 4. Converte response in AssistantMessage standardizzato
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Converte risposta OCI in AssistantMessage standardizzato.
        Usa la strategy corrente per estrarre tool_calls con il formato giusto.
        """
        # Usa il mapping della strategy per estrarre tool_calls (Generic o Cohere)
        mapping = self.strategy.get_mapping()

        tool_calls = mapping["tool_output"]["tool_calls"](response)

        # Estrae il contenuto testuale
        # GENERIC: content in choices[0].message.content
        # COHERE: content in chat_response.text
        content = ""

        # Try Generic format first
        if (hasattr(response, 'data') and
                hasattr(response.data, 'chat_response') and
                hasattr(response.data.chat_response, 'choices') and
                response.data.chat_response.choices and
                response.data.chat_response.choices[0].message.content):
            content = "".join([
                c.text for c in response.data.chat_response.choices[0].message.content
                if hasattr(c, 'text') and c.type == "TEXT"
            ])
        # Try Cohere format
        elif (hasattr(response, 'data') and
              hasattr(response.data, 'chat_response') and
              hasattr(response.data.chat_response, 'text')):
            content = response.data.chat_response.text

        # Estrae usage dalla response OCI
        usage = None
        try:
            usage_data = response.data.chat_response.usage
            usage = Usage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens
            )
        except AttributeError:
            # Se non riesce a estrarre usage, continua senza (usage rimane None)
            pass

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls,
            usage=usage
        )
