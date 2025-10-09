# src/llm_providers/oci_provider.py
from oci.generative_ai_inference.models import (
    ChatDetails,
    OnDemandServingMode,
    GenericChatRequest,
    BaseChatRequest,
    SystemMessage, AssistantMessage, ToolMessage, Message)
from oci.generative_ai_inference import GenerativeAiInferenceClient
from typing import List, Any, Optional
import os

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.tool_message import ToolMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.mapping.provider_mapping import ProviderRegistry
from src.providers import Providers

# Import strategies
from src.llm_providers.oci_strategies.base_strategy import OCIRequestStrategy
from src.llm_providers.oci_strategies.generic_strategy import GenericRequestStrategy
from src.llm_providers.oci_strategies.cohere_strategy import CohereRequestStrategy


class OCILLm(AbstractLLMProvider):
    """Provider per OCI Generative AI con parametri configurabili"""

    # Available strategies
    _STRATEGIES = [
        GenericRequestStrategy(),
        CohereRequestStrategy()
    ]

    def __init__(self,
                 model_id: str = "meta.llama-3.3-70b-instruct",
                 max_tokens: int = 2000,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 frequency_penalty: Optional[float] = None,
                 presence_penalty: Optional[float] = None,
                 stop_sequences: Optional[List[str]] = None,
                 is_stream: bool = False,
                 strategy: Optional[OCIRequestStrategy] = None,
                 **strategy_kwargs):
        """
        Inizializza il provider OCI con auto-detection della strategia

        Args:
            model_id: ID del modello OCI (default: "meta.llama-3.3-70b-instruct")
            max_tokens: Numero massimo di token (default: 500)
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
        """

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

        # Inizializza client OCI
        oci_config = {
            'user': os.getenv('OCI_USER_ID'),
            'fingerprint': os.getenv('OCI_USER_FINGERPRINT'),
            'key_content': os.getenv('OCI_USER_KEY_CONTENT'),
            'tenancy': os.getenv('OCI_TENANCY'),
            'region': os.getenv('OCI_REGION'),
        }
        self.client = GenerativeAiInferenceClient(oci_config)

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

        # 3. Chiama OCI
        response = self.client.chat(
            chat_details=ChatDetails(
                compartment_id=os.getenv("OCI_COMPARTMENT_ID"),
                serving_mode=OnDemandServingMode(model_id=self.model_id),
                chat_request=chat_request
            )
        )

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

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )