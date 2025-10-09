# src/llm_providers/ibm_provider.py
from typing import List, Dict, Any, Optional

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.human_message import HumanMessage
from src.messages.system_message import SystemMessage
from src.messages.tool_message import ToolMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.mapping.provider_mapping import ProviderRegistry
from src.providers import Providers
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
import os
#mistralai/mistral-large
#meta-llama/llama-3-3-70b-instruct

class IBMWatsonXLLm(AbstractLLMProvider):
    """Provider per IBM Watson X con parametri configurabili"""

    def __init__(self,
                 model_id: str = "mistralai/mistral-large",
                 max_tokens: int = 1000,
                 temperature: float = 0.1,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 repetition_penalty: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 stop_sequences: Optional[List[str]] = None,
                 decoding_method: str = "greedy",
                 length_penalty: Optional[Dict[str, float]] = None):
        """
        Inizializza il provider IBM Watson X

        Args:
            model_id: ID del modello (default: "mistralai/mistral-large")
            max_tokens: Numero massimo di token (default: 1500)
            temperature: Temperatura per sampling (default: 0.1)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            repetition_penalty: Penalità per ripetizioni (default: None)
            random_seed: Seed per riproducibilità (default: None)
            stop_sequences: Sequenze di stop (default: None)
            decoding_method: Metodo decoding "greedy" o "sample" (default: "greedy")
            length_penalty: Penalità per lunghezza (default: None)
        """

        # Costruisce i parametri per TextChatParameters
        params_dict = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Aggiungi parametri opzionali solo se specificati
        if top_p is not None:
            params_dict["top_p"] = top_p
        if top_k is not None:
            params_dict["top_k"] = top_k
        if repetition_penalty is not None:
            params_dict["repetition_penalty"] = repetition_penalty
        if random_seed is not None:
            params_dict["random_seed"] = random_seed
        if stop_sequences is not None:
            params_dict["stop_sequences"] = stop_sequences
        if decoding_method != "greedy":
            params_dict["decoding_method"] = decoding_method
        if length_penalty is not None:
            params_dict["length_penalty"] = length_penalty

        self.client = ModelInference(
            model_id=model_id,
            params=TextChatParameters(**params_dict),
            credentials=Credentials(
                url="https://eu-de.ml.cloud.ibm.com",
                api_key=os.getenv("IBM_WATSONX_API_KEY"),
            ),
            project_id=os.getenv("IBM_WATSONX_PROJECT_ID")
        )

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello IBM Watson con messaggi e tool standardizzati
        """
        # 1. Converte messaggi e tool nel formato IBM
        ibm_messages = self._convert_messages_to_provider_format(messages)
        ibm_tools = self._convert_tools_to_provider_format(tools)

        # 2. Chiama IBM Watson
        response = self.client.chat(
            messages=ibm_messages,
            tools=ibm_tools,
            tool_choice_option="auto"
        )
        assistant_message = self._convert_response_to_assistant_message(response)
        # 3. Converte response in AssistantMessage standardizzato
        return assistant_message

    def _convert_messages_to_provider_format(self, messages: List[StandardMessage]) -> List[Dict[str, str]]:
        """
        Converte StandardMessage nel formato IBM Watson usando il mapping centralizzato
        """
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        message_converters = mapping["message_input"]

        converted_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                converted_messages.append(message_converters["human_message"](message))
            elif isinstance(message, SystemMessage):
                converted_messages.append(message_converters["system_message"](message))
            elif isinstance(message, AssistantMessage):
                converted_messages.append(message_converters["assistant_message"](message))
            elif isinstance(message, ToolMessage):
                # ToolMessage può generare multiple messages
                converted_messages.extend(message_converters["tool_message"](message))

        return converted_messages

    def _convert_tools_to_provider_format(self, tools: List[ToolBase]) -> List[Dict[str, Any]]:
        """
        Converte ToolBase nel formato IBM Watson
        """
        if not tools:
            return []

        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        tool_mapper = mapping["tool_input"]["tool_schema"]

        return [tool_mapper(tool.create_schema()) for tool in tools]

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Converte risposta IBM Watson in AssistantMessage standardizzato
        """
        # Estrae tool_calls usando il mapping esistente
        mapping = ProviderRegistry.get_mapping(Providers.IBM_WATSON)
        tool_calls = mapping["tool_output"]["tool_calls"](response)

        # Estrae il contenuto testuale
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )