# src/llm_providers/ibm_provider.py
from typing import List, Dict, Any, Optional

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.connections.llm_connection import IBMConnection
from src.logging_config import get_logger

try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
except ImportError:
    raise ImportError(
        "ibm-watsonx-ai is not installed. Install with: pip install ibm-watsonx-ai"
    )

# Logger per IBM Watson X provider
logger = get_logger(__name__)

#mistralai/mistral-large
#meta-llama/llama-3-3-70b-instruct
#['ibm/granite-13b-instruct-v2', 'ibm/granite-3-2b-instruct', 'ibm/granite-3-3-8b-instruct', 'ibm/granite-3-8b-instruct', 'ibm/granite-4-h-small', 'meta-llama/llama-2-13b-chat', 'meta-llama/llama-3-2-11b-vision-instruct', 'meta-llama/llama-3-2-90b-vision-instruct', 'meta-llama/llama-3-3-70b-instruct', 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8', 'mistralai/mistral-medium-2505', 'mistralai/mistral-small-3-1-24b-instruct-2503', 'sdaia/allam-1-13b-instruct']
class IBMWatsonXLLm(AbstractLLMProvider):
    """Provider per IBM Watson X con parametri configurabili"""

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
        Inizializza il provider IBM Watson X con dependency injection della connection

        Args:
            connection: IBMConnection singleton (default: None, riusa da GlobalConfig se provider match)
            model_id: ID del modello (default: "meta-llama/llama-3-3-70b-instruct")
            max_tokens: Numero massimo di token (default: 3000)
            temperature: Temperatura per sampling (default: 0.3)
            top_p: Top-p sampling (default: None)
            seed: Seed per riproducibilità (default: None)
            stop: Sequenze di stop (default: None)
            frequency_penalty: Penalità frequenza token (default: None)
            presence_penalty: Penalità presenza token (default: None)
            logprobs: Restituisci log probabilities (default: None)
            top_logprobs: Numero top log probabilities (default: None)
            n: Numero di completamenti da generare (default: None)
            logit_bias: Bias per specifici token (default: None)

        Raises:
            ValueError: Se connection=None e GlobalConfig non ha IBM_WATSON settato
        """
        # Dependency injection della connection con fallback a GlobalConfig
        if connection is None:
            connection = self._get_connection_from_global_config(
                Providers.IBM_WATSON,
                "IBMWatsonXLLm"
            )

        self.connection = connection

        # Salva model_id
        self.model_id = model_id

        # Costruisci parametri
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

        # Crea ModelInference usando credenziali dalla connection
        credentials = self.connection.get_client()
        self.client = ModelInference(
            model_id=model_id,
            params=TextChatParameters(**params_dict),
            credentials=credentials,
            project_id=self.connection.get_project_id()
        )

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello IBM Watson con messaggi e tool standardizzati
        """
        logger.debug(f"IBM Watson invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}")

        # 1. Converte messaggi e tool nel formato IBM (usa metodi base class)
        ibm_messages = self._convert_messages_to_provider_format(messages)
        ibm_tools = self._convert_tools_to_provider_format(tools)

        # 2. Chiama IBM Watson
        # NOTA: tool_choice_option va passato SOLO se ci sono tools definiti
        try:
            if ibm_tools:
                response = self.client.chat(
                    messages=ibm_messages,
                    tools=ibm_tools,
                    tool_choice_option="auto"
                )
            else:
                response = self.client.chat(
                    messages=ibm_messages
                )

            logger.info(f"IBM Watson chat completed: {self.model_id}")

            # Log usage se disponibile
            usage = response.get("usage", {})
            if usage:
                logger.debug(f"IBM Watson tokens: input={usage.get('prompt_tokens')}, output={usage.get('completion_tokens')}, total={usage.get('total_tokens')}")

        except Exception as e:
            logger.error(f"IBM Watson request failed: {e}")
            raise

        # 3. Converte response in AssistantMessage standardizzato
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Converte risposta IBM Watson in AssistantMessage standardizzato
        """
        # Estrae tool_calls usando il metodo centralizzato
        tool_calls = self._extract_tool_calls(response)

        # Estrae il contenuto testuale
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

        logger.debug(f"IBM Watson response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
