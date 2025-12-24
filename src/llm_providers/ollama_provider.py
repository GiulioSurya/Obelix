# src/llm_providers/ollama_provider.py
from typing import List, Optional

try:
    from ollama import Client
except ImportError:
    raise ImportError(
        "ollama is not installed. Install with: pip install ollama"
    )

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.logging_config import get_logger

# Logger per Ollama provider
logger = get_logger(__name__)


class OllamaProvider(AbstractLLMProvider):
    """Provider per Ollama con parametri configurabili"""

    @property
    def provider_type(self) -> Providers:
        return Providers.OLLAMA

    def __init__(self,
                 model_id: str = "a-kore/Arctic-Text2SQL-R1-7B",
                 base_url: Optional[str] = None,
                 temperature: float = 0.1,
                 max_tokens: Optional[int] = 2000,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 seed: Optional[int] = None,
                 stop: Optional[List[str]] = None,
                 keep_alive: Optional[str] = None):
        """
        Inizializza il provider Ollama

        Args:
            model_id: ID del modello Ollama (default: "a-kore/Arctic-Text2SQL-R1-7B")
            base_url: URL base del server Ollama (default: None = http://localhost:11434)
            temperature: Temperatura per sampling (default: 0.1)
            max_tokens: Numero massimo di token (default: None)
            top_p: Top-p sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed per riproducibilità (default: None)
            stop: Sequenze di stop (default: None)
            keep_alive: Keep model in memory (default: None)
        """
        self.model_id = model_id

        # Inizializza client Ollama
        if base_url:
            self.client = Client(host=base_url)
        else:
            self.client = Client()

        # Costruisci options dict solo con parametri non-None
        self.options = {}
        if temperature is not None:
            self.options["temperature"] = temperature
        if max_tokens is not None:
            self.options["num_predict"] = max_tokens
        if top_p is not None:
            self.options["top_p"] = top_p
        if top_k is not None:
            self.options["top_k"] = top_k
        if seed is not None:
            self.options["seed"] = seed
        if stop is not None:
            self.options["stop"] = stop

        self.keep_alive = keep_alive

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello Ollama con messaggi e tool standardizzati
        """
        logger.debug(f"Ollama invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}")

        # 1. Converte messaggi e tool nel formato Ollama (usa metodi base class)
        ollama_messages = self._convert_messages_to_provider_format(messages)
        ollama_tools = self._convert_tools_to_provider_format(tools)

        # 2. Costruisci parametri chiamata
        chat_params = {
            "model": self.model_id,
            "messages": ollama_messages,
        }

        if ollama_tools:
            chat_params["tools"] = ollama_tools

        if self.options:
            chat_params["options"] = self.options

        if self.keep_alive is not None:
            chat_params["keep_alive"] = self.keep_alive

        # 3. Chiama Ollama
        try:
            response = self.client.chat(**chat_params)

            logger.info(f"Ollama chat completed: {self.model_id}")

            # Log usage se disponibile
            if hasattr(response, 'prompt_eval_count') and hasattr(response, 'eval_count'):
                logger.debug(f"Ollama tokens: prompt={response.prompt_eval_count}, completion={response.eval_count}")

        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise

        # 4. Converte response in AssistantMessage standardizzato
        assistant_message = self._convert_response_to_assistant_message(response)
        return assistant_message

    def _convert_response_to_assistant_message(self, response) -> AssistantMessage:
        """
        Converte risposta Ollama in AssistantMessage standardizzato
        """
        # Estrae tool_calls usando il metodo centralizzato
        tool_calls = self._extract_tool_calls(response)

        # Estrae il contenuto testuale
        content = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content or ""

        logger.debug(f"Ollama response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
