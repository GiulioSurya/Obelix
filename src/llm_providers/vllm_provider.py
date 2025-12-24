# src/llm_providers/vllm_provider.py
from typing import List, Optional, Dict, Any

from src.llm_providers.llm_abstraction import AbstractLLMProvider
from src.messages.assistant_message import AssistantMessage
from src.messages.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.logging_config import get_logger

# Logger per vLLM provider
logger = get_logger(__name__)

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    raise ImportError(
        "vLLM non è installato. Installa con: pip install vllm"
    )


class VLLMProvider(AbstractLLMProvider):
    """Provider per vLLM con parametri configurabili per offline inference"""

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
        Inizializza il provider vLLM per offline inference

        Args:
            model_id: ID del modello da caricare (default: "meta-llama/Llama-3.2-1B-Instruct")
            temperature: Temperatura per sampling (default: 0.1)
            max_tokens: Numero massimo di token da generare (default: 2000)
            top_p: Top-p (nucleus) sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed per riproducibilità (default: None)
            stop: Sequenze di stop (default: None)
            frequency_penalty: Penalità per frequenza token (default: None)
            presence_penalty: Penalità per presenza token (default: None)
            repetition_penalty: Penalità per ripetizioni (default: None)
            min_p: Minimum probability threshold (default: None)
            tensor_parallel_size: Numero di GPU per tensor parallelism (default: 1)
            dtype: Data type per i pesi del modello (default: "auto")
            quantization: Metodo di quantizzazione (es. "awq", "gptq") (default: None)
            gpu_memory_utilization: Frazione della memoria GPU da usare (default: 0.9)
            max_model_len: Lunghezza massima del contesto (default: None, usa config modello)
            trust_remote_code: Permetti esecuzione di codice remoto (default: False)
            **kwargs: Altri parametri da passare a vLLM
        """
        self.model_id = model_id

        # Inizializza il modello vLLM
        llm_kwargs = {
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

        # Aggiungi eventuali parametri extra
        llm_kwargs.update(kwargs)

        self.llm = LLM(**llm_kwargs)

        # Costruisci sampling parameters
        sampling_params_dict = {}

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

    def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoca il modello vLLM con messaggi e tool standardizzati
        """
        logger.debug(f"vLLM invoke: model={self.model_id}, messages={len(messages)}, tools={len(tools)}")

        # 1. Converte messaggi e tool nel formato vLLM (usa metodi base class)
        vllm_messages = self._convert_messages_to_provider_format(messages)
        vllm_tools = self._convert_tools_to_provider_format(tools)

        # 2. Chiama vLLM con llm.chat()
        try:
            if vllm_tools:
                outputs = self.llm.chat(
                    vllm_messages,
                    sampling_params=self.sampling_params,
                    tools=vllm_tools,
                    use_tqdm=False
                )
            else:
                outputs = self.llm.chat(
                    vllm_messages,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )

            logger.info(f"vLLM chat completed: {self.model_id}")

            # Log usage se disponibile
            if outputs and hasattr(outputs[0], 'metrics'):
                metrics = outputs[0].metrics
                logger.debug(f"vLLM metrics: {metrics}")

        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise

        # 3. Converte response in AssistantMessage standardizzato
        assistant_message = self._convert_response_to_assistant_message(outputs[0], vllm_tools)
        return assistant_message

    def _convert_response_to_assistant_message(self, output, tools: List[Dict[str, Any]]) -> AssistantMessage:
        """
        Converte risposta vLLM in AssistantMessage standardizzato

        Args:
            output: RequestOutput di vLLM
            tools: Lista di tools passati alla richiesta (per determinare se aspettarsi tool calls)
        """
        # Estrae tool_calls usando il metodo centralizzato (vLLM extractor richiede tools)
        tool_calls = self._extract_tool_calls(output, tools=tools)

        # Estrae il contenuto testuale
        content = ""
        if hasattr(output, 'outputs') and output.outputs:
            # output.outputs è una lista di CompletionOutput
            first_output = output.outputs[0]
            if hasattr(first_output, 'text'):
                content = first_output.text.strip()

        logger.debug(f"vLLM response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
