# src/client_adapters/vllm_provider.py
import asyncio
from typing import List, Optional, Dict, Any

from src.client_adapters.llm_abstraction import AbstractLLMProvider
from src.client_adapters._legacy_mapping_mixin import LegacyMappingMixin
from src.obelix_types.assistant_message import AssistantMessage
from src.obelix_types.standard_message import StandardMessage
from src.tools.tool_base import ToolBase
from src.providers import Providers
from src.logging_config import get_logger

# Logger for vLLM provider
logger = get_logger(__name__)

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:
    raise ImportError(
        "vLLM is not installed. Install with: pip install vllm"
    )


class VLLMProvider(LegacyMappingMixin, AbstractLLMProvider):
    """Provider for vLLM with configurable parameters for offline inference"""

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
        Initialize the vLLM provider for offline inference

        Args:
            model_id: Model ID to load (default: "meta-llama/Llama-3.2-1B-Instruct")
            temperature: Sampling temperature (default: 0.1)
            max_tokens: Maximum number of tokens to generate (default: 2000)
            top_p: Top-p (nucleus) sampling (default: None)
            top_k: Top-k sampling (default: None)
            seed: Seed for reproducibility (default: None)
            stop: Stop sequences (default: None)
            frequency_penalty: Penalty for token frequency (default: None)
            presence_penalty: Penalty for token presence (default: None)
            repetition_penalty: Penalty for repetitions (default: None)
            min_p: Minimum probability threshold (default: None)
            tensor_parallel_size: Number of GPUs for tensor parallelism (default: 1)
            dtype: Data type for model weights (default: "auto")
            quantization: Quantization method (e.g. "awq", "gptq") (default: None)
            gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
            max_model_len: Maximum context length (default: None, uses model config)
            trust_remote_code: Allow execution of remote code (default: False)
            **kwargs: Other parameters to pass to vLLM
        """
        self.model_id = model_id

        # Initialize the vLLM model
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

        # Add any extra parameters
        llm_kwargs.update(kwargs)

        self.llm = LLM(**llm_kwargs)

        # Build sampling parameters
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

    async def invoke(self, messages: List[StandardMessage], tools: List[ToolBase]) -> AssistantMessage:
        """
        Invoke the vLLM model with standardized obelix_types and tools (async).

        Uses asyncio.to_thread() to run the sync vLLM engine without
        blocking the event loop.
        """
        logger.debug(f"vLLM invoke: model={self.model_id}, obelix_types={len(messages)}, tools={len(tools)}")

        # 1. Convert obelix_types and tools to vLLM format (use base class methods)
        vllm_messages = self._convert_messages_to_provider_format(messages)
        vllm_tools = self._convert_tools_to_provider_format(tools)

        # 2. Call vLLM with llm.chat() via thread pool to avoid blocking event loop
        try:
            if vllm_tools:
                outputs = await asyncio.to_thread(
                    self.llm.chat,
                    vllm_messages,
                    sampling_params=self.sampling_params,
                    tools=vllm_tools,
                    use_tqdm=False
                )
            else:
                outputs = await asyncio.to_thread(
                    self.llm.chat,
                    vllm_messages,
                    sampling_params=self.sampling_params,
                    use_tqdm=False
                )

            logger.info(f"vLLM chat completed: {self.model_id}")

            # Log usage if available
            if outputs and hasattr(outputs[0], 'metrics'):
                metrics = outputs[0].metrics
                logger.debug(f"vLLM metrics: {metrics}")

        except Exception as e:
            logger.error(f"vLLM request failed: {e}")
            raise

        # 3. Convert response to standardized AssistantMessage
        assistant_message = self._convert_response_to_assistant_message(outputs[0], vllm_tools)
        return assistant_message

    def _convert_response_to_assistant_message(self, output, tools: List[Dict[str, Any]]) -> AssistantMessage:
        """
        Convert vLLM response to standardized AssistantMessage

        Args:
            output: RequestOutput from vLLM
            tools: List of tools passed to the request (to determine if tool calls are expected)
        """
        # Extract tool_calls using centralized method (vLLM extractor requires tools)
        tool_calls = self._extract_tool_calls(output, tools=tools)

        # Extract text content
        content = ""
        if hasattr(output, 'outputs') and output.outputs:
            # output.outputs is a list of CompletionOutput
            first_output = output.outputs[0]
            if hasattr(first_output, 'text'):
                content = first_output.text.strip()

        logger.debug(f"vLLM response: content_length={len(content)}, tool_calls={len(tool_calls)}")

        return AssistantMessage(
            content=content,
            tool_calls=tool_calls
        )
