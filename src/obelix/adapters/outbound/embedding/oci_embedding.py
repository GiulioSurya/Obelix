# src/embedding_providers/oci_embedding_provider.py
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)
from typing import List, Union, Literal
import numpy as np

from obelix.ports.outbound.embedding_provider import AbstractEmbeddingProvider


class OCIEmbeddingProvider(AbstractEmbeddingProvider):
    """
    Provider for OCI Cohere Embed v4 (1024 dimensions).

    Supports the following models:
    - cohere.embed-english-v4.0: Embedding for English text
    - cohere.embed-multilingual-v4.0: Multilingual embedding

    Docs: https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-embed-4.htm
    """

    # Cohere Embed v4 specifics
    EMBEDDING_DIM = 1024
    MAX_BATCH_SIZE = 96  # OCI API limit for batch embedding

    def __init__(
        self,
        model_id: str = "cohere.embed-multilingual-v3.0",
        input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document",
        truncate: Literal["NONE", "START", "END"] = "END"
    ):
        """
        Initialize the OCI Embedding provider.

        Args:
            model_id: OCI model ID
                - "cohere.embed-v4.0"
            input_type: Input type for embedding optimization
                - "search_document": for document indexing in vector DB
                - "search_query": for semantic search queries
                - "classification": for classification tasks
                - "clustering": for clustering tasks
            truncate: Truncation strategy for text exceeding context window
                - "NONE": error if text is too long
                - "START": truncate from beginning
                - "END": truncate from end (default)
        """
        self.model_id = model_id
        self.input_type = input_type
        self.truncate = truncate

        # Import config for centralized configuration
        from obelix.infrastructure.k8s import YamlConfig
        import os

        # Read complete OCI configuration from infrastructure.yaml (includes private_key_content)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_provider_config = infra_config.get("llm_providers.oci")

        # Validate presence of private key
        if not oci_provider_config.get("private_key_content"):
            raise ValueError(
                "Credential private_key_content missing in infrastructure.yaml. "
                "This key must be configured in the Kubernetes ConfigMap or Secrets."
            )

        # Initialize OCI configuration
        oci_config = {
            'user': oci_provider_config["user_id"],
            'fingerprint': oci_provider_config["fingerprint"],
            'key_content': oci_provider_config["private_key_content"],
            'tenancy': oci_provider_config["tenancy"],
            'region': oci_provider_config["region"],
        }

        # Initialize OCI client
        self.client = GenerativeAiInferenceClient(oci_config)

        # Use compartment_id from config
        self.compartment_id = oci_provider_config["compartment_id"]

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generates embeddings using Cohere Embed v4 on OCI.

        Args:
            texts: Single text (str) or list of texts (List[str])
                   Batch limit: max 96 texts per API call

        Returns:
            - If input is str: np.ndarray shape (1024,) dtype float32
            - If input is List[str]: List[np.ndarray], len = len(texts)

        Raises:
            ValueError: If batch size > 96 (OCI API limit)

        Example:
            >>> provider = OCIEmbeddingProvider()
            >>> # Single text
            >>> emb = provider.embed("What is the revenue?")
            >>> emb.shape
            (1024,)
            >>> # Batch
            >>> embs = provider.embed(["text1", "text2"])
            >>> len(embs)
            2
        """
        # Normalize input to list
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        # Validate batch size
        if len(text_list) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(text_list)} exceeds OCI API limit ({self.MAX_BATCH_SIZE}). "
                f"Split the request into smaller batches."
            )

        # Prepare embedding request
        # OCI API requires input_type in UPPERCASE
        embed_details = EmbedTextDetails(
            serving_mode=OnDemandServingMode(model_id=self.model_id),
            inputs=text_list,
            input_type=self.input_type.upper(),
            truncate=self.truncate,
            compartment_id=self.compartment_id
        )

        # OCI API call
        response = self.client.embed_text(embed_details)

        # Extract embeddings and convert to numpy array
        embeddings = [
            np.array(emb, dtype=np.float32)
            for emb in response.data.embeddings
        ]

        # Return appropriate format
        return embeddings[0] if is_single else embeddings

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimensionality of Cohere v4 embeddings.

        Returns:
            int: 1024 (fixed size for Cohere Embed v4)
        """
        return self.EMBEDDING_DIM