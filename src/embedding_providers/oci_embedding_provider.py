# src/embedding_providers/oci_embedding_provider.py
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    EmbedTextDetails,
    OnDemandServingMode
)
from typing import List, Union, Literal
import numpy as np

from src.embedding_providers.abstract_embedding_provider import AbstractEmbeddingProvider


class OCIEmbeddingProvider(AbstractEmbeddingProvider):
    """
    Provider per OCI Cohere Embed v4 (1024 dimensioni).

    Supporta i modelli:
    - cohere.embed-english-v4.0: Embedding per testo inglese
    - cohere.embed-multilingual-v4.0: Embedding multilingua

    Docs: https://docs.oracle.com/en-us/iaas/Content/generative-ai/cohere-embed-4.htm
    """

    # Cohere Embed v4 specifics
    EMBEDDING_DIM = 1024
    MAX_BATCH_SIZE = 96  # Limite API OCI per batch embedding

    def __init__(
        self,
        model_id: str = "cohere.embed-multilingual-v3.0",
        input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document",
        truncate: Literal["NONE", "START", "END"] = "END"
    ):
        """
        Inizializza il provider OCI Embedding.

        Args:
            model_id: ID del modello OCI
                - "cohere.embed-v4.0"
            input_type: Tipo di input per ottimizzazione embedding
                - "search_document": per indicizzazione documenti in vector DB
                - "search_query": per query di ricerca semantica
                - "classification": per task di classificazione
                - "clustering": per task di clustering
            truncate: Strategia di troncamento per testo eccedente context window
                - "NONE": errore se testo troppo lungo
                - "START": tronca dall'inizio
                - "END": tronca dalla fine (default)
        """
        self.model_id = model_id
        self.input_type = input_type
        self.truncate = truncate

        # Import config per configurazione centralizzata
        from src.k8s_config import YamlConfig
        import os

        # Leggi configurazione OCI completa da infrastructure.yaml (include private_key_content)
        infra_config = YamlConfig(os.getenv("INFRASTRUCTURE_CONFIG_PATH"))
        oci_provider_config = infra_config.get("llm_providers.oci")

        # Validazione presenza chiave privata
        if not oci_provider_config.get("private_key_content"):
            raise ValueError(
                "Credenziale private_key_content mancante in infrastructure.yaml. "
                "Questa chiave deve essere configurata nel ConfigMap o Secrets di Kubernetes."
            )

        # Inizializza configurazione OCI
        oci_config = {
            'user': oci_provider_config["user_id"],
            'fingerprint': oci_provider_config["fingerprint"],
            'key_content': oci_provider_config["private_key_content"],
            'tenancy': oci_provider_config["tenancy"],
            'region': oci_provider_config["region"],
        }

        # Inizializza client OCI
        self.client = GenerativeAiInferenceClient(oci_config)

        # Usa compartment_id da config
        self.compartment_id = oci_provider_config["compartment_id"]

    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Genera embeddings usando Cohere Embed v4 su OCI.

        Args:
            texts: Singolo testo (str) o lista di testi (List[str])
                   Limite batch: max 96 testi per chiamata API

        Returns:
            - Se input è str: np.ndarray shape (1024,) dtype float32
            - Se input è List[str]: List[np.ndarray], len = len(texts)

        Raises:
            ValueError: Se batch size > 96 (limite API OCI)

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
        # Normalizza input in lista
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        # Validazione batch size
        if len(text_list) > self.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(text_list)} eccede il limite API OCI ({self.MAX_BATCH_SIZE}). "
                f"Suddividi la richiesta in batch più piccoli."
            )

        # Prepara richiesta embedding
        # OCI API richiede input_type in UPPERCASE
        embed_details = EmbedTextDetails(
            serving_mode=OnDemandServingMode(model_id=self.model_id),
            inputs=text_list,
            input_type=self.input_type.upper(),
            truncate=self.truncate,
            compartment_id=self.compartment_id
        )

        # Chiamata API OCI
        response = self.client.embed_text(embed_details)

        # Estrai embeddings e converti in numpy array
        embeddings = [
            np.array(emb, dtype=np.float32)
            for emb in response.data.embeddings
        ]

        # Restituisci formato appropriato
        return embeddings[0] if is_single else embeddings

    def get_embedding_dimension(self) -> int:
        """
        Restituisce la dimensionalità degli embedding Cohere v4.

        Returns:
            int: 1024 (dimensione fissa per Cohere Embed v4)
        """
        return self.EMBEDDING_DIM