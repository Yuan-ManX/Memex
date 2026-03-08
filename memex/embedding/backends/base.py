from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """
    Base adapter for embedding providers.

    This abstraction allows Memex to support multiple embedding
    providers through a unified interface.

    Providers supported via subclasses:
        - OpenAI
        - Doubao
        - Gemini
        - Ollama
        - Local embedding models

    Each backend is responsible for:
        - Building provider-specific request payloads
        - Parsing embedding responses
        - Normalizing output format
    """

    #: Provider name used in backend registry
    name: str = "base"

    #: Default embedding endpoint
    embedding_endpoint: str = "/embeddings"

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    def build_embedding_payload(
        self,
        *,
        inputs: List[str],
        embed_model: str,
    ) -> Dict[str, Any]:
        """
        Build provider-specific embedding request payload.

        Args
        ----
        inputs:
            List of input texts to embed.

        embed_model:
            Embedding model identifier.

        Returns
        -------
        Dict[str, Any]
            JSON payload sent to the provider API.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_embedding_payload()"
        )

    def parse_embedding_response(
        self,
        data: Dict[str, Any],
    ) -> List[List[float]]:
        """
        Parse embedding response returned by provider.

        Expected normalized output format
        ---------------------------------
        [
            [0.12, 0.45, ...],
            [0.98, 0.11, ...],
        ]

        Args
        ----
        data:
            Raw JSON response from embedding API.

        Returns
        -------
        List[List[float]]
            Embedding vectors.
        """

        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_embedding_response()"
        )

    # ------------------------------------------------------------------
    # Optional Extensions
    # ------------------------------------------------------------------

    def supports_multimodal(self) -> bool:
        """
        Return whether the backend supports multimodal embeddings.
        """

        return False

    # ------------------------------------------------------------------
    # Response Validation
    # ------------------------------------------------------------------

    def validate_response(self, data: Dict[str, Any]) -> None:
        """
        Validate embedding response.

        Providers may override this method to add
        provider-specific validation logic.

        Raises
        ------
        RuntimeError
            If the response appears invalid.
        """

        if not isinstance(data, dict):
            logger.error("Invalid embedding response type: %s", type(data))
            raise RuntimeError("Invalid embedding response format")

        if "error" in data:
            logger.error("Embedding provider returned error: %s", data["error"])
            raise RuntimeError(f"Embedding provider error: {data['error']}")
          
