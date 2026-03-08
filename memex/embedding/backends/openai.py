from __future__ import annotations

import logging
from typing import Any, Dict, List, cast

from memex.embedding.backends.base import EmbeddingBackend

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """
    Backend adapter for OpenAI-compatible embedding APIs.

    This backend supports embedding providers that follow
    the OpenAI embeddings API format.

    Supported providers
    -------------------
    - OpenAI
    - OpenRouter
    - Grok (if embedding enabled)
    - Together
    - Fireworks
    - Other OpenAI-compatible APIs
    """

    name: str = "openai"

    #: Default embedding endpoint
    embedding_endpoint: str = "/embeddings"

    # ------------------------------------------------------------------
    # Payload Builder
    # ------------------------------------------------------------------

    def build_embedding_payload(
        self,
        *,
        inputs: List[str],
        embed_model: str,
    ) -> Dict[str, Any]:
        """
        Build embedding request payload.

        Args
        ----
        inputs:
            List of input texts.

        embed_model:
            Embedding model identifier.

        Returns
        -------
        Dict[str, Any]
            JSON payload sent to the embedding API.
        """

        return {
            "model": embed_model,
            "input": inputs,
        }

    # ------------------------------------------------------------------
    # Response Parser
    # ------------------------------------------------------------------

    def parse_embedding_response(
        self,
        data: Dict[str, Any],
    ) -> List[List[float]]:
        """
        Parse embedding response from OpenAI-compatible API.

        Expected format
        ---------------
        {
            "data": [
                {"embedding": [...]},
                {"embedding": [...]}
            ]
        }

        Returns
        -------
        List[List[float]]
            Embedding vectors.
        """

        self.validate_response(data)

        try:
            items = data.get("data")

            if not isinstance(items, list):
                raise ValueError("Invalid embedding response format: 'data' field missing")

            embeddings: List[List[float]] = []

            for item in items:
                vector = item.get("embedding")

                if not isinstance(vector, list):
                    raise ValueError("Invalid embedding vector format")

                embeddings.append(cast(List[float], vector))

            return embeddings

        except Exception:
            logger.exception("Failed to parse OpenAI embedding response")
            raise
          
