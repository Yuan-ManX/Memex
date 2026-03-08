from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple, Literal

import httpx

from memex.embedding.backends.base import EmbeddingBackend
from memex.embedding.backends.openai_backend import OpenAIEmbeddingBackend
from memex.embedding.backends.doubao_backend import (
    DoubaoEmbeddingBackend,
    DoubaoMultimodalEmbeddingInput,
)

logger = logging.getLogger(__name__)


def _load_proxy() -> Optional[str]:
    """
    Load proxy configuration from environment variables.
    """
    return (
        os.getenv("MEMEX_HTTP_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("HTTPS_PROXY")
        or None
    )


# ---------------------------------------------------------------------
# Backend Registry
# ---------------------------------------------------------------------

EMBEDDING_BACKENDS: Dict[str, Callable[[], EmbeddingBackend]] = {
    OpenAIEmbeddingBackend.name: OpenAIEmbeddingBackend,
    DoubaoEmbeddingBackend.name: DoubaoEmbeddingBackend,
}


# ---------------------------------------------------------------------
# HTTP Embedding Client
# ---------------------------------------------------------------------


class HTTPEmbeddingClient:
    """
    Generic HTTP client for embedding APIs.

    This client supports multiple embedding providers through
    the Memex embedding backend adapter layer.

    Supported providers
    -------------------
    - openai
    - doubao
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        embed_model: str,
        provider: str = "openai",
        endpoint_overrides: Optional[Dict[str, str]] = None,
        timeout: int = 60,
    ) -> None:
        """
        Initialize embedding HTTP client.

        Args
        ----
        base_url:
            Base URL of the embedding API.

        api_key:
            API key for authentication.

        embed_model:
            Embedding model name.

        provider:
            Embedding provider name.

        endpoint_overrides:
            Optional endpoint overrides.

        timeout:
            HTTP request timeout (seconds).
        """

        # Ensure base_url ends with "/" so httpx preserves path joining
        self.base_url: str = base_url.rstrip("/") + "/"
        self.api_key: str = api_key or ""
        self.embed_model: str = embed_model
        self.provider: str = provider.lower()
        self.timeout: int = timeout

        self.proxy: Optional[str] = _load_proxy()

        self.backend: EmbeddingBackend = self._load_backend(self.provider)

        overrides = endpoint_overrides or {}

        raw_embedding_endpoint = (
            overrides.get("embeddings")
            or overrides.get("embedding")
            or overrides.get("embed")
            or self.backend.embedding_endpoint
        )

        # httpx requires relative path
        self.embedding_endpoint: str = raw_embedding_endpoint.lstrip("/")

    # ------------------------------------------------------------------
    # Text Embeddings
    # ------------------------------------------------------------------

    async def embed(self, inputs: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text inputs.

        Args
        ----
        inputs:
            List of text strings.

        Returns
        -------
        List[List[float]]
            Embedding vectors.
        """

        if not inputs:
            return []

        payload = self.backend.build_embedding_payload(
            inputs=inputs,
            embed_model=self.embed_model,
        )

        data = await self._post(self.embedding_endpoint, payload)

        logger.debug("Embedding response: %s", data)

        return self.backend.parse_embedding_response(data)

    # ------------------------------------------------------------------
    # Multimodal Embeddings (Doubao)
    # ------------------------------------------------------------------

    async def embed_multimodal(
        self,
        inputs: List[Tuple[Literal["text", "image_url", "video_url"], str]],
        *,
        encoding_format: str = "float",
    ) -> List[List[float]]:
        """
        Generate multimodal embeddings (Doubao only).

        Args
        ----
        inputs:
            List of tuples (input_type, content)

        encoding_format:
            Encoding format ('float' or 'base64')

        Returns
        -------
        List[List[float]]
            Embedding vectors.

        Raises
        ------
        TypeError
            If backend does not support multimodal embedding.
        """

        if not isinstance(self.backend, DoubaoEmbeddingBackend):
            raise TypeError(
                "Multimodal embeddings are only supported by the 'doubao' provider "
                f"(current provider: {self.provider})"
            )

        multimodal_inputs = [
            DoubaoMultimodalEmbeddingInput(
                input_type=input_type,
                content=content,
            )
            for input_type, content in inputs
        ]

        payload = self.backend.build_multimodal_embedding_payload(
            inputs=multimodal_inputs,
            embed_model=self.embed_model,
            encoding_format=encoding_format,
        )

        endpoint = self.backend.multimodal_embedding_endpoint.lstrip("/")

        data = await self._post(endpoint, payload)

        logger.debug("Multimodal embedding response: %s", data)

        return self.backend.parse_multimodal_embedding_response(data)

    # ------------------------------------------------------------------
    # HTTP Helpers
    # ------------------------------------------------------------------

    async def _post(self, endpoint: str, payload: Dict) -> Dict:
        """
        Send POST request to embedding API.
        """

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            proxy=self.proxy,
        ) as client:

            resp = await client.post(
                endpoint,
                json=payload,
                headers=self._headers(),
            )

            resp.raise_for_status()

            return resp.json()

    def _headers(self) -> Dict[str, str]:
        """
        Build request headers.
        """

        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Backend Loader
    # ------------------------------------------------------------------

    def _load_backend(self, provider: str) -> EmbeddingBackend:
        """
        Load embedding backend adapter.
        """

        factory = EMBEDDING_BACKENDS.get(provider)

        if not factory:
            raise ValueError(
                f"Unsupported embedding provider '{provider}'. "
                f"Available providers: {', '.join(EMBEDDING_BACKENDS.keys())}"
            )

        return factory()
      
