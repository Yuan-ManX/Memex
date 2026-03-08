from __future__ import annotations

import logging
from typing import List, cast

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbeddingSDKClient:
    """
    OpenAI Embedding Client using the official OpenAI Python SDK.

    This client provides an async interface for generating text
    embeddings. It supports automatic batching to avoid provider
    limits on the number of inputs per request.

    Supported providers
    -------------------
    - OpenAI
    - OpenRouter
    - Grok (if embedding enabled)
    - Other OpenAI-compatible APIs
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        embed_model: str,
        batch_size: int = 25,
    ) -> None:
        """
        Initialize embedding client.

        Args
        ----
        base_url:
            Base URL of the OpenAI-compatible API.

        api_key:
            API key used for authentication.

        embed_model:
            Embedding model name.

        batch_size:
            Maximum number of inputs per request.
        """

        self.base_url: str = base_url.rstrip("/")
        self.api_key: str = api_key or ""
        self.embed_model: str = embed_model
        self.batch_size: int = batch_size

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def embed(self, inputs: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of input texts.

        The method automatically splits requests into batches to
        respect provider API limits.

        Args
        ----
        inputs:
            List of text strings to embed.

        Returns
        -------
        List[List[float]]
            List of embedding vectors.
        """

        if not inputs:
            return []

        logger.debug("Embedding %d inputs using model %s", len(inputs), self.embed_model)

        all_embeddings: List[List[float]] = []

        for batch in self._batch_inputs(inputs):
            try:
                response = await self.client.embeddings.create(
                    model=self.embed_model,
                    input=batch,
                )

                batch_embeddings = [
                    cast(List[float], item.embedding) for item in response.data
                ]

                all_embeddings.extend(batch_embeddings)

            except Exception:
                logger.exception("Embedding request failed")
                raise

        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_inputs(self, inputs: List[str]) -> List[List[str]]:
        """
        Split inputs into batches.

        Args
        ----
        inputs:
            List of input texts.

        Returns
        -------
        List[List[str]]
            Batched input lists.
        """

        if len(inputs) <= self.batch_size:
            return [inputs]

        batches: List[List[str]] = []

        for i in range(0, len(inputs), self.batch_size):
            batches.append(inputs[i : i + self.batch_size])

        return batches
      
