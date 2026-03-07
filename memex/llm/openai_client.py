"""
Memex OpenAI SDK Client

This module provides a unified OpenAI client used across Memex
for LLM interaction, embedding generation, multimodal analysis,
and audio transcription.

Design goals:
- Async-first architecture
- Strong typing
- Modular API wrappers
- Clean logging
- Compatible with Memex memory pipelines
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any, Literal, cast

from openai import AsyncOpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

logger = logging.getLogger(__name__)


class OpenAISDKClient:
    """
    Memex OpenAI Client

    Wrapper around the official OpenAI Python SDK used for:

    - LLM chat completion
    - summarization
    - vision analysis
    - embedding generation
    - audio transcription

    This client acts as the **LLM interface layer** for Memex.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        chat_model: str,
        embed_model: str,
        embed_batch_size: int = 32,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.embed_batch_size = embed_batch_size

        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    # -------------------------------------------------------------------------
    # Internal Utilities
    # -------------------------------------------------------------------------

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[ChatCompletionMessageParam]:
        """Construct standard chat messages."""
        messages: list[ChatCompletionMessageParam] = []

        if system_prompt:
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": system_prompt,
            }
            messages.append(system_message)

        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": prompt,
        }

        messages.append(user_message)

        return messages

    @staticmethod
    def _encode_image(image_path: str) -> tuple[str, str]:
        """
        Encode image file as base64 data URL.

        Returns:
            (mime_type, base64_string)
        """
        path = Path(image_path)

        image_bytes = path.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")

        return mime_type, base64_image

    # -------------------------------------------------------------------------
    # Chat Completion
    # -------------------------------------------------------------------------

    async def chat(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, ChatCompletion]:
        """
        Run a standard chat completion.

        Returns:
            Tuple[str, ChatCompletion]
        """

        messages = self._build_messages(prompt, system_prompt)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""

        logger.debug("Chat completion finished | tokens=%s", response.usage)

        return content, response

    # -------------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------------

    async def summarize(
        self,
        text: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ChatCompletion]:
        """
        Summarize text content.

        Used in Memex memory compression pipelines.
        """

        prompt = system_prompt or "Summarize the following text in one concise paragraph."

        messages = [
            cast(
                ChatCompletionSystemMessageParam,
                {"role": "system", "content": prompt},
            ),
            cast(
                ChatCompletionUserMessageParam,
                {"role": "user", "content": text},
            ),
        ]

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""

        logger.debug("Summarization completed")

        return content, response

    # -------------------------------------------------------------------------
    # Vision API
    # -------------------------------------------------------------------------

    async def vision(
        self,
        prompt: str,
        image_path: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, ChatCompletion]:
        """
        Analyze an image using a multimodal model.
        """

        mime_type, base64_image = self._encode_image(image_path)

        messages: list[ChatCompletionMessageParam] = []

        if system_prompt:
            messages.append(
                cast(
                    ChatCompletionSystemMessageParam,
                    {"role": "system", "content": system_prompt},
                )
            )

        text_part: ChatCompletionContentPartTextParam = {
            "type": "text",
            "text": prompt,
        }

        image_part: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}",
            },
        }

        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [text_part, image_part],
        }

        messages.append(user_message)

        response = await self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content or ""

        logger.debug("Vision request completed")

        return content, response

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    async def embed(
        self,
        inputs: list[str],
    ) -> tuple[list[list[float]], CreateEmbeddingResponse | None]:
        """
        Generate embeddings for a list of texts.

        Supports batching for large datasets.
        """

        if not inputs:
            return [], None

        all_embeddings: list[list[float]] = []
        last_response: CreateEmbeddingResponse | None = None

        for start in range(0, len(inputs), self.embed_batch_size):
            batch = inputs[start : start + self.embed_batch_size]

            response = await self.client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )

            embeddings = [cast(list[float], d.embedding) for d in response.data]

            all_embeddings.extend(embeddings)
            last_response = response

        logger.debug("Generated %d embeddings", len(all_embeddings))

        return all_embeddings, last_response

    # -------------------------------------------------------------------------
    # Audio Transcription
    # -------------------------------------------------------------------------

    async def transcribe(
        self,
        audio_path: str,
        *,
        prompt: str | None = None,
        language: str | None = None,
        response_format: Literal["text", "json", "verbose_json"] = "text",
    ) -> tuple[str, Any]:
        """
        Transcribe audio using OpenAI speech models.
        """

        kwargs: dict[str, Any] = {}

        if prompt:
            kwargs["prompt"] = prompt

        if language:
            kwargs["language"] = language

        try:
            with open(audio_path, "rb") as audio_stream:
                transcription = await self.client.audio.transcriptions.create(
                    file=audio_stream,
                    model="gpt-4o-mini-transcribe",
                    response_format=response_format,
                    **kwargs,
                )

            if response_format == "text":
                text = transcription if isinstance(transcription, str) else transcription.text
            else:
                text = transcription.text if hasattr(transcription, "text") else str(transcription)

            logger.debug("Transcription completed | file=%s", audio_path)

            return text or "", transcription

        except Exception:
            logger.exception("Audio transcription failed | file=%s", audio_path)
            raise
    
