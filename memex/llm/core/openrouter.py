from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from memex.llm.core.base import LLMBackend

logger = logging.getLogger(__name__)


class OpenRouterLLMBackend(LLMBackend):
    """
    Backend adapter for OpenRouter LLM APIs.

    OpenRouter provides an OpenAI-compatible interface, allowing Memex
    to interact with multiple LLM providers through a unified endpoint.

    Supported capabilities
    ----------------------
    - Chat completions
    - Text summarization
    - Vision (image + text)
    """

    name: str = "openrouter"
    summary_endpoint: str = "/api/v1/chat/completions"

    # ------------------------------------------------------------------
    # Summary / Chat API
    # ------------------------------------------------------------------

    def build_summary_payload(
        self,
        *,
        text: str,
        system_prompt: Optional[str],
        chat_model: str,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build request payload for OpenRouter chat completion.

        Args
        ----
        text:
            Input text to summarize.

        system_prompt:
            Optional system instruction.

        chat_model:
            Model name (e.g. openai/gpt-4o, anthropic/claude-3).

        max_tokens:
            Optional maximum response tokens.

        Returns
        -------
        dict
            JSON payload ready to send to the OpenRouter API.
        """

        prompt = system_prompt or "Summarize the text in one short paragraph."

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]

        payload: Dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
            "temperature": 0.2,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload

    def parse_summary_response(self, data: Dict[str, Any]) -> str:
        """
        Extract response text from OpenRouter API response.

        Expected format
        ---------------
        {
            "choices": [
                {
                    "message": {
                        "content": "response text"
                    }
                }
            ]
        }

        Returns
        -------
        str
            Parsed model response text.
        """

        try:
            choices = data.get("choices")
            if not choices:
                raise ValueError("Missing 'choices' field in OpenRouter response")

            message = choices[0].get("message", {})
            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError("Invalid message content format")

            return cast(str, content)

        except Exception:
            logger.exception("Failed to parse OpenRouter response")
            raise

    # ------------------------------------------------------------------
    # Vision API
    # ------------------------------------------------------------------

    def build_vision_payload(
        self,
        *,
        prompt: str,
        base64_image: str,
        mime_type: str,
        system_prompt: Optional[str],
        chat_model: str,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build payload for OpenRouter Vision API.

        Args
        ----
        prompt:
            Text instruction accompanying the image.

        base64_image:
            Base64 encoded image.

        mime_type:
            Image MIME type (image/png, image/jpeg, etc).

        system_prompt:
            Optional system instruction.

        chat_model:
            Vision-capable model identifier.

        max_tokens:
            Optional response token limit.

        Returns
        -------
        dict
            JSON payload for multimodal inference.
        """

        messages: List[Dict[str, Any]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                },
            },
        ]

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        payload: Dict[str, Any] = {
            "model": chat_model,
            "messages": messages,
            "temperature": 0.2,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        return payload
      
