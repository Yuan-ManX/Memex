from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

from memex.llm.core.base import LLMBackend

logger = logging.getLogger(__name__)


class DoubaoLLMBackend(LLMBackend):
    """
    Backend adapter for Doubao LLM API.

    Doubao provides an OpenAI-compatible interface for chat and
    multimodal inference.

    Supported capabilities
    ----------------------
    - Chat completions
    - Text summarization
    - Vision (image + text)
    """

    name: str = "doubao"
    summary_endpoint: str = "/api/v3/chat/completions"

    # ------------------------------------------------------------------
    # Chat / Summarization
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
        Build payload for Doubao chat completion API.

        Args
        ----
        text:
            Text to summarize.

        system_prompt:
            Optional system instruction.

        chat_model:
            Doubao model name.

        max_tokens:
            Optional response token limit.

        Returns
        -------
        dict
            JSON payload for Doubao API.
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
        Parse Doubao API response.

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
            Parsed model output.
        """

        try:
            choices = data.get("choices")

            if not choices:
                raise ValueError("Missing 'choices' field in Doubao response")

            message = choices[0].get("message", {})
            content = message.get("content")

            if not isinstance(content, str):
                raise ValueError("Invalid message content format")

            return cast(str, content)

        except Exception:
            logger.exception("Failed to parse Doubao response")
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
        Build payload for Doubao Vision API.

        Args
        ----
        prompt:
            Instruction for the image.

        base64_image:
            Base64 encoded image.

        mime_type:
            Image MIME type (image/png, image/jpeg, etc).

        system_prompt:
            Optional system prompt.

        chat_model:
            Vision-capable model name.

        max_tokens:
            Optional token limit.

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
      
