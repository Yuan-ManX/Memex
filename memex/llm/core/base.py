from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LLMBackend:
    """
    Base interface for HTTP-based LLM providers.

    Memex supports multiple LLM backends (OpenAI, Anthropic, Gemini, local models).
    Each provider implements this adapter to translate Memex requests into
    provider-specific API payloads and parse the responses.

    Responsibilities:
        - Build request payloads
        - Parse response payloads
        - Normalize output formats for Memex memory pipelines
    """

    #: Backend name (for registry / logging)
    name: str = "base"

    #: Default endpoint used for summarization
    summary_endpoint: str = "/chat/completions"

    # ---------------------------------------------------------------------
    # Summary API
    # ---------------------------------------------------------------------

    def build_summary_payload(
        self,
        *,
        text: str,
        system_prompt: Optional[str],
        chat_model: str,
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build the request payload for text summarization.

        Args:
            text:
                Input text to summarize.

            system_prompt:
                Optional system instruction for the LLM.

            chat_model:
                Model identifier used by the provider.

            max_tokens:
                Optional max tokens limit for the response.

        Returns
        -------
        dict
            Provider-specific JSON payload.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_summary_payload()"
        )

    def parse_summary_response(self, data: Dict[str, Any]) -> str:
        """
        Extract the summarized text from a provider response.

        Args:
            data:
                Raw JSON response from the provider API.

        Returns
        -------
        str
            Extracted summary text.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement parse_summary_response()"
        )

    # ---------------------------------------------------------------------
    # Vision API
    # ---------------------------------------------------------------------

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
        Build payload for multimodal (vision) inference.

        Args:
            prompt:
                Text instruction accompanying the image.

            base64_image:
                Image encoded in base64.

            mime_type:
                MIME type of the image (image/png, image/jpeg, etc).

            system_prompt:
                Optional system instruction.

            chat_model:
                Model identifier.

            max_tokens:
                Optional token limit.

        Returns
        -------
        dict
            Provider-specific JSON payload for multimodal request.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_vision_payload()"
        )

    # ---------------------------------------------------------------------
    # Optional helper methods
    # ---------------------------------------------------------------------

    def get_summary_endpoint(self) -> str:
        """
        Return the endpoint used for summarization.

        Providers may override this if they use different routes.

        Returns
        -------
        str
        """
        return self.summary_endpoint

    def validate_response(self, data: Dict[str, Any]) -> None:
        """
        Basic validation for provider responses.

        Providers may override this method if their response format
        includes error fields or metadata validation.

        Raises
        ------
        RuntimeError
            If the response appears invalid.
        """
        if not isinstance(data, dict):
            logger.error("Invalid LLM response type: %s", type(data))
            raise RuntimeError("Invalid LLM response: expected JSON object")

        if "error" in data:
            logger.error("LLM provider returned error: %s", data["error"])
            raise RuntimeError(f"LLM provider error: {data['error']}")
          
