#!/usr/bin/env python3
"""
Framework-neutral client for Capella AI Model Services.

Capella exposes OpenAI-compatible ``/v1/embeddings`` and ``/v1/chat/completions``
endpoints with a few quirks that every framework adapter has to handle the same way:

- the ``/v1`` suffix must be appended to the endpoint exactly once
- asymmetric ``nv-embedqa`` models require ``input_type`` ("query" or "passage")
- embedding inputs must be truncated to the model's token budget

This module owns those rules. The LangChain and LlamaIndex adapters in
``capella_model_services_langchain.py`` and ``capella_model_services_llamaindex.py``
only translate between their framework's types and these plain-Python calls.
"""

import logging
import math
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3
TRUNCATION_SAFETY_BUFFER = 0.8


def v1_base_url(base_url: str) -> str:
    """Return ``base_url`` with exactly one trailing ``/v1``."""
    url = base_url.rstrip("/")
    return url if url.endswith("/v1") else f"{url}/v1"


def needs_input_type(model: str) -> bool:
    """Asymmetric NVIDIA embedqa models need an ``input_type`` per request."""
    return "nv-embedqa" in model.lower()


def default_embedding_max_tokens() -> int:
    return int(os.getenv("CAPELLA_API_EMBEDDING_MAX_TOKENS", "512"))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate ``text`` using a conservative ~3 characters per token estimate."""
    estimated_tokens = math.ceil(len(text) / CHARS_PER_TOKEN)
    if estimated_tokens <= max_tokens:
        return text

    max_chars = int(max_tokens * CHARS_PER_TOKEN * TRUNCATION_SAFETY_BUFFER)
    truncated = text[:max_chars]
    logger.warning(
        f"⚠️ Truncated text from {len(text)} to {len(truncated)} characters "
        f"(estimated {estimated_tokens} tokens, limit {max_tokens})"
    )
    return truncated


class CapellaClient:
    """Minimal HTTP client for one Capella AI endpoint and API key."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = v1_base_url(base_url)

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{self.base_url}{path}", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def embed(
        self,
        model: str,
        texts: List[str],
        input_type: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout: float = 30,
    ) -> List[List[float]]:
        """Embed ``texts``; ``input_type`` is sent only for models that need it."""
        budget = max_tokens or default_embedding_max_tokens()
        payload: Dict[str, Any] = {
            "model": model,
            "input": [truncate_to_token_budget(text, budget) for text in texts],
        }
        if input_type and needs_input_type(model):
            payload["input_type"] = input_type

        try:
            result = self._post("/embeddings", payload, timeout)
        except Exception as e:
            logger.error(f"❌ Capella embeddings API call failed: {e}")
            raise
        return [item["embedding"] for item in result["data"]]

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        timeout: float = 30,
        **params: Any,
    ) -> Dict[str, Any]:
        """Send OpenAI-format ``messages`` and return the raw completion response."""
        payload = {"model": model, "messages": messages, "temperature": temperature, **params}
        try:
            return self._post("/chat/completions", payload, timeout)
        except Exception as e:
            logger.error(f"❌ Capella LLM API call failed: {e}")
            raise


def completion_text(response: Dict[str, Any]) -> str:
    """Extract the assistant text from a chat completion response."""
    return response["choices"][0]["message"]["content"]
