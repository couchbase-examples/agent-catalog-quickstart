#!/usr/bin/env python3
"""
LlamaIndex-specific Capella AI Model Services

LlamaIndex adapters over the framework-neutral ``CapellaClient`` in ``capella_http.py``.

The Capella wire protocol (``/v1`` URL handling, ``input_type`` for asymmetric
embedding models, token-budget truncation, auth headers) lives in ``capella_http``;
this module only converts between LlamaIndex types and plain Python values.
"""

import logging
from typing import List, Optional, Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms.llm import LLM as BaseLLM
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, MessageRole, LLMMetadata

from .capella_http import CapellaClient, completion_text, default_embedding_max_tokens

logger = logging.getLogger(__name__)


class CapellaLlamaIndexEmbeddings(BaseEmbedding):
    """
    LlamaIndex-compatible embeddings class for Capella AI.
    
    This class wraps the Capella AI API for use with LlamaIndex framework.
    """
    
    _client: CapellaClient = PrivateAttr()
    _model: str = PrivateAttr()
    _input_type_for_query: str = PrivateAttr()
    _input_type_for_passage: str = PrivateAttr()
    _max_tokens: int = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        input_type_for_query: str = "query",
        input_type_for_passage: str = "passage",
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = CapellaClient(api_key=api_key, base_url=base_url)
        self._model = model
        self._input_type_for_query = input_type_for_query
        self._input_type_for_passage = input_type_for_passage
        self._max_tokens = max_tokens or default_embedding_max_tokens()

        logger.info("✅ Using direct Capella embeddings API key")
        logger.info(f"✅ Using Capella direct API for model: {model}")

    def _make_embedding_request(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Make embedding request to Capella API."""
        return self._client.embed(self._model, texts, input_type, self._max_tokens)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query."""
        result = self._make_embedding_request([query], self._input_type_for_query)
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        result = self._make_embedding_request([text], self._input_type_for_passage)
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        return self._make_embedding_request(texts, self._input_type_for_passage)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding."""
        return self._get_text_embedding(text)

    @classmethod
    def class_name(cls) -> str:
        return "capella_llamaindex"


class CapellaLlamaIndexLLM(BaseLLM):
    """
    LlamaIndex-compatible LLM class for Capella AI.
    
    This class wraps the Capella AI API for use with LlamaIndex framework.
    """
    
    _client: CapellaClient = PrivateAttr()
    _model: str = PrivateAttr()
    _temperature: float = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._client = CapellaClient(api_key=api_key, base_url=base_url)
        self._model = model
        self._temperature = temperature

        logger.info("✅ Using direct Capella LLM API key")
        logger.info(f"✅ Using Capella direct API for LLM: {model}")

    @property
    def metadata(self):
        """LLM metadata."""
        return LLMMetadata(
            context_window=4096,
            num_output=1024,
            is_chat_model=True,
            model_name=self._model,
        )

    @classmethod
    def class_name(cls) -> str:
        return "capella_llamaindex_llm"

    def _make_request(self, messages, **kwargs):
        """Make request to Capella API."""
        return self._client.chat(
            self._model, messages, temperature=self._temperature, timeout=30, **kwargs
        )

    def complete(self, prompt: str, **kwargs):
        """Complete endpoint for LlamaIndex."""
        # Convert prompt to chat messages format
        messages = [{"role": "user", "content": prompt}]
        
        result = self._make_request(messages, **kwargs)
        content = completion_text(result)
        
        return CompletionResponse(text=content, raw=result)

    def chat(self, messages, **kwargs):
        """Chat endpoint for LlamaIndex."""
        # Convert LlamaIndex ChatMessage objects to API format
        api_messages = []
        for msg in messages:
            api_messages.append({
                "role": msg.role.value,
                "content": msg.content or ""
            })
        
        result = self._make_request(api_messages, **kwargs)
        content = completion_text(result)
        
        response_msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content
        )
        
        return ChatResponse(message=response_msg, raw=result)

    def stream_complete(self, prompt: str, **kwargs):
        """Streaming completion - fallback to non-streaming for now."""
        response = self.complete(prompt, **kwargs)
        yield response

    def stream_chat(self, messages, **kwargs):
        """Streaming chat - fallback to non-streaming for now."""
        response = self.chat(messages, **kwargs)
        yield response

    async def acomplete(self, prompt: str, **kwargs):
        """Async complete - fallback to sync for now."""
        return self.complete(prompt, **kwargs)

    async def achat(self, messages, **kwargs):
        """Async chat - fallback to sync for now."""
        return self.chat(messages, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs):
        """Async streaming complete - fallback to sync for now."""
        response = await self.acomplete(prompt, **kwargs)
        yield response

    async def astream_chat(self, messages, **kwargs):
        """Async streaming chat - fallback to sync for now."""
        response = await self.achat(messages, **kwargs)
        yield response


def create_capella_embeddings(
    api_key: str,
    base_url: str,
    model: str,
    input_type_for_query: str = "query",
    input_type_for_passage: str = "passage",
    **kwargs
) -> CapellaLlamaIndexEmbeddings:
    """Factory function to create Capella embeddings instance for LlamaIndex."""
    return CapellaLlamaIndexEmbeddings(
        api_key=api_key,
        base_url=base_url,
        model=model,
        input_type_for_query=input_type_for_query,
        input_type_for_passage=input_type_for_passage,
        **kwargs
    )


def create_capella_chat_llm(
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.0,
    callbacks: Optional[List] = None,
    **kwargs
) -> CapellaLlamaIndexLLM:
    """Factory function to create Capella LLM instance for LlamaIndex."""
    return CapellaLlamaIndexLLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        **kwargs
    )