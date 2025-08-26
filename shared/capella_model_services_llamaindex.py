#!/usr/bin/env python3
"""
LlamaIndex-specific Capella AI Model Services

Custom implementations for Capella AI embeddings and LLM that handle:
- input_type parameter for asymmetric embedding models
- Correct URL construction (/v1 suffix)
- Token limits and text truncation
- LlamaIndex framework compatibility
"""

import logging
import math
import os
from typing import List, Optional, Any

import httpx
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.llms.llm import LLM as BaseLLM
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, CompletionResponse, MessageRole, LLMMetadata

logger = logging.getLogger(__name__)


class CapellaLlamaIndexEmbeddings(BaseEmbedding):
    """
    LlamaIndex-compatible embeddings class for Capella AI.
    
    This class wraps the Capella AI API for use with LlamaIndex framework.
    """
    
    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
    _model: str = PrivateAttr()
    _input_type_for_query: str = PrivateAttr()
    _input_type_for_passage: str = PrivateAttr()
    _max_tokens: int = PrivateAttr()
    _needs_input_type: bool = PrivateAttr()
    
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
        self._api_key = api_key
        # Ensure base_url has /v1 suffix for embeddings endpoint
        self._base_url = base_url.rstrip('/') + '/v1' if not base_url.endswith('/v1') else base_url
        self._model = model
        self._input_type_for_query = input_type_for_query
        self._input_type_for_passage = input_type_for_passage
        # Use environment variable with 512 as fallback
        self._max_tokens = max_tokens or int(os.getenv("CAPELLA_API_EMBEDDING_MAX_TOKENS", "512"))
        
        # Check if this model needs input_type (nv-embedqa models)
        self._needs_input_type = "nv-embedqa" in model.lower()
        
        logger.info("✅ Using direct Capella embeddings API key")
        logger.info(f"✅ Using Capella direct API for model: {model}")

    def _estimate_token_count(self, text: str) -> int:
        """Conservative token estimation using character count."""
        return math.ceil(len(text) / 3)  # ~3 chars per token

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits."""
        estimated_tokens = self._estimate_token_count(text)
        
        if estimated_tokens <= self._max_tokens:
            return text
            
        # Calculate max characters with safety buffer
        max_chars = int(self._max_tokens * 3 * 0.8)  # 0.8 safety buffer
        truncated = text[:max_chars]
        
        logger.warning(f"⚠️ Truncated text from {len(text)} to {len(truncated)} characters")
        return truncated

    def _make_embedding_request(self, texts: List[str], input_type: str) -> List[List[float]]:
        """Make embedding request to Capella API."""
        try:
            # Truncate texts to fit token limits
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            # Prepare request data
            data = {
                "model": self._model,
                "input": truncated_texts,
            }
            
            # Add input_type if model requires it
            if self._needs_input_type:
                data["input_type"] = input_type
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self._base_url}/embeddings",
                    json=data,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                return [item["embedding"] for item in result["data"]]
                
        except Exception as e:
            logger.error(f"❌ Capella embeddings API call failed: {e}")
            raise

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
    
    _api_key: str = PrivateAttr()
    _base_url: str = PrivateAttr()
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
        self._api_key = api_key
        # Ensure base_url has /v1 suffix for chat endpoint
        self._base_url = base_url.rstrip('/') + '/v1' if not base_url.endswith('/v1') else base_url
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
        try:
            data = {
                "model": self._model,
                "messages": messages,
                "temperature": self._temperature,
                **kwargs
            }
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            with httpx.Client(timeout=30) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    json=data,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"❌ Capella LLM API call failed: {e}")
            raise

    def complete(self, prompt: str, **kwargs):
        """Complete endpoint for LlamaIndex."""
        # Convert prompt to chat messages format
        messages = [{"role": "user", "content": prompt}]
        
        result = self._make_request(messages, **kwargs)
        content = result["choices"][0]["message"]["content"]
        
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
        content = result["choices"][0]["message"]["content"]
        
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