#!/usr/bin/env python3
"""
LangChain-specific Capella AI Model Services

LangChain adapters over the framework-neutral ``CapellaClient`` in ``capella_http.py``.

The Capella wire protocol (``/v1`` URL handling, ``input_type`` for asymmetric
embedding models, token-budget truncation, auth headers) lives in ``capella_http``;
this module only converts between LangChain types and plain Python values.
"""

import logging
from typing import List, Optional, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, SecretStr

from .capella_http import CapellaClient, completion_text, default_embedding_max_tokens, v1_base_url

logger = logging.getLogger(__name__)


class CapellaLangChainEmbeddings(Embeddings):
    """
    Custom embeddings class for Capella AI that handles input_type parameter.
    
    Capella's asymmetric embedding models (like nvidia/nv-embedqa-e5-v5) require
    different input_type values for queries vs passages, which OpenAI wrapper doesn't support.
    """
    
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
        self.client = CapellaClient(api_key=api_key, base_url=base_url)
        self.model = model
        self.input_type_for_query = input_type_for_query
        self.input_type_for_passage = input_type_for_passage
        self.max_tokens = max_tokens or default_embedding_max_tokens()

        logger.info("✅ Using direct Capella embeddings API key")
        logger.info(f"✅ Using Capella direct API for model: {model}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages)."""
        return self.client.embed(
            self.model, texts, self.input_type_for_passage, self.max_tokens
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.client.embed(
            self.model, [text], self.input_type_for_query, self.max_tokens
        )[0]


class CapellaLangChainLLM(BaseChatModel):
    """
    Custom LLM class for Capella AI that handles URL construction correctly.
    
    Ensures the base_url has proper /v1 suffix and handles Capella-specific responses.
    """
    
    model: str = Field(..., description="Model name to use")
    api_key: SecretStr = Field(..., description="API key for authentication")
    base_url: str = Field(..., description="Base URL for the API")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = v1_base_url(self.base_url)

        logger.info("✅ Using direct Capella LLM API key")
        logger.info(f"✅ Using Capella direct API for LLM: {self.model}")

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "capella"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response."""
        api_messages = [
            {
                "role": "assistant" if isinstance(message, AIMessage) else "user",
                "content": message.content,
            }
            for message in messages
        ]

        params = dict(kwargs)
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens
        if stop:
            params["stop"] = stop

        client = CapellaClient(api_key=self.api_key.get_secret_value(), base_url=self.base_url)
        response = client.chat(
            self.model, api_messages, temperature=self.temperature, timeout=60, **params
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=completion_text(response)))]
        )

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        """Fallback to non-streaming for Capella LLM."""
        # Convert single generation to streaming chunks
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        # Get the first (and only) generation
        if result.generations and result.generations[0]:
            generation = result.generations[0]
            content = generation.text
            
            # Import here to avoid circular imports
            from langchain_core.outputs import ChatGenerationChunk
            from langchain_core.messages import AIMessageChunk
            
            # Yield the content as a single chunk
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=content),
                generation_info=generation.generation_info
            )
            yield chunk


def create_capella_embeddings(
    api_key: str,
    base_url: str,
    model: str,
    input_type_for_query: str = "query",
    input_type_for_passage: str = "passage",
    **kwargs
) -> CapellaLangChainEmbeddings:
    """Factory function to create Capella embeddings instance for LangChain."""
    return CapellaLangChainEmbeddings(
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
) -> CapellaLangChainLLM:
    """Factory function to create Capella LLM instance for LangChain."""
    llm = CapellaLangChainLLM(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )
    
    # Add callbacks if provided (for LangChain agents)
    if callbacks:
        llm.callbacks = callbacks
        
    return llm