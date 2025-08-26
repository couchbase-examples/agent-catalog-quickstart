#!/usr/bin/env python3
"""
LangChain-specific Capella AI Model Services

Custom implementations for Capella AI embeddings and LLM that handle:
- input_type parameter for asymmetric embedding models
- Correct URL construction (/v1 suffix)
- Token limits and text truncation
- LangChain framework compatibility
"""

import logging
import math
import os
from typing import List, Optional, Any

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field, SecretStr

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
        self.api_key = api_key
        # Ensure base_url has /v1 suffix for embeddings endpoint
        self.base_url = base_url.rstrip('/') + '/v1' if not base_url.endswith('/v1') else base_url
        self.model = model
        self.input_type_for_query = input_type_for_query
        self.input_type_for_passage = input_type_for_passage
        # Use environment variable with 512 as fallback
        self.max_tokens = max_tokens or int(os.getenv("CAPELLA_API_EMBEDDING_MAX_TOKENS", "512"))
        
        # Check if this model needs input_type (nv-embedqa models)
        self.needs_input_type = "nv-embedqa" in model.lower()
        
        logger.info("✅ Using direct Capella embeddings API key")
        logger.info(f"✅ Using Capella direct API for model: {model}")

    def _estimate_token_count(self, text: str) -> int:
        """Conservative token estimation using character count."""
        # More conservative estimation for token counting
        return math.ceil(len(text) / 3)  # ~3 chars per token

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits."""
        estimated_tokens = self._estimate_token_count(text)
        
        if estimated_tokens <= self.max_tokens:
            return text
            
        # Calculate max characters with safety buffer
        max_chars = int(self.max_tokens * 3 * 0.8)  # 0.8 safety buffer
        truncated = text[:max_chars]
        
        # Log truncation details
        logger.warning(f"⚠️ Truncated text from {len(text)} to {len(truncated)} characters (estimated {estimated_tokens} → {self._estimate_token_count(truncated)} tokens)")
        logger.info(f"📏 Text truncated: {len(text)} → {len(truncated)} chars, est. tokens: {self._estimate_token_count(truncated)}")
        
        return truncated

    def _make_embedding_request(self, texts: List[str], input_type: Optional[str] = None) -> List[List[float]]:
        """Make embedding request to Capella API."""
        try:
            # Truncate texts to fit token limits
            truncated_texts = [self._truncate_text(text) for text in texts]
            
            # Prepare request data
            data = {
                "model": self.model,
                "input": truncated_texts,
            }
            
            # Add input_type if model requires it and type is specified
            if self.needs_input_type and input_type:
                data["input_type"] = input_type
                logger.debug(f"🔧 Using input_type: {input_type} for {len(texts)} texts")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            with httpx.Client(timeout=30) as client:
                logger.debug(f"📡 Making embedding request for {len(texts)} texts")
                response = client.post(
                    f"{self.base_url}/embeddings",
                    json=data,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                return [item["embedding"] for item in result["data"]]
                
        except Exception as e:
            logger.error(f"❌ Capella embeddings API call failed: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents (passages)."""
        return self._make_embedding_request(texts, self.input_type_for_passage)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        result = self._make_embedding_request([text], self.input_type_for_query)
        return result[0]


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
        # Ensure base_url has /v1 suffix for chat endpoint
        if not self.base_url.endswith('/v1'):
            self.base_url = self.base_url.rstrip('/') + '/v1'
        
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
        try:
            # Convert messages to API format
            api_messages = []
            for message in messages:
                if isinstance(message, HumanMessage):
                    api_messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    api_messages.append({"role": "assistant", "content": message.content})
                else:
                    # Default to user for other message types
                    api_messages.append({"role": "user", "content": message.content})
            
            # Prepare request data
            data = {
                "model": self.model,
                "messages": api_messages,
                "temperature": self.temperature,
                **kwargs
            }
            
            if self.max_tokens:
                data["max_tokens"] = self.max_tokens
            
            if stop:
                data["stop"] = stop
            
            headers = {
                "Authorization": f"Bearer {self.api_key.get_secret_value()}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            with httpx.Client(timeout=60) as client:
                logger.debug(f"📡 Making LLM request with {len(api_messages)} messages")
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=data,
                    headers=headers
                )
                response.raise_for_status()
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=content))]
                )
                
        except Exception as e:
            logger.error(f"❌ Capella LLM API call failed: {e}")
            raise

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