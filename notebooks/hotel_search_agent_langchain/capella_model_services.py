#!/usr/bin/env python3
"""
Custom Capella AI model services for embeddings and LLMs.

This module provides custom implementations that properly handle:
- input_type parameter for Capella's asymmetric embedding models (nvidia/nv-embedqa-e5-v5)
- Token length limits with automatic text truncation
- Endpoint URL formatting for Capella AI chat completions API
"""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import Field

logger = logging.getLogger(__name__)


class CapellaEmbeddings(Embeddings):
    """
    Custom Capella AI embeddings class that properly handles input_type parameter and token limits.
    
    This class makes direct API calls to Capella AI when the model requires input_type,
    automatically truncates text to fit within token limits, and falls back to standard 
    OpenAI embeddings for other models.
    """
    
    model: str = Field(...)
    api_key: str = Field(...)
    base_url: str = Field(...)
    input_type_query: str = Field(default="query")
    input_type_passage: str = Field(default="passage")
    max_retries: int = Field(default=3)
    request_timeout: int = Field(default=60)
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        input_type_query: str = "query",
        input_type_passage: str = "passage",
        max_retries: int = 3,
        request_timeout: int = 60,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.input_type_query = input_type_query
        self.input_type_passage = input_type_passage
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        
        # Check if this model needs special handling
        self.needs_input_type = (
            "llama-3.2-nv-embedqa" in model or "nv-embedqa" in model
        )
        
        if self.needs_input_type:
            logger.info(f"✅ Using Capella direct API for model: {model}")
        else:
            logger.info(f"✅ Model {model} doesn't require input_type, using standard OpenAI API")
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count using a conservative heuristic.
        
        This is an approximation - actual tokenization may vary.
        We use a conservative estimate to avoid exceeding limits.
        """
        # Remove extra whitespace and count characters
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        # Conservative estimate: 1 token per 3 characters (instead of 4)
        # This accounts for tokenization variability and special characters
        estimated_tokens = len(cleaned_text) // 3
        return max(1, estimated_tokens)  # At least 1 token
    
    def _truncate_text(self, text: str, max_tokens: int = 512) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Input text to truncate
            max_tokens: Maximum allowed tokens (default: 512 for nv-embedqa)
            
        Returns:
            Truncated text that fits within token limit
        """
        if not text:
            return text
            
        estimated_tokens = self._estimate_token_count(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate approximate character limit
        # Use conservative estimate: 3 chars per token, with 80% safety buffer
        max_chars = int(max_tokens * 3 * 0.8)
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at word boundary to avoid cutting words
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > max_chars * 0.8:  # If we can find a space in the last 20%
            truncated = truncated[:last_space]
        
        logger.warning(f"⚠️ Truncated text from {len(text)} to {len(truncated)} characters "
                      f"(estimated {estimated_tokens} → {self._estimate_token_count(truncated)} tokens)")
        
        return truncated
    
    def _make_embedding_request(
        self, 
        texts: List[str], 
        input_type: str
    ) -> List[List[float]]:
        """Make direct API request to Capella embeddings endpoint."""
        
        # Construct URL
        if self.base_url.endswith("/v1"):
            url = f"{self.base_url}/embeddings"
        else:
            url = f"{self.base_url}/v1/embeddings"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Truncate texts if needed to fit within token limits
        truncated_texts = []
        for i, text in enumerate(texts):
            original_length = len(text)
            truncated = self._truncate_text(text)
            truncated_texts.append(truncated)
            
            if len(truncated) < original_length:
                logger.info(f"📏 Text {i+1}: {original_length} → {len(truncated)} chars, "
                           f"est. tokens: {self._estimate_token_count(truncated)}")
            else:
                logger.debug(f"📏 Text {i+1}: {original_length} chars, "
                            f"est. tokens: {self._estimate_token_count(text)}")
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "input": truncated_texts,
        }
        
        # Add input_type for models that need it
        if self.needs_input_type:
            payload["input_type"] = input_type
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making embedding request (attempt {attempt + 1}): {len(texts)} texts")
                
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    logger.debug(f"✅ Successfully got {len(embeddings)} embeddings")
                    return embeddings
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.warning(f"❌ {error_msg}")
                    
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(error_msg)
                    
                    # Wait before retry
                    time.sleep(2 ** attempt)
                    
            except requests.RequestException as e:
                error_msg = f"Request failed: {e}"
                logger.warning(f"❌ {error_msg}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError(error_msg)
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs using passage input_type."""
        if not texts:
            return []
        
        if self.needs_input_type:
            return self._make_embedding_request(texts, self.input_type_passage)
        else:
            # Fallback to standard OpenAI embeddings
            fallback = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            return fallback.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text using query input_type."""
        if self.needs_input_type:
            results = self._make_embedding_request([text], self.input_type_query)
            return results[0]
        else:
            # Fallback to standard OpenAI embeddings
            fallback = OpenAIEmbeddings(
                model=self.model,
                api_key=self.api_key,
                base_url=self.base_url,
            )
            return fallback.embed_query(text)


class CapellaChatLLM(BaseChatModel):
    """
    Custom Capella AI chat LLM class that properly handles API calls.
    
    This class makes direct API calls to Capella AI when needed,
    bypassing LangChain's URL formatting issues.
    """
    
    model: str = Field(...)
    api_key: str = Field(...)
    base_url: str = Field(...)
    temperature: float = Field(default=0.0)
    max_tokens: Optional[int] = Field(default=None)
    max_retries: int = Field(default=3)
    request_timeout: int = Field(default=60)
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
        request_timeout: int = 60,
        **kwargs
    ):
        # Pass all parameters to the parent constructor
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
            **kwargs
        )
        
        logger.info(f"✅ Using Capella direct API for LLM: {model}")
    
    def _convert_messages_to_api_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to OpenAI API format."""
        api_messages = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                # Default to user for unknown message types
                role = "user"
            
            api_messages.append({
                "role": role,
                "content": message.content
            })
        
        return api_messages
    
    def _make_chat_request(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Make direct API request to Capella chat completions endpoint."""
        
        # Construct URL
        if self.base_url.endswith("/v1"):
            url = f"{self.base_url}/chat/completions"
        else:
            url = f"{self.base_url}/v1/chat/completions"
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Convert messages to API format
        api_messages = self._convert_messages_to_api_format(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making chat completion request (attempt {attempt + 1})")
                
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"✅ Successfully got chat completion")
                    return result
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.warning(f"❌ {error_msg}")
                    
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(error_msg)
                    
                    # Wait before retry
                    time.sleep(2 ** attempt)
                    
            except requests.RequestException as e:
                error_msg = f"Request failed: {e}"
                logger.warning(f"❌ {error_msg}")
                
                if attempt == self.max_retries - 1:
                    raise RuntimeError(error_msg)
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        raise RuntimeError(f"Failed after {self.max_retries} attempts")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        try:
            # Make API request
            result = self._make_chat_request(messages)
            
            # Extract response content
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
                
                # Create response message
                ai_message = AIMessage(content=content)
                
                # Create generation
                generation = ChatGeneration(message=ai_message)
                
                # Return chat result
                return ChatResult(generations=[generation])
            else:
                raise RuntimeError(f"Invalid API response format: {result}")
                
        except Exception as e:
            logger.error(f"❌ Chat completion failed: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "capella_chat"


# Factory Functions

def create_capella_embeddings(
    model: Optional[str] = None,
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    input_type_for_query: str = "query",
    input_type_for_passage: str = "passage"
) -> CapellaEmbeddings:
    """
    Factory function to create Capella embeddings with proper configuration.
    
    Args:
        model: Model name (from CAPELLA_API_EMBEDDING_MODEL env var if not provided)
        api_key: Direct API key (from CAPELLA_API_EMBEDDINGS_KEY env var if not provided)
        base_url: Base URL (from CAPELLA_API_ENDPOINT env var if not provided)
        username: Username for base64 encoding (from CB_USERNAME env var if not provided)
        password: Password for base64 encoding (from CB_PASSWORD env var if not provided)
        input_type_for_query: Input type for query embeddings (default: "query")
        input_type_for_passage: Input type for passage embeddings (default: "passage")
    
    Returns:
        CapellaEmbeddings instance configured for the environment
    """
    
    # Get configuration from environment
    model = model or os.getenv("CAPELLA_API_EMBEDDING_MODEL")
    base_url = base_url or os.getenv("CAPELLA_API_ENDPOINT")
    
    if not model:
        raise ValueError("Model name is required (set CAPELLA_API_EMBEDDING_MODEL)")
    if not base_url:
        raise ValueError("Base URL is required (set CAPELLA_API_ENDPOINT)")
    
    # Determine API key
    auth_key = api_key or os.getenv("CAPELLA_API_EMBEDDINGS_KEY")
    
    if auth_key:
        logger.info("✅ Using direct Capella embeddings API key")
    else:
        # Generate from username/password
        username = username or os.getenv("CB_USERNAME")
        password = password or os.getenv("CB_PASSWORD")
        
        if not username or not password:
            raise ValueError(
                "Either CAPELLA_API_EMBEDDINGS_KEY or both CB_USERNAME and CB_PASSWORD are required"
            )
        
        auth_key = base64.b64encode(f"{username}:{password}".encode()).decode()
        logger.info("✅ Generated Capella embeddings API key from username:password")
    
    return CapellaEmbeddings(
        model=model,
        api_key=auth_key,
        base_url=base_url,
        input_type_query=input_type_for_query,
        input_type_passage=input_type_for_passage,
    )


def create_capella_chat_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None, 
    base_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    callbacks: Optional[List] = None
) -> CapellaChatLLM:
    """
    Factory function to create Capella chat LLM with proper configuration.
    
    Args:
        model: Model name (from CAPELLA_API_LLM_MODEL env var if not provided)
        api_key: Direct API key (from CAPELLA_API_LLM_KEY env var if not provided)
        base_url: Base URL (from CAPELLA_API_ENDPOINT env var if not provided)
        username: Username for base64 encoding (from CB_USERNAME env var if not provided)
        password: Password for base64 encoding (from CB_PASSWORD env var if not provided)
        temperature: Temperature for generation (default: 0.0)
        max_tokens: Maximum tokens to generate (default: None)
        callbacks: Callbacks for LangChain integration (default: None)
    
    Returns:
        CapellaChatLLM instance configured for the environment
    """
    
    # Get configuration from environment
    model = model or os.getenv("CAPELLA_API_LLM_MODEL")
    base_url = base_url or os.getenv("CAPELLA_API_ENDPOINT")
    
    if not model:
        raise ValueError("Model name is required (set CAPELLA_API_LLM_MODEL)")
    if not base_url:
        raise ValueError("Base URL is required (set CAPELLA_API_ENDPOINT)")
    
    # Determine API key
    auth_key = api_key or os.getenv("CAPELLA_API_LLM_KEY")
    
    if auth_key:
        logger.info("✅ Using direct Capella LLM API key")
    else:
        # Generate from username/password
        username = username or os.getenv("CB_USERNAME")
        password = password or os.getenv("CB_PASSWORD")
        
        if not username or not password:
            raise ValueError(
                "Either CAPELLA_API_LLM_KEY or both CB_USERNAME and CB_PASSWORD are required"
            )
        
        auth_key = base64.b64encode(f"{username}:{password}".encode()).decode()
        logger.info("✅ Generated Capella LLM API key from username:password")
    
    return CapellaChatLLM(
        model=model,
        api_key=auth_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks or [],
    )