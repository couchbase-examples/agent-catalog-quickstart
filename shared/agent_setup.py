#!/usr/bin/env python3
"""
Universal Agent Setup Module

Provides consistent 4-case priority AI service setup for all agent frameworks:
1. New Capella (direct API keys) - Custom classes
2. Old Capella (base64 encoding) - OpenAI wrappers  
3. NVIDIA NIM API - Native clients
4. OpenAI fallback - Native clients

Supports: LangChain, LlamaIndex, LangGraph
"""

import base64
import logging
import os
from typing import Tuple, Any, Optional, List

logger = logging.getLogger(__name__)


def setup_ai_services(
    framework: str = "langchain",
    temperature: float = 0.0,
    callbacks: Optional[List] = None,
    application_span: Optional[Any] = None
) -> Tuple[Any, Any]:
    """
    Universal AI service setup with 4-case priority ladder.
    
    Args:
        framework: "langchain", "llamaindex", or "langgraph" 
        temperature: LLM temperature setting
        callbacks: Optional callbacks for LangChain agents
        application_span: Optional span for observability
        
    Returns:
        Tuple[embeddings, llm]: Framework-appropriate instances
    """
    embeddings = None
    llm = None
    
    logger.info(f"🔧 Setting up AI services for {framework} framework...")
    
    # ====================================================================
    # 1. NEW CAPELLA MODEL SERVICES (with direct API key) - PRIORITY 1
    # ====================================================================
    if (
        not embeddings 
        and os.getenv("CAPELLA_API_ENDPOINT") 
        and os.getenv("CAPELLA_API_EMBEDDINGS_KEY")
    ):
        try:
            from .capella_model_services import create_capella_embeddings
            
            embeddings = create_capella_embeddings(
                api_key=os.getenv("CAPELLA_API_EMBEDDINGS_KEY"),
                base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                input_type_for_query="query",
                input_type_for_passage="passage"
            )
            logger.info("✅ Using new Capella AI embeddings (direct API key)")
        except Exception as e:
            logger.warning(f"⚠️ New Capella AI embeddings failed: {e}")

    if (
        not llm 
        and os.getenv("CAPELLA_API_ENDPOINT") 
        and os.getenv("CAPELLA_API_LLM_KEY")
    ):
        try:
            from .capella_model_services import create_capella_chat_llm
            
            # Framework-specific callback handling
            llm_callbacks = None
            if framework in ["langchain", "langgraph"] and callbacks:
                llm_callbacks = callbacks
            
            llm = create_capella_chat_llm(
                api_key=os.getenv("CAPELLA_API_LLM_KEY"),
                base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                model=os.getenv("CAPELLA_API_LLM_MODEL"),
                temperature=temperature,
                callbacks=llm_callbacks,
            )
            
            # Test the LLM works
            if framework == "llamaindex":
                # LlamaIndex has different interface
                llm.complete("Hello")
            else:
                llm.invoke("Hello")
                
            logger.info("✅ Using new Capella AI LLM (direct API key)")
        except Exception as e:
            logger.warning(f"⚠️ New Capella AI LLM failed: {e}")
            llm = None

    # ====================================================================
    # 2. OLD CAPELLA MODEL SERVICES (with base64 encoding) - PRIORITY 2  
    # ====================================================================
    if (
        not embeddings 
        and os.getenv("CAPELLA_API_ENDPOINT") 
        and os.getenv("CB_USERNAME") 
        and os.getenv("CB_PASSWORD")
    ):
        try:
            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()
            
            if framework == "llamaindex":
                from llama_index.embeddings.openai import OpenAIEmbedding
                embeddings = OpenAIEmbedding(
                    api_key=api_key,
                    api_base=os.getenv("CAPELLA_API_ENDPOINT"),
                    model_name=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                    embed_batch_size=30,
                )
            else:  # langchain, langgraph
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                    api_key=api_key,
                    base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                )
            logger.info("✅ Using old Capella AI embeddings (base64 encoding)")
        except Exception as e:
            logger.warning(f"⚠️ Old Capella AI embeddings failed: {e}")

    if (
        not llm 
        and os.getenv("CAPELLA_API_ENDPOINT") 
        and os.getenv("CB_USERNAME") 
        and os.getenv("CB_PASSWORD")
    ):
        try:
            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()
            
            if framework == "llamaindex":
                from llama_index.llms.openai_like import OpenAILike
                llm = OpenAILike(
                    model=os.getenv("CAPELLA_API_LLM_MODEL"),
                    api_base=f"{os.getenv('CAPELLA_API_ENDPOINT')}/v1",
                    api_key=api_key,
                    is_chat_model=True,
                    temperature=temperature,
                )
            else:  # langchain, langgraph
                from langchain_openai import ChatOpenAI
                
                # Add callbacks for LangChain/LangGraph
                chat_kwargs = {
                    "api_key": api_key,
                    "base_url": os.getenv("CAPELLA_API_ENDPOINT"),
                    "model": os.getenv("CAPELLA_API_LLM_MODEL"),
                    "temperature": temperature,
                }
                if callbacks:
                    chat_kwargs["callbacks"] = callbacks
                    
                llm = ChatOpenAI(**chat_kwargs)
                
            # Test the LLM works
            if framework == "llamaindex":
                llm.complete("Hello")
            else:
                llm.invoke("Hello")
                
            logger.info("✅ Using old Capella AI LLM (base64 encoding)")
        except Exception as e:
            logger.warning(f"⚠️ Old Capella AI LLM failed: {e}")
            llm = None

    # ====================================================================
    # 3. NVIDIA NIM API - PRIORITY 3
    # ====================================================================
    if not embeddings and os.getenv("NVIDIA_API_KEY"):
        try:
            if framework == "llamaindex":
                from llama_index.embeddings.nvidia import NVIDIAEmbedding
                embeddings = NVIDIAEmbedding(
                    model=os.getenv("NVIDIA_API_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    truncate="END",
                )
            else:  # langchain, langgraph
                from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
                embeddings = NVIDIAEmbeddings(
                    model=os.getenv("NVIDIA_API_EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    truncate="END",
                )
            logger.info("✅ Using NVIDIA NIM embeddings")
        except Exception as e:
            logger.warning(f"⚠️ NVIDIA NIM embeddings failed: {e}")

    if not llm and os.getenv("NVIDIA_API_KEY"):
        try:
            if framework == "llamaindex":
                from llama_index.llms.nvidia import NVIDIA
                llm = NVIDIA(
                    model=os.getenv("NVIDIA_API_LLM_MODEL", "meta/llama-3.1-70b-instruct"),
                    api_key=os.getenv("NVIDIA_API_KEY"),
                    temperature=temperature,
                )
            else:  # langchain, langgraph
                from langchain_nvidia_ai_endpoints import ChatNVIDIA
                
                chat_kwargs = {
                    "model": os.getenv("NVIDIA_API_LLM_MODEL", "meta/llama-3.1-70b-instruct"),
                    "api_key": os.getenv("NVIDIA_API_KEY"),
                    "temperature": temperature,
                }
                if callbacks:
                    chat_kwargs["callbacks"] = callbacks
                    
                llm = ChatNVIDIA(**chat_kwargs)
                
            logger.info("✅ Using NVIDIA NIM LLM")
        except Exception as e:
            logger.warning(f"⚠️ NVIDIA NIM LLM failed: {e}")

    # ====================================================================
    # 4. OPENAI FALLBACK - PRIORITY 4
    # ====================================================================
    if not embeddings and os.getenv("OPENAI_API_KEY"):
        try:
            if framework == "llamaindex":
                from llama_index.embeddings.openai import OpenAIEmbedding
                embeddings = OpenAIEmbedding(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small",
                )
            else:  # langchain, langgraph
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_API_ENDPOINT"),
                )
            logger.info("✅ Using OpenAI embeddings fallback")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI embeddings failed: {e}")

    if not llm and os.getenv("OPENAI_API_KEY"):
        try:
            if framework == "llamaindex":
                from llama_index.llms.openai_like import OpenAILike
                llm = OpenAILike(
                    model="gpt-4o",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    is_chat_model=True,
                    temperature=temperature,
                )
            else:  # langchain, langgraph
                from langchain_openai import ChatOpenAI
                
                chat_kwargs = {
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "model": "gpt-4o",
                    "temperature": temperature,
                }
                if callbacks:
                    chat_kwargs["callbacks"] = callbacks
                    
                llm = ChatOpenAI(**chat_kwargs)
                
            logger.info("✅ Using OpenAI LLM fallback")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI LLM failed: {e}")

    # ====================================================================
    # VALIDATION
    # ====================================================================
    if not embeddings:
        raise ValueError("❌ No embeddings service could be initialized")
    if not llm:
        raise ValueError("❌ No LLM service could be initialized")

    logger.info(f"✅ AI services setup completed for {framework}")
    return embeddings, llm


def setup_environment():
    """Setup default environment variables for agent operations."""
    # Set default values if not already defined
    defaults = {
        "CB_BUCKET": "travel-sample",
        "CB_SCOPE": "agentc_data", 
        "CB_COLLECTION": "hotel_data",
        "CB_INDEX": "hotel_data_index",
        "CAPELLA_API_EMBEDDING_MODEL": "nvidia/nv-embedqa-e5-v5",
        "CAPELLA_API_LLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
        "NVIDIA_API_EMBEDDING_MODEL": "nvidia/nv-embedqa-e5-v5",
        "NVIDIA_API_LLM_MODEL": "meta/llama-3.1-70b-instruct",
    }
    
    for key, value in defaults.items():
        if not os.getenv(key):
            os.environ[key] = value
            
    logger.info("✅ Environment variables configured")


def test_capella_connectivity(api_key: str = None, endpoint: str = None) -> bool:
    """Test connectivity to Capella AI services."""
    try:
        import httpx
        
        test_key = api_key or os.getenv("CAPELLA_API_EMBEDDINGS_KEY") or os.getenv("CAPELLA_API_LLM_KEY")
        test_endpoint = endpoint or os.getenv("CAPELLA_API_ENDPOINT")
        
        if not test_key or not test_endpoint:
            return False
            
        # Simple connectivity test
        headers = {"Authorization": f"Bearer {test_key}"}
        
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{test_endpoint.rstrip('/')}/v1/models", headers=headers)
            return response.status_code < 500  # Accept any non-server error
            
    except Exception as e:
        logger.warning(f"⚠️ Capella connectivity test failed: {e}")
        return False


