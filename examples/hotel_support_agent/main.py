#!/usr/bin/env python3
"""
Hotel Support Agent - Agent Catalog + LangChain Implementation

A streamlined hotel support agent demonstrating Agent Catalog integration
with LangChain and Couchbase vector search for hotel booking assistance.
"""

import base64
import getpass
import json
import logging
import os
import sys
import time
from datetime import timedelta

import agentc
import agentc_langchain
import dotenv
import requests
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import hotel data from the data module
from data.hotel_data import get_hotel_texts

# Setup logging with essential level only
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("agentc_core").setLevel(logging.WARNING)

# Load environment variables
dotenv.load_dotenv(override=True)


def setup_capella_ai_config():
    """Setup Capella AI configuration - requires environment variables to be set."""
    # Verify required environment variables are set (no defaults)
    required_capella_vars = [
        "CB_USERNAME",
        "CB_PASSWORD",
        "CAPELLA_API_ENDPOINT",
        "CAPELLA_API_EMBEDDING_MODEL",
        "CAPELLA_API_LLM_MODEL",
    ]
    missing_vars = [var for var in required_capella_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required Capella AI environment variables: {missing_vars}")

    return {
        "endpoint": os.getenv("CAPELLA_API_ENDPOINT"),
        "embedding_model": os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
        "llm_model": os.getenv("CAPELLA_API_LLM_MODEL"),
        "dimensions": 4096,
    }


def test_capella_connectivity():
    """Test connectivity to Capella AI services."""
    try:
        endpoint = os.getenv("CAPELLA_API_ENDPOINT")
        if not endpoint:
            logger.warning("CAPELLA_API_ENDPOINT not configured")
            return False

        # Test embedding model (requires API key)
        if os.getenv("CB_USERNAME") and os.getenv("CB_PASSWORD"):
            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()

            headers = {"Authorization": f"Basic {api_key}", "Content-Type": "application/json"}

            # Test embedding
            logger.info("Testing Capella AI connectivity...")
            embedding_data = {
                "model": os.getenv(
                    "CAPELLA_API_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
                ),
                "input": "test connectivity",
            }

            embedding_response = requests.post(
                f"{endpoint}/embeddings", headers=headers, json=embedding_data, timeout=30
            )

            if embedding_response.status_code == 200:
                embed_result = embedding_response.json()
                embed_dims = len(embed_result["data"][0]["embedding"])
                logger.info(f"✅ Capella AI embedding test successful - dimensions: {embed_dims}")

                if embed_dims != 4096:
                    logger.warning(f"Expected 4096 dimensions, got {embed_dims}")
                    return False
            else:
                logger.warning(
                    f"Capella AI embedding test failed: {embedding_response.status_code}"
                )
                logger.warning(f"Response: {embedding_response.text[:200]}...")
                return False

            # Test LLM
            llm_data = {
                "model": os.getenv("CAPELLA_API_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
            }

            llm_response = requests.post(
                f"{endpoint}/chat/completions", headers=headers, json=llm_data, timeout=30
            )

            if llm_response.status_code == 200:
                logger.info("✅ Capella AI LLM test successful")
            else:
                logger.warning(f"Capella AI LLM test failed: {llm_response.status_code}")
                logger.warning(f"Response: {llm_response.text[:200]}...")
                return False

        logger.info("✅ Capella AI connectivity tests completed successfully")
        return True

    except Exception as e:
        logger.warning(f"Capella AI connectivity test failed: {e}")
        return False


def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup required environment variables with defaults."""
    # Setup Capella AI configuration first
    setup_capella_ai_config()

    # Required variables
    required_vars = ["OPENAI_API_KEY", "CB_CONN_STRING", "CB_USERNAME", "CB_PASSWORD", "CB_BUCKET"]
    for var in required_vars:
        _set_if_undefined(var)

    defaults = {
        "CB_CONN_STRING": "couchbase://localhost",
        "CB_USERNAME": "Administrator",
        "CB_PASSWORD": "password",
        "CB_BUCKET": "vector-search-testing",
    }

    for key, default_value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = input(f"Enter {key} (default: {default_value}): ") or default_value

    os.environ["CB_INDEX"] = os.getenv("CB_INDEX", "hotel_data_index")
    os.environ["CB_SCOPE"] = os.getenv("CB_SCOPE", "agentc_data")
    os.environ["CB_COLLECTION"] = os.getenv("CB_COLLECTION", "hotel_data")

    # Generate Capella AI API key from username and password if endpoint is provided
    if os.getenv('CAPELLA_API_ENDPOINT'):
        os.environ['CAPELLA_API_KEY'] = base64.b64encode(
            f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode("utf-8")
        ).decode("utf-8")
        
        # Ensure endpoint has /v1 suffix for OpenAI compatibility
        if not os.getenv('CAPELLA_API_ENDPOINT').endswith('/v1'):
            os.environ['CAPELLA_API_ENDPOINT'] = os.getenv('CAPELLA_API_ENDPOINT').rstrip('/') + '/v1'
            logger.info(f"Added /v1 suffix to endpoint: {os.getenv('CAPELLA_API_ENDPOINT')}")

    # Test Capella AI connectivity
    test_capella_connectivity()


class CouchbaseClient:
    """Centralized Couchbase client for all database operations."""

    def __init__(self, conn_string: str, username: str, password: str, bucket_name: str):
        """Initialize Couchbase client with connection details."""
        self.conn_string = conn_string
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self.cluster = None
        self.bucket = None
        self._collections = {}

    def connect(self):
        """Establish connection to Couchbase cluster."""
        try:
            auth = PasswordAuthenticator(self.username, self.password)
            options = ClusterOptions(auth)
            
            # Use WAN profile for better timeout handling with remote clusters
            options.apply_profile("wan_development")
            
            # Additional timeout configurations for Capella cloud connections
            from couchbase.options import (
                ClusterTimeoutOptions,
                ClusterTracingOptions,
            )
            
            # Configure extended timeouts for cloud connectivity
            timeout_options = ClusterTimeoutOptions(
                kv_timeout=timedelta(seconds=10),  # Key-value operations
                kv_durable_timeout=timedelta(seconds=15),  # Durable writes
                query_timeout=timedelta(seconds=30),  # N1QL queries
                search_timeout=timedelta(seconds=30),  # Search operations
                management_timeout=timedelta(seconds=30),  # Management operations
                bootstrap_timeout=timedelta(seconds=20),  # Initial connection
            )
            options.timeout_options = timeout_options
            
            self.cluster = Cluster(self.conn_string, options)
            # Increased wait time for cloud connections
            self.cluster.wait_until_ready(timedelta(seconds=20))
            logger.info("Successfully connected to Couchbase")
            return self.cluster
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Couchbase: {e!s}")

    def setup_collection(self, scope_name: str, collection_name: str):
        """Setup bucket, scope and collection all in one function."""
        try:
            # Ensure cluster connection
            if not self.cluster:
                self.connect()

            # Setup bucket
            if not self.bucket:
                try:
                    self.bucket = self.cluster.bucket(self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' exists")
                except Exception:
                    logger.info(f"Creating bucket '{self.bucket_name}'...")
                    bucket_settings = CreateBucketSettings(
                        name=self.bucket_name,
                        bucket_type="couchbase",
                        ram_quota_mb=1024,
                        flush_enabled=True,
                        num_replicas=0,
                    )
                    self.cluster.buckets().create_bucket(bucket_settings)
                    time.sleep(5)
                    self.bucket = self.cluster.bucket(self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' created successfully")

            # Setup scope
            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                logger.info(f"Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                logger.info(f"Scope '{scope_name}' created successfully")

            # Setup collection
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if not collection_exists:
                logger.info(f"Creating collection '{collection_name}'...")
                bucket_manager.create_collection(scope_name, collection_name)
                logger.info(f"Collection '{collection_name}' created successfully")

            time.sleep(3)

            # Create primary index
            try:
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
                ).execute()
                logger.info("Primary index created successfully")
            except Exception as e:
                logger.warning(f"Error creating primary index: {e}")

            logger.info("Collection setup complete")
            return self.bucket.scope(scope_name).collection(collection_name)

        except Exception as e:
            raise RuntimeError(f"Error setting up collection: {e!s}")

    def get_collection(self, scope_name: str, collection_name: str):
        """Get a collection object."""
        key = f"{scope_name}.{collection_name}"
        if key not in self._collections:
            self._collections[key] = self.bucket.scope(scope_name).collection(collection_name)
        return self._collections[key]

    def setup_vector_search_index(self, index_definition: dict, scope_name: str):
        """Setup vector search index."""
        try:
            scope_index_manager = self.bucket.scope(scope_name).search_indexes()
            
            existing_indexes = scope_index_manager.get_all_indexes()
            index_name = index_definition["name"]

            if index_name not in [index.name for index in existing_indexes]:
                logger.info(f"Creating vector search index '{index_name}'...")
                search_index = SearchIndex.from_json(index_definition)
                scope_index_manager.upsert_index(search_index)
                logger.info(f"Vector search index '{index_name}' created successfully")
            else:
                logger.info(f"Vector search index '{index_name}' already exists")
        except Exception as e:
            raise RuntimeError(f"Error setting up vector search index: {e!s}")

    def load_hotel_data(self, scope_name, collection_name, index_name, embeddings):
        """Load hotel data into Couchbase."""
        try:
            # Clear existing data first
            self.clear_collection_data(scope_name, collection_name)
            
            # Setup vector store
            vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )
            
            # Load hotel data using the data loading script
            try:
                from data.hotel_data import load_hotel_data_to_couchbase
                load_hotel_data_to_couchbase(
                    cluster=self.cluster,
                    bucket_name=self.bucket_name,
                    scope_name=scope_name,
                    collection_name=collection_name,
                    embeddings=embeddings,
                    index_name=index_name
                )
                logger.info("Hotel data loaded into vector store successfully using data loading script")
            except Exception as e:
                logger.warning(f"Error loading hotel data with script: {e}. Falling back to direct method.")
                # Fallback to the original method
                hotel_data = get_hotel_texts()
                vector_store.add_texts(texts=hotel_data, batch_size=10)
                logger.info("Hotel data loaded into vector store successfully using fallback method")
                
        except Exception as e:
            raise RuntimeError(f"Error loading hotel data: {e!s}")

    def setup_vector_store(
        self, scope_name: str, collection_name: str, index_name: str, embeddings
    ):
        """Setup vector store with hotel data."""
        try:
            # Use the embeddings parameter passed in - no fallbacks
            if not embeddings:
                raise RuntimeError("Embeddings parameter is required - no fallbacks available")
            
            logger.info("✅ Using provided embeddings for vector store setup")

            # Load hotel data
            self.load_hotel_data(scope_name, collection_name, index_name, embeddings)

            # Create vector store
            vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )

            logger.info("Vector store setup complete")
            return vector_store

        except Exception as e:
            raise RuntimeError(f"Error setting up vector store: {e!s}")

    def clear_collection_data(self, scope_name: str, collection_name: str):
        """Clear all documents from the collection to start fresh."""
        try:
            # Delete all documents in the collection
            delete_query = f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
            result = self.cluster.query(delete_query)
            
            logger.info(f"Cleared existing data from collection {scope_name}.{collection_name}")
            
        except Exception as e:
            logger.warning(f"Could not clear collection data: {e}. Continuing with existing data...")

    def clear_scope(self, scope_name: str):
        """Clear all collections in scope."""
        try:
            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()
            
            for scope in scopes:
                if scope.name == scope_name:
                    for collection in scope.collections:
                        self.clear_collection_data(scope_name, collection.name)
            
            logger.info(f"Cleared all collections in scope: {scope_name}")
            
        except Exception as e:
            logger.warning(f"Could not clear scope: {e}")


def clear_hotel_data():
    """Clear existing hotel data from the database."""
    try:
        couchbase_client = CouchbaseClient(
            conn_string=os.getenv("CB_CONN_STRING"),
            username=os.getenv("CB_USERNAME"),
            password=os.getenv("CB_PASSWORD"),
            bucket_name=os.getenv("CB_BUCKET")
        )
        
        couchbase_client.connect()
        couchbase_client.bucket = couchbase_client.cluster.bucket(os.getenv("CB_BUCKET"))
        
        # Clear hotel data
        couchbase_client.clear_collection_data(
            os.getenv("CB_SCOPE"), 
            os.getenv("CB_COLLECTION")
        )
        
        logger.info("Hotel data cleared successfully")
        
    except Exception as e:
        logger.warning(f"Could not clear hotel data: {e}")


def setup_hotel_support_agent():
    """Setup the hotel support agent with all required components."""
    try:
        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Hotel Support Agent")

        with application_span.new("Environment Setup"):
            setup_environment()

        with application_span.new("Capella AI Test"):
            if os.getenv('CAPELLA_API_ENDPOINT'):
                if not test_capella_connectivity():
                    logger.warning("❌ Capella AI connectivity test failed. Will use OpenAI fallback.")
            else:
                logger.info("ℹ️ Capella API not configured - will use OpenAI models")

        with application_span.new("Couchbase Connection"):
            couchbase_client = CouchbaseClient(
                conn_string=os.getenv("CB_CONN_STRING"),
                username=os.getenv("CB_USERNAME"),
                password=os.getenv("CB_PASSWORD"),
                bucket_name=os.getenv("CB_BUCKET")
            )
            
            couchbase_client.connect()

        with application_span.new("Couchbase Collection Setup"):
            couchbase_client.setup_collection(
                os.getenv("CB_SCOPE"),
                os.getenv("CB_COLLECTION")
            )

        with application_span.new("Vector Index Setup"):
            try:
                with open('agentcatalog_index.json', 'r') as file:
                    index_definition = json.load(file)
                logger.info("Loaded vector search index definition from agentcatalog_index.json")
            except Exception as e:
                raise ValueError(f"Error loading index definition: {e!s}")
            
            couchbase_client.setup_vector_search_index(index_definition, os.getenv("CB_SCOPE"))

        with application_span.new("Vector Store Setup"):
            # Setup embeddings using CB_USERNAME/CB_PASSWORD like flight search agent
            try:
                if (
                    os.getenv("CB_USERNAME")
                    and os.getenv("CB_PASSWORD")
                    and os.getenv("CAPELLA_API_ENDPOINT")
                    and os.getenv("CAPELLA_API_EMBEDDING_MODEL")
                ):
                    # Create API key for Capella AI
                    import base64
                    api_key = base64.b64encode(
                        f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
                    ).decode()

                    # Use OpenAI embeddings client with Capella endpoint
                    embeddings = OpenAIEmbeddings(
                        model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                        api_key=api_key,
                        base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                    )
                    logger.info("✅ Using Capella AI for embeddings (4096 dimensions)")
                else:
                    raise ValueError("Capella AI credentials not available")
            except Exception as e:
                logger.error(f"❌ Capella AI embeddings failed: {e}")
                raise RuntimeError("Capella AI embeddings required for this configuration")
            
            couchbase_client.setup_vector_store(
                os.getenv("CB_SCOPE"),
                os.getenv("CB_COLLECTION"),
                os.getenv("CB_INDEX"),
                embeddings
            )

        with application_span.new("LLM Setup"):
            # Setup LLM with Agent Catalog callback - try Capella AI first, fallback to OpenAI
            try:
                # Create API key for Capella AI using same pattern as embeddings
                api_key = base64.b64encode(
                    f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
                ).decode()
                
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url=os.getenv('CAPELLA_API_ENDPOINT'),
                    model=os.getenv('CAPELLA_API_LLM_MODEL'),
                    temperature=0,
                    callbacks=[agentc_langchain.chat.Callback(span=application_span)]
                )
                # Test the LLM works
                llm.invoke("Hello")
                logger.info("✅ Using Capella AI LLM")
            except Exception as e:
                logger.warning(f"⚠️ Capella AI LLM failed: {e}")
                logger.info("🔄 Falling back to OpenAI LLM...")
                _set_if_undefined("OPENAI_API_KEY")
                llm = ChatOpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    model="gpt-4o",
                    temperature=0,
                    callbacks=[agentc_langchain.chat.Callback(span=application_span)]
                )
                logger.info("✅ Using OpenAI LLM as fallback")

        with application_span.new("Tool Loading"):
            # Load tools from Agent Catalog - they are now properly decorated
            tool_search = catalog.find("tool", name="search_vector_database")
            tool_details = catalog.find("tool", name="get_hotel_details")
            
            if not tool_search:
                raise ValueError("Could not find search_vector_database tool. Make sure it's indexed with 'agentc index tools/'")
            if not tool_details:
                raise ValueError("Could not find get_hotel_details tool. Make sure it's indexed with 'agentc index tools/'")
            
            tools = [
                Tool(
                    name=tool_search.meta.name,
                    description=tool_search.meta.description,
                    func=tool_search.func
                ),
                Tool(
                    name=tool_details.meta.name, 
                    description=tool_details.meta.description,
                    func=tool_details.func
                )
            ]

        with application_span.new("Agent Creation"):
            # Get prompt from Agent Catalog
            hotel_prompt = catalog.find("prompt", name="hotel_search_assistant")
            if not hotel_prompt:
                raise ValueError("Could not find hotel_search_assistant prompt in catalog. Make sure it's indexed with 'agentc index prompts/'")
            
            # Create a custom prompt using the catalog prompt content
            prompt_content = hotel_prompt.content.strip()
            
            custom_prompt = PromptTemplate(
                template=prompt_content,
                input_variables=["input", "agent_scratchpad"],
                partial_variables={
                    "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    "tool_names": ", ".join([tool.name for tool in tools])
                }
            )
            
            agent = create_react_agent(llm, tools, custom_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                handle_parsing_errors=True,
                max_iterations=3,  # Reduced to prevent infinite loops
                return_intermediate_steps=True,
                early_stopping_method="force",
                max_execution_time=30  # 30 second timeout to prevent hanging
            )

        return agent_executor, application_span

    except Exception as e:
        logger.exception(f"Error setting up hotel support agent: {e}")
        raise


def run_interactive_demo():
    """Run an interactive hotel support demo."""
    logger.info("Hotel Support Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        agent_executor, application_span = setup_hotel_support_agent()

        # Interactive hotel search loop
        with application_span.new("Query Execution") as span:
            logger.info("Available commands:")
            logger.info("- Enter hotel search queries (e.g., 'Find luxury hotels with spa')")
            logger.info("- 'quit' - Exit the demo")
            logger.info(
                "Try asking: 'Find me a beach resort in Miami' or 'Get details about Ocean Breeze Resort'"
            )
            logger.info("─" * 40)

            while True:
                query = input("🔍 Enter hotel search query (or 'quit' to exit): ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    logger.info("Thanks for using Hotel Support Agent!")
                    break

                if not query:
                    continue

                with span.new(f"Query: {query}") as query_span:
                    try:
                        logger.info(f"Hotel Query: {query}")
                        query_span["query"] = query

                        # Execute the query
                        response = agent_executor.invoke({"input": query})
                        query_span["response"] = response['output']

                        # Display results
                        logger.info(f"✅ Response: {response['output']}")

                    except Exception as e:
                        logger.exception(f"Search error: {e}")
                        query_span["error"] = str(e)

                    logger.info("-" * 50)

    except Exception as e:
        logger.exception(f"Demo initialization error: {e}")


def run_test():
    """Run comprehensive test of hotel support agent with 3 test queries."""
    logger.info("Hotel Support Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        # Clear existing data first for a clean test run
        clear_hotel_data()

        agent_executor, application_span = setup_hotel_support_agent()

        # Test scenarios covering different types of hotel searches
        test_queries = [
            "Find me a luxury hotel with a pool and spa",
            "I need a beach resort in Miami for my vacation",
            "Get me details about Ocean Breeze Resort"
        ]

        with application_span.new("Test Queries") as span:
            for i, query in enumerate(test_queries, 1):
                with span.new(f"Test {i}: {query}") as query_span:
                    logger.info(f"\n🔍 Test {i}: {query}")
                    try:
                        query_span["query"] = query
                        response = agent_executor.invoke({"input": query})
                        query_span["response"] = response['output']

                        # Display the response
                        logger.info(f"🤖 AI Response: {response['output']}")
                        logger.info(f"✅ Test {i} completed successfully")

                    except Exception as e:
                        logger.exception(f"❌ Test {i} failed: {e}")
                        query_span["error"] = str(e)

                    logger.info("-" * 50)

        logger.info("All tests completed!")

    except Exception as e:
        logger.exception(f"Test error: {e}")


def run_hotel_support_demo():
    """Legacy function - redirects to interactive demo for compatibility."""
    run_interactive_demo()


def main():
    """Main entry point - runs interactive demo by default."""
    run_interactive_demo()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        else:
            run_interactive_demo()
    else:
        run_interactive_demo()
