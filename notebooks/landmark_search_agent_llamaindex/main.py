#!/usr/bin/env python3
"""
Landmark Search Agent - Agent Catalog + LlamaIndex Implementation

A streamlined landmark search agent demonstrating Agent Catalog integration
with LlamaIndex and Couchbase vector search for landmark discovery assistance.
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
import dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

# Import landmark data from the data module
from data.landmark_data import load_landmark_data_to_couchbase

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Reduce noise from various libraries during embedding/vector operations
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Load environment variables
dotenv.load_dotenv(override=True)

# Set default values for travel-sample bucket configuration
DEFAULT_BUCKET = "travel-sample"
DEFAULT_SCOPE = "agentc_data"
DEFAULT_COLLECTION = "landmark_data"
DEFAULT_INDEX = "landmark_data_index"
DEFAULT_CAPELLA_API_EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
DEFAULT_CAPELLA_API_LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_NVIDIA_API_LLM_MODEL = "meta/llama-3.1-70b-instruct"


def _set_if_undefined(env_var: str, default_value: str = None):
    """Set environment variable if not already defined."""
    if not os.getenv(env_var):
        if default_value is None:
            value = getpass.getpass(f"Enter {env_var}: ")
        else:
            value = default_value
        os.environ[env_var] = value


def setup_environment():
    """Setup required environment variables with defaults for travel-sample configuration."""
    logger.info("Setting up environment variables...")

    # Set default bucket configuration
    _set_if_undefined("CB_BUCKET", DEFAULT_BUCKET)
    _set_if_undefined("CB_SCOPE", DEFAULT_SCOPE)
    _set_if_undefined("CB_COLLECTION", DEFAULT_COLLECTION)
    _set_if_undefined("CB_INDEX", DEFAULT_INDEX)

    # Required Couchbase connection variables (no defaults)
    _set_if_undefined("CB_CONN_STRING")
    _set_if_undefined("CB_USERNAME")
    _set_if_undefined("CB_PASSWORD")
    
    # NVIDIA NIMs API key (for LLM)
    _set_if_undefined("NVIDIA_API_KEY")

    # Optional Capella AI variables
    optional_vars = ["CAPELLA_API_ENDPOINT", "CAPELLA_API_EMBEDDING_MODEL", "CAPELLA_API_LLM_MODEL"]
    for var in optional_vars:
        if not os.environ.get(var):
            print(f"ℹ️ {var} not provided - will use OpenAI fallback")

    # Set Capella model defaults
    _set_if_undefined("CAPELLA_API_EMBEDDING_MODEL", DEFAULT_CAPELLA_API_EMBEDDING_MODEL)
    _set_if_undefined("CAPELLA_API_LLM_MODEL", DEFAULT_CAPELLA_API_LLM_MODEL)

    # Generate Capella AI API key if endpoint is provided
    if os.environ.get("CAPELLA_API_ENDPOINT"):
        os.environ["CAPELLA_API_KEY"] = base64.b64encode(
            f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode("utf-8")
        ).decode("utf-8")

        print(f"Using Capella AI endpoint for embeddings: {os.environ['CAPELLA_API_ENDPOINT']}")
        print(f"Using NVIDIA NIMs for LLM with API key: {os.environ['NVIDIA_API_KEY'][:10]}...")

    # Validate configuration consistency
    print(f"✅ Configuration loaded:")
    print(f"   Bucket: {os.environ['CB_BUCKET']}")
    print(f"   Scope: {os.environ['CB_SCOPE']}")
    print(f"   Collection: {os.environ['CB_COLLECTION']}")
    print(f"   Index: {os.environ['CB_INDEX']}")

    # Validate configuration consistency
    print(f"✅ Configuration loaded:")
    print(f"   Bucket: {os.environ['CB_BUCKET']}")
    print(f"   Scope: {os.environ['CB_SCOPE']}")
    print(f"   Collection: {os.environ['CB_COLLECTION']}")
    print(f"   Index: {os.environ['CB_INDEX']}")


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
            self.cluster = Cluster(self.conn_string, options)
            self.cluster.wait_until_ready(timedelta(seconds=20))
            logger.info("Successfully connected to Couchbase")
            return self.cluster
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Couchbase: {e!s}")

    def setup_collection(self, scope_name: str, collection_name: str):
        """Setup collection - create scope and collection if they don't exist, but don't clear scope."""
        try:
            # Ensure cluster connection
            if not self.cluster:
                self.connect()

            # For travel-sample bucket, assume it exists
            if not self.bucket:
                self.bucket = self.cluster.bucket(self.bucket_name)
                logger.info(f"Connected to bucket '{self.bucket_name}'")

            # Setup scope
            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                logger.info(f"Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                logger.info(f"Scope '{scope_name}' created successfully")

            # Setup collection - clear if exists, create if doesn't
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name
                and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if collection_exists:
                logger.info(f"Collection '{collection_name}' exists, clearing data...")
                # Clear existing data
                self.clear_collection_data(scope_name, collection_name)
            else:
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

    def clear_collection_data(self, scope_name: str, collection_name: str):
        """Clear all data from a collection."""
        try:
            logger.info(f"Clearing data from {self.bucket_name}.{scope_name}.{collection_name}...")

            # Use N1QL to delete all documents with explicit execution
            delete_query = f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
            result = self.cluster.query(delete_query)

            # Execute the query and get the results
            rows = list(result)

            # Wait a moment for the deletion to propagate
            time.sleep(2)

            # Verify collection is empty
            count_query = f"SELECT COUNT(*) as count FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
            count_result = self.cluster.query(count_query)
            count_row = list(count_result)[0]
            remaining_count = count_row["count"]

            if remaining_count == 0:
                logger.info(
                    f"Collection cleared successfully, {remaining_count} documents remaining"
                )
            else:
                logger.warning(
                    f"Collection clear incomplete, {remaining_count} documents remaining"
                )

        except Exception as e:
            logger.warning(f"Error clearing collection data: {e}")
            # If N1QL fails, try to continue anyway
            pass

    def get_collection(self, scope_name: str, collection_name: str):
        """Get a collection object."""
        key = f"{scope_name}.{collection_name}"
        if key not in self._collections:
            self._collections[key] = self.bucket.scope(scope_name).collection(collection_name)
        return self._collections[key]

    def setup_vector_search_index(self, index_definition: dict, scope_name: str):
        """Setup vector search index for the specified scope."""
        try:
            if not self.bucket:
                raise RuntimeError("Bucket not initialized. Call setup_collection first.")

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

    def load_landmark_data(self, scope_name, collection_name, index_name, embeddings):
        """Load landmark data into Couchbase."""
        try:
            # Load landmark data using the data loading script
            load_landmark_data_to_couchbase(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embeddings=embeddings,
                index_name=index_name,
            )
            logger.info("Landmark data loaded into vector store successfully")

        except Exception as e:
            raise RuntimeError(f"Error loading landmark data: {e!s}")

    def setup_vector_store(self, catalog, span):
        """Setup vector store with landmark data."""
        # Setup LLM and embeddings
        llm = None
        embeddings = None
        
        try:
            if os.environ.get("CAPELLA_API_ENDPOINT"):
                # Use NVIDIA NIMs for LLM, Capella for embeddings
                llm = NVIDIA(
                    model=DEFAULT_NVIDIA_API_LLM_MODEL,
                    api_key=os.environ["NVIDIA_API_KEY"],
                    temperature=0.1,
                )
                embeddings = OpenAIEmbedding(
                    api_key=os.environ["CAPELLA_API_KEY"],
                    api_base=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
                    model_name=os.environ["CAPELLA_API_EMBEDDING_MODEL"],
                    embed_batch_size=30,
                )
                # Comment out old Capella LLM configuration
                # llm = OpenAILike(
                #     model=os.environ["CAPELLA_API_LLM_MODEL"],
                #     api_base=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
                #     api_key=os.environ["CAPELLA_API_KEY"],
                #     is_chat_model=True,
                #     temperature=0.1,
                # )
            else:
                # Fallback to OpenAI if Capella not available
                llm = OpenAILike(
                    model="gpt-4o",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    is_chat_model=True,
                    temperature=0.1,
                )
                logger.exception("Capella API not available, cannot use fallback embeddings")
        except Exception as e:
            raise ValueError(f"Error setting up LLM and embeddings: {e!s}")

        # Set global settings
        try:
            Settings.llm = llm
            Settings.embed_model = embeddings
        except Exception as e:
            raise ValueError(f"Error setting global settings: {e!s}")

        # Setup collection
        try:
            self.setup_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
        except Exception as e:
            raise ValueError(f"Error setting up collection: {e!s}")

        # Setup vector search index
        try:
            with open("agentcatalog_index.json") as file:
                index_definition = json.load(file)
            logger.info("Loaded vector search index definition from agentcatalog_index.json")
            self.setup_vector_search_index(index_definition, os.environ["CB_SCOPE"])
        except FileNotFoundError:
            logger.warning("agentcatalog_index.json not found, continuing without vector search index...")
        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing index definition JSON: {e!s}")
            logger.info("Continuing without vector search index...")
        except Exception as e:
            logger.warning(f"Error setting up vector search index: {e!s}")
            logger.info("Continuing without vector search index...")

        # Load landmark data
        try:
            self.load_landmark_data(
                os.environ["CB_SCOPE"],
                os.environ["CB_COLLECTION"],
                os.environ["CB_INDEX"],
                embeddings,
            )
        except Exception as e:
            raise ValueError(f"Error loading landmark data: {e!s}")

        # Create vector store
        try:
            vector_store = CouchbaseSearchVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=os.environ["CB_SCOPE"],
                collection_name=os.environ["CB_COLLECTION"],
                index_name=os.environ["CB_INDEX"],
            )
            return vector_store
        except Exception as e:
            raise ValueError(f"Error creating vector store: {e!s}")

    def create_llamaindex_agent(self, catalog, span):
        """Create LlamaIndex ReAct agent with landmark search tool from Agent Catalog."""
        try:
            # Get tools from Agent Catalog
            tools = []

            # Search landmarks tool
            search_tool_result = catalog.find("tool", name="search_landmarks")
            if search_tool_result:
                tools.append(
                    FunctionTool.from_defaults(
                        fn=search_tool_result.func,
                        name="search_landmarks",
                        description=getattr(search_tool_result.meta, "description", None)
                        or "Search for landmark information using semantic vector search. Use for finding attractions, monuments, museums, parks, and other points of interest.",
                    )
                )
                logger.info("Loaded search_landmarks tool from AgentC")

            if not tools:
                logger.warning("No tools found in Agent Catalog")
            else:
                logger.info(f"Loaded {len(tools)} tools from Agent Catalog")

            # Get prompt from Agent Catalog - REQUIRED, no fallbacks
            prompt_result = catalog.find("prompt", name="landmark_search_assistant")
            if not prompt_result:
                raise RuntimeError("Prompt 'landmark_search_assistant' not found in Agent Catalog")

            # Try different possible attributes for the prompt content
            system_prompt = (
                getattr(prompt_result, "content", None)
                or getattr(prompt_result, "template", None)
                or getattr(prompt_result, "text", None)
            )
            if not system_prompt:
                raise RuntimeError(
                    "Could not access prompt content from AgentC - prompt content is None or empty"
                )

            logger.info("Loaded system prompt from Agent Catalog")

            # Create ReAct agent with limits to prevent excessive iterations
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=Settings.llm,
                verbose=True,  # Turn back on for debugging
                system_prompt=system_prompt,
                max_iterations=12,  # Medium level - enough for complex queries, not too much
            )

            logger.info("LlamaIndex ReAct agent created successfully")
            return agent

        except Exception as e:
            raise RuntimeError(f"Error creating LlamaIndex agent: {e!s}")


def setup_landmark_agent():
    """Setup the complete landmark search agent infrastructure and return the agent."""
    setup_environment()

    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    span = catalog.Span(name="Landmark Search Agent Setup")

    # Setup database client
    client = CouchbaseClient(
        conn_string=os.environ["CB_CONN_STRING"],
        username=os.environ["CB_USERNAME"],
        password=os.environ["CB_PASSWORD"],
        bucket_name=os.environ["CB_BUCKET"],
    )

    # Setup vector store
    vector_store = client.setup_vector_store(catalog, span)

    # Create agent
    agent = client.create_llamaindex_agent(catalog, span)

    return agent, client


def run_interactive_demo():
    """Run an interactive landmark search demo."""
    logger.info("Landmark Search Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        agent, client = setup_landmark_agent()

        # Interactive landmark search loop
        logger.info("Available commands:")
        logger.info("- Enter landmark search queries (e.g., 'Find landmarks in Paris')")
        logger.info("- 'quit' - Exit the demo")
        logger.info("Try asking: 'Find me landmarks in Tokyo' or 'Show me museums in London'")
        logger.info("─" * 40)

        while True:
            query = input("🔍 Enter landmark search query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using Landmark Search Agent!")
                break

            if not query:
                logger.warning("Please enter a search query")
                continue

            try:
                response = agent.chat(query, chat_history=[])
                result = response.response

                print(f"\n🏛️ Agent Response:\n{result}\n")
                print("─" * 40)

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"❌ Error: {e}")
                print("─" * 40)

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.exception(f"Demo error: {e}")
    finally:
        logger.info("Demo completed")


def run_test():
    """Run comprehensive test of landmark search agent with queries from queries.py."""
    logger.info("Landmark Search Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        agent, client = setup_landmark_agent()

        # Import shared queries
        from data.queries import get_queries_for_evaluation

        # Test scenarios covering different types of landmark searches
        test_queries = get_queries_for_evaluation()

        logger.info(f"Running {len(test_queries)} test queries...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}: {query}")
            try:
                response = agent.chat(query, chat_history=[])
                result = response.response

                # Display the response
                logger.info(f"🤖 AI Response: {result}")
                logger.info(f"✅ Test {i} completed successfully")

            except Exception as e:
                logger.exception(f"❌ Test {i} failed: {e}")

            logger.info("-" * 50)

        logger.info("All tests completed!")

    except Exception as e:
        logger.exception(f"Test error: {e}")


def main():
    """Main entry point - runs interactive demo by default."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        else:
            run_interactive_demo()
    else:
        run_interactive_demo()


if __name__ == "__main__":
    main()
