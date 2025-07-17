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
from agentc_llamaindex.chat import Callback
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

# Import landmark data from the data module
from data.landmark_data import get_landmark_texts, load_landmark_data_to_couchbase

# Import queries for testing
from data.queries import LANDMARK_SEARCH_QUERIES, get_queries_for_evaluation, get_reference_answer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(override=True)


def _set_if_undefined(var: str):
    """Helper function to prompt for missing environment variables."""
    if os.environ.get(var) is None:
        import getpass

        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup environment variables with defaults and validation."""
    required_vars = ["CB_CONN_STRING", "CB_USERNAME", "CB_PASSWORD", "CB_BUCKET"]
    for var in required_vars:
        _set_if_undefined(var)

    # Optional Capella AI variables
    optional_vars = ["CAPELLA_API_ENDPOINT", "CAPELLA_API_EMBEDDING_MODEL", "CAPELLA_API_LLM_MODEL"]
    for var in optional_vars:
        if not os.environ.get(var):
            print(f"ℹ️ {var} not provided - will use OpenAI fallback")

    # Set defaults for landmark search (non-sensitive only)
    defaults = {
        "CB_BUCKET": "travel-sample",
        "CB_INDEX": "landmark_vector_index",
        "CB_SCOPE": "inventory",
        "CB_COLLECTION": "landmark",
        "CAPELLA_API_EMBEDDING_MODEL": "intfloat/e5-mistral-7b-instruct",
        "CAPELLA_API_LLM_MODEL": "meta-llama/Llama-3.1-8B-Instruct",
    }

    for key, default_value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = default_value

    # Generate Capella AI API key if endpoint is provided
    if os.environ.get("CAPELLA_API_ENDPOINT"):
        os.environ["CAPELLA_API_KEY"] = base64.b64encode(
            f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode("utf-8")
        ).decode("utf-8")

        # Use endpoint as provided
        print(f"Using Capella AI endpoint: {os.environ['CAPELLA_API_ENDPOINT']}")


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
            from couchbase.options import ClusterTimeoutOptions

            timeout_options = ClusterTimeoutOptions(
                kv_timeout=timedelta(seconds=10),
                kv_durable_timeout=timedelta(seconds=15),
                query_timeout=timedelta(seconds=30),
                search_timeout=timedelta(seconds=30),
                management_timeout=timedelta(seconds=30),
                bootstrap_timeout=timedelta(seconds=20),
            )
            options.timeout_options = timeout_options

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
            remaining_count = count_row['count']
            
            if remaining_count == 0:
                logger.info(f"Collection cleared successfully, {remaining_count} documents remaining")
            else:
                logger.warning(f"Collection clear incomplete, {remaining_count} documents remaining")
            
        except Exception as e:
            logger.warning(f"Error clearing collection data: {e}")
            # If N1QL fails, try to continue anyway
            pass

    def get_collection(self, scope_name: str, collection_name: str):
        """Get a collection object."""
        key = f"{scope_name}.{collection_name}"
        if key not in self._collections:
            self._collections[key] = self.bucket.scope(scope_name).collection(
                collection_name
            )
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
        try:
            # Setup LLM and embeddings
            if os.environ.get("CAPELLA_API_ENDPOINT"):
                llm = OpenAILike(
                    model=os.environ["CAPELLA_API_LLM_MODEL"],
                    api_base=os.environ["CAPELLA_API_ENDPOINT"],
                    api_key=os.environ["CAPELLA_API_KEY"],
                    is_chat_model=True,
                    temperature=0.1,
                )
                embeddings = OpenAIEmbedding(
                    model=os.environ["CAPELLA_API_EMBEDDING_MODEL"],
                    api_base=os.environ["CAPELLA_API_ENDPOINT"],
                    api_key=os.environ["CAPELLA_API_KEY"],
                    dimensions=1024,
                )
            else:
                llm = OpenAILike(
                    model="gpt-4o",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    is_chat_model=True,
                    temperature=0.1,
                )
                embeddings = OpenAIEmbedding(
                    model="text-embedding-3-small",
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )

            # Set global settings
            Settings.llm = llm
            Settings.embed_model = embeddings

            # Setup collection
            self.setup_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])

            # Load landmark data
            self.load_landmark_data(
                os.environ["CB_SCOPE"],
                os.environ["CB_COLLECTION"],
                os.environ["CB_INDEX"],
                embeddings,
            )

            # Create vector store
            vector_store = CouchbaseSearchVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=os.environ["CB_SCOPE"],
                collection_name=os.environ["CB_COLLECTION"],
                index_name=os.environ["CB_INDEX"],
            )

            return vector_store

        except Exception as e:
            raise ValueError(f"Error setting up vector store: {e!s}")

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
            # Enhance prompt to be more decisive
            system_prompt += "\n\nIMPORTANT: Once you have gathered sufficient information to answer the user's question, provide a complete final answer immediately. Do not continue searching if you already have the key information requested."

            # Create ReAct agent with limits to prevent excessive iterations
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=Settings.llm,
                verbose=True,
                system_prompt=system_prompt,
                max_iterations=10,  # Keep reasonable max
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

    span.end()
    return agent, client


def demo_queries(agent):
    """Run a few demo queries to show the agent's capabilities."""
    print("\n🚀 DEMO: Running sample landmark search queries...")
    print("=" * 50)

    # Get a few sample queries
    sample_queries = get_queries_for_evaluation(limit=3)

    for i, query in enumerate(sample_queries, 1):
        print(f"\n🏛️ Query {i}: {query}")
        print("-" * 30)

        try:
            response = agent.chat(query, chat_history=[])
            print(f"Response: {response.response}")

            # Show reference answer for comparison
            reference = get_reference_answer(query)
            if reference != "No reference answer available for this query.":
                print(f"\n📚 Reference: {reference[:200]}...")

        except Exception as e:
            print(f"❌ Error: {e}")

        print()


def run_interactive_demo():
    """Run an interactive landmark search demo."""
    logger.info("Landmark Search Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        agent, client = setup_landmark_agent()

        # Interactive landmark search loop
        logger.info("Available commands:")
        logger.info(
            "- Enter landmark search queries (e.g., 'Find landmarks in Paris')"
        )
        logger.info("- 'quit' - Exit the demo")
        logger.info(
            "Try asking: 'Find me landmarks in Tokyo' or 'Show me museums in London'"
        )
        logger.info("─" * 40)

        while True:
            query = input(
                "🔍 Enter landmark search query (or 'quit' to exit): "
            ).strip()

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
