#!/usr/bin/env python3
"""
Route Planner Agent - Agent Catalog + LlamaIndex Implementation

A streamlined route planner agent demonstrating Agent Catalog integration
with LlamaIndex and Couchbase vector search for route planning assistance.
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

    # Set defaults
    defaults = {
        "CB_CONN_STRING": "couchbases://cb.hlcup4o4jmjr55yf.cloud.couchbase.com",
        "CB_USERNAME": "kaustavcluster",
        "CB_PASSWORD": "Password@123",
        "CB_BUCKET": "vector-search-testing",
        "CB_INDEX": "route_data_index",
        "CB_SCOPE": "agentc_data",
        "CB_COLLECTION": "route_data",
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

            bucket_manager = self.bucket.collections()

            # Handle scope creation
            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                logger.info(f"Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                logger.info(f"Scope '{scope_name}' created successfully")

            # Handle collection creation
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name
                and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if not collection_exists:
                logger.info(f"Creating collection '{collection_name}'...")
                bucket_manager.create_collection(scope_name, collection_name)
                logger.info(f"Collection '{collection_name}' created successfully")

            collection = self.bucket.scope(scope_name).collection(collection_name)
            time.sleep(3)

            # Create primary index
            try:
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
                ).execute()
                logger.info("Primary index created successfully")
            except Exception as e:
                logger.warning(f"Error creating primary index: {e!s}")

            # Cache the collection for reuse
            collection_key = f"{scope_name}.{collection_name}"
            self._collections[collection_key] = collection

            logger.info(f"Collection setup complete for {scope_name}.{collection_name}")
            return collection

        except Exception as e:
            raise RuntimeError(f"Error setting up collection: {e!s}")

    def get_collection(self, scope_name: str, collection_name: str):
        """Get a collection, creating it if it doesn't exist."""
        collection_key = f"{scope_name}.{collection_name}"
        if collection_key not in self._collections:
            self.setup_collection(scope_name, collection_name)
        return self._collections[collection_key]

    def setup_vector_search_index(self, index_definition: dict):
        """Setup vector search index."""
        try:
            if not self.bucket:
                raise RuntimeError("Bucket not initialized. Call setup_collection first.")

            scope_name = os.environ["CB_SCOPE"]
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

    def setup_ai_models(self, span):
        """Setup AI models for embeddings and LLM."""
        try:
            # Setup embedding model
            if os.environ.get("CAPELLA_API_ENDPOINT"):
                embed_model = OpenAIEmbedding(
                    api_key=os.environ["CAPELLA_API_KEY"],
                    api_base=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
                    model_name=os.environ["CAPELLA_API_EMBEDDING_MODEL"],
                    embed_batch_size=30,
                )
            else:
                embed_model = OpenAIEmbedding()

            # Setup LLM
            if os.environ.get("CAPELLA_API_ENDPOINT"):
                llm = OpenAILike(
                    api_base=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
                    api_key=os.environ["CAPELLA_API_KEY"],
                    model=os.environ["CAPELLA_API_LLM_MODEL"],
                    temperature=0.1,
                )
            else:
                from llama_index.llms.openai import OpenAI

                llm = OpenAI(temperature=0.1)

            # Configure global settings
            Settings.embed_model = embed_model
            Settings.llm = llm

            logger.info("AI models configured successfully")
            return embed_model, llm

        except Exception as e:
            raise RuntimeError(f"Error setting up AI models: {e!s}")

    def load_route_data(self, vector_store, span, embeddings):
        """Load route data using dedicated data loading script."""
        try:
            from data.route_data import load_route_data_to_couchbase

            logger.info("Loading route data using data loading script...")

            # Load data using the dedicated script
            load_route_data_to_couchbase(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=os.environ["CB_SCOPE"],
                collection_name=os.environ["CB_COLLECTION"],
                embeddings=embeddings,
                index_name=os.environ["CB_INDEX"],
            )

            logger.info("Route data loaded successfully")

        except Exception as e:
            logger.error(f"Error loading route data: {e}")
            raise ValueError(f"Error loading route data: {e!s}")

    def setup_vector_store(self, span, embeddings):
        """Setup LlamaIndex vector store with route data."""
        try:
            if not self.cluster:
                raise RuntimeError("Cluster not connected. Call connect first.")

            # Create vector store
            vector_store = CouchbaseSearchVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=os.environ["CB_SCOPE"],
                collection_name=os.environ["CB_COLLECTION"],
                index_name=os.environ["CB_INDEX"],
            )

            # Load route data
            try:
                self.load_route_data(vector_store, span, embeddings)
                logger.info("Route data loaded into vector store successfully")
            except Exception as e:
                logger.error(f"Failed to load route data: {e}")
                logger.warning("Vector store created but data not loaded.")

            return vector_store

        except Exception as e:
            raise ValueError(f"Error setting up vector store: {e!s}")

    def create_llamaindex_agent(self, catalog, span):
        """Create LlamaIndex ReAct agent with tools from Agent Catalog."""
        try:
            # Get tools from Agent Catalog
            tools = []

            # Search routes tool
            search_tool_result = catalog.find("tool", name="search_routes")
            if search_tool_result:
                tools.append(
                    FunctionTool.from_defaults(
                        fn=search_tool_result.func,
                        name="search_routes",
                        description=getattr(search_tool_result.meta, "description", None)
                        or "Search for route information using semantic vector search. Use for finding travel routes, scenic drives, and transportation information.",
                    )
                )
                logger.info("Loaded search_routes tool from AgentC")

            # Calculate distance tool
            distance_tool_result = catalog.find("tool", name="calculate_distance")
            if distance_tool_result:
                tools.append(
                    FunctionTool.from_defaults(
                        fn=distance_tool_result.func,
                        name="calculate_distance",
                        description=getattr(distance_tool_result.meta, "description", None)
                        or "Calculate distance, travel time, and cost between two cities using different transportation modes (car, train, bus, flight).",
                    )
                )
                logger.info("Loaded calculate_distance tool from AgentC")

            if not tools:
                logger.warning("No tools found in Agent Catalog")
            else:
                logger.info(f"Loaded {len(tools)} tools from Agent Catalog")

            # Get prompt from Agent Catalog - REQUIRED, no fallbacks
            prompt_result = catalog.find("prompt", name="route_planner_assistant")
            if not prompt_result:
                raise RuntimeError("Prompt 'route_planner_assistant' not found in Agent Catalog")

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


def setup_route_agent():
    """Setup the complete route planning agent infrastructure and return the agent."""
    setup_environment()

    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    span = catalog.Span(name="Route Planner Agent Setup")

    # Initialize Couchbase client
    client = CouchbaseClient(
        conn_string=os.environ["CB_CONN_STRING"],
        username=os.environ["CB_USERNAME"],
        password=os.environ["CB_PASSWORD"],
        bucket_name=os.environ["CB_BUCKET"],
    )

    # Setup infrastructure
    client.connect()
    client.setup_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])

    # Setup vector search index
    try:
        with open("agentcatalog_index.json", "r") as f:
            index_definition = json.load(f)
        client.setup_vector_search_index(index_definition)
    except Exception as e:
        logger.warning(f"Could not setup vector search index: {e}")

    # Setup AI models
    embed_model, llm = client.setup_ai_models(span)

    # Setup vector store
    vector_store = client.setup_vector_store(span, embed_model)

    # Create agent
    agent = client.create_llamaindex_agent(catalog, span)

    logger.info("Route Planner Agent initialized successfully")
    return agent, catalog


def run_interactive_demo():
    """Run an interactive demo of the route planner agent."""
    print("🚀 Starting Route Planner Agent Interactive Demo...")
    agent, catalog = setup_route_agent()

    print("\n🗺️ Route Planner Agent Ready!")
    print("Ask me about routes, distances, or travel planning. Type 'quit' to exit.")
    print("\nExample queries:")
    print("- 'Find scenic routes from San Francisco to Los Angeles'")
    print("- 'Calculate distance from New York to Chicago by car'")
    print("- 'What are good road trip routes in Colorado?'")

    while True:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        try:
            span = catalog.Span(name="Route Query")
            with span:
                logger.info(f"Processing query: {user_input}")
                response = agent.chat(user_input)
                print(f"\n🤖 Agent: {response}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"❌ Error: {e}")


def run_test():
    """Run predefined test queries to demonstrate route planner agent capabilities."""
    print("🧪 Starting Route Planner Agent Test Suite...")
    agent, catalog = setup_route_agent()

    # Test queries with specific, realistic scenarios
    test_queries = [
        "Find scenic driving routes from Denver to Aspen in Colorado",
        "Calculate the distance from New York to San Francisco by different transportation modes like car, train, and flight",
        "Plan a road trip from Salt Lake City to Zion National Park and Bryce Canyon",
    ]

    print(f"\n🔍 Running {len(test_queries)} test queries...\n")

    for i, query in enumerate(test_queries, 1):
        print(f"📋 Test {i}/{len(test_queries)}: {query}")
        print("-" * 60)

        try:
            span = catalog.Span(name=f"Test Query {i}")
            with span:
                start_time = time.time()
                response = agent.chat(query)
                end_time = time.time()

                print(f"✅ Response ({end_time - start_time:.1f}s):")
                print(response)
                print("\n" + "=" * 60 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n" + "=" * 60 + "\n")

    print("🎉 Test suite completed!")


def main():
    """Main function with command line argument support."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "demo":
            run_interactive_demo()
        elif mode == "test":
            run_test()
        elif mode == "interactive":
            run_interactive_demo()
        else:
            print("Usage: python main.py [demo|test|interactive]")
            print("  demo/interactive: Run interactive demo")
            print("  test: Run predefined test queries")
            sys.exit(1)
    else:
        # Default behavior - interactive demo
        run_interactive_demo()


if __name__ == "__main__":
    main()
