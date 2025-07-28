#!/usr/bin/env python3
"""
Hotel Support Agent - Agent Catalog + LangChain Implementation

A streamlined hotel support agent demonstrating Agent Catalog integration
with LangChain and Couchbase vector search for hotel booking assistance.
Uses real hotel data from travel-sample.inventory.hotel collection.
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
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Import hotel data from the data module
from data.hotel_data import get_hotel_texts, load_hotel_data_to_couchbase

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

# Generate Capella AI API key if endpoint is provided
if (
    os.getenv("CAPELLA_API_ENDPOINT")
    and os.getenv("CB_USERNAME")
    and os.getenv("CB_PASSWORD")
):
    os.environ["CAPELLA_API_KEY"] = base64.b64encode(
        f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode("utf-8")
    ).decode("utf-8")

# Set default values for travel-sample bucket configuration
DEFAULT_BUCKET = "travel-sample"
DEFAULT_SCOPE = "agentc_data"
DEFAULT_COLLECTION = "hotel_data"
DEFAULT_INDEX = "hotel_data_index"
DEFAULT_NVIDIA_API_LLM_MODEL = "meta/llama-4-maverick-17b-128e-instruct"


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
        raise ValueError(
            f"Missing required Capella AI environment variables: {missing_vars}"
        )

    return {
        "endpoint": os.getenv("CAPELLA_API_ENDPOINT"),
        "embedding_model": os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
        "llm_model": os.getenv("CAPELLA_API_LLM_MODEL"),
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

            headers = {
                "Authorization": f"Basic {api_key}",
                "Content-Type": "application/json",
            }

            # Test embedding
            logger.info("Testing Capella AI connectivity...")
            embedding_data = {
                "model": os.getenv(
                    "CAPELLA_API_EMBEDDING_MODEL", "intfloat/e5-mistral-7b-instruct"
                ),
                "input": "test connectivity",
            }

            response = requests.post(
                f"{endpoint}/embeddings", json=embedding_data, headers=headers
            )
            if response.status_code == 200:
                logger.info("✅ Capella AI embedding test successful")
                return True
            else:
                logger.warning(f"❌ Capella AI embedding test failed: {response.text}")
                return False
        else:
            logger.warning("Capella AI credentials not available")
            return False
    except Exception as e:
        logger.warning(f"❌ Capella AI connectivity test failed: {e}")
        return False


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

    # Required Couchbase connection variables
    _set_if_undefined("CB_CONN_STRING")
    _set_if_undefined("CB_USERNAME")
    _set_if_undefined("CB_PASSWORD")

    # Optional Capella AI configuration
    if os.getenv("CAPELLA_API_ENDPOINT"):
        # Ensure endpoint has /v1 suffix for OpenAI compatibility
        if not os.getenv("CAPELLA_API_ENDPOINT").endswith("/v1"):
            os.environ["CAPELLA_API_ENDPOINT"] = (
                os.getenv("CAPELLA_API_ENDPOINT").rstrip("/") + "/v1"
            )
            logger.info(
                f"Added /v1 suffix to endpoint: {os.getenv('CAPELLA_API_ENDPOINT')}"
            )

    # Test Capella AI connectivity
    test_capella_connectivity()


class CouchbaseClient:
    """Centralized Couchbase client for all database operations."""

    def __init__(
        self, conn_string: str, username: str, password: str, bucket_name: str
    ):
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
            self.cluster.wait_until_ready(timedelta(seconds=10))
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
            logger.info(
                f"Clearing data from {self.bucket_name}.{scope_name}.{collection_name}..."
            )

            # Use N1QL to delete all documents with explicit execution
            delete_query = (
                f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
            )
            result = self.cluster.query(delete_query)

            # Execute the query and get the results
            rows = list(result)

            # Wait a moment for the deletion to propagate
            import time

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

    def load_hotel_data(self, scope_name, collection_name, index_name, embeddings):
        """Load hotel data into Couchbase."""
        try:
            # Load hotel data using the data loading script
            load_hotel_data_to_couchbase(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embeddings=embeddings,
                index_name=index_name,
            )
            logger.info("Hotel data loaded into vector store successfully")

        except Exception as e:
            raise RuntimeError(f"Error loading hotel data: {e!s}")

    def setup_vector_store(
        self, scope_name: str, collection_name: str, index_name: str, embeddings
    ):
        """Setup vector store with hotel data."""
        try:
            # Load hotel data
            self.load_hotel_data(scope_name, collection_name, index_name, embeddings)

            # Create vector store instance
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


def setup_hotel_support_agent():
    """Setup the hotel support agent with Agent Catalog integration."""
    try:
        # Initialize Agent Catalog with single application span
        catalog = agentc.catalog.Catalog()
        application_span = catalog.Span(name="Hotel Support Agent")

        # Setup environment
        setup_environment()

        # Test Capella AI connectivity if configured
        if os.getenv("CAPELLA_API_ENDPOINT"):
            if not test_capella_connectivity():
                logger.warning(
                    "❌ Capella AI connectivity test failed. Will use OpenAI fallback."
                )
        else:
            logger.info("ℹ️ Capella API not configured - will use OpenAI models")

        # Setup Couchbase connection and collections
        couchbase_client = CouchbaseClient(
            conn_string=os.getenv("CB_CONN_STRING"),
            username=os.getenv("CB_USERNAME"),
            password=os.getenv("CB_PASSWORD"),
            bucket_name=os.getenv("CB_BUCKET", DEFAULT_BUCKET),
        )
        couchbase_client.connect()
        couchbase_client.setup_collection(
            os.getenv("CB_SCOPE", DEFAULT_SCOPE),
            os.getenv("CB_COLLECTION", DEFAULT_COLLECTION),
        )

        # Setup vector index
        try:
            with open("agentcatalog_index.json", "r") as file:
                index_definition = json.load(file)
            logger.info(
                "Loaded vector search index definition from agentcatalog_index.json"
            )
        except Exception as e:
            raise ValueError(f"Error loading index definition: {e!s}")

        couchbase_client.setup_vector_search_index(
            index_definition, os.getenv("CB_SCOPE", DEFAULT_SCOPE)
        )

        # Setup embeddings
        try:
            # Capella AI embeddings
            # if os.getenv("CAPELLA_API_ENDPOINT") and os.getenv("CAPELLA_API_KEY"):
            #     embeddings = OpenAIEmbeddings(
            #         model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
            #         api_key=os.getenv("CAPELLA_API_KEY"),
            #         base_url=os.getenv("CAPELLA_API_ENDPOINT"),
            #     )

            # NVIDIA embeddings
            # _set_if_undefined("NVIDIA_API_KEY")
            embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embedqa-e5-v5",
                api_key=os.getenv("NVIDIA_API_KEY"),
                truncate="END",
            )
            # logger.info("✅ Using nvidia/nv-embedqa-e5-v5 for embeddings")

            # OpenAI embeddings
            # embeddings = OpenAIEmbeddings(
            #     model="text-embedding-3-small",
            #     api_key=os.getenv("OPENAI_API_KEY"),
            #     base_url=os.getenv("OPENAI_API_ENDPOINT"),
            # )

        except Exception as e:
            logger.error(f"❌ Embeddings setup failed: {e}")
            raise RuntimeError("Embeddings configuration required")

        couchbase_client.setup_vector_store(
            os.getenv("CB_SCOPE", DEFAULT_SCOPE),
            os.getenv("CB_COLLECTION", DEFAULT_COLLECTION),
            os.getenv("CB_INDEX", DEFAULT_INDEX),
            embeddings,
        )

        # Setup LLM with deterministic settings
        llm = None

        # Try Capella AI LLM first
        try:
            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                model=os.getenv("CAPELLA_API_LLM_MODEL"),
                temperature=0.0,
                callbacks=[agentc_langchain.chat.Callback(span=application_span)],
            )
            llm.invoke("Hello")  # Test the LLM works
            logger.info("✅ Using Capella AI LLM")
        except Exception as e:
            logger.warning(f"⚠️ Capella AI LLM failed: {e}")

        # COMMENTED OUT - NIM LLM (only for prototype)
        # try:
        #     _set_if_undefined("NVIDIA_API_KEY")
        #     llm = ChatOpenAI(
        #         api_key=os.getenv("NVIDIA_API_KEY"),
        #         base_url=os.getenv("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        #         model=os.getenv("NVIDIA_API_LLM_MODEL", DEFAULT_NVIDIA_API_LLM_MODEL),
        #         temperature=0,
        #         callbacks=[agentc_langchain.chat.Callback(span=application_span)],
        #     )
        #     logger.info("✅ Using NIM LLM")
        # except Exception as e:
        #     logger.warning(f"⚠️ NIM LLM failed: {e}")

        # Fallback to OpenAI if no other LLM worked
        # if not llm:
        #     logger.info("🔄 Falling back to OpenAI LLM...")
        #     _set_if_undefined("OPENAI_API_KEY")
        #     llm = ChatOpenAI(
        #         api_key=os.getenv("OPENAI_API_KEY"),
        #         model="gpt-4o",
        #         temperature=0,
        #         callbacks=[agentc_langchain.chat.Callback(span=application_span)],
        #     )
        #     logger.info("✅ Using OpenAI LLM as fallback")

        # Load tools and create agent
        tool_search = catalog.find("tool", name="search_vector_database")
        if not tool_search:
            raise ValueError(
                "Could not find search_vector_database tool. Make sure it's indexed with 'agentc index tools/'"
            )

        tools = [
            Tool(
                name=tool_search.meta.name,
                description=tool_search.meta.description,
                func=tool_search.func,
            ),
        ]

        hotel_prompt = catalog.find("prompt", name="hotel_search_assistant")
        if not hotel_prompt:
            raise ValueError(
                "Could not find hotel_search_assistant prompt in catalog. Make sure it's indexed with 'agentc index prompts/'"
            )

        custom_prompt = PromptTemplate(
            template=hotel_prompt.content.strip(),
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join(
                    [f"{tool.name}: {tool.description}" for tool in tools]
                ),
                "tool_names": ", ".join([tool.name for tool in tools]),
            },
        )

        def handle_parsing_error(error) -> str:
            """Custom error handler for parsing errors that guides agent back to ReAct format."""
            logger.warning(f"Parsing error occurred: {error}")
            return """I need to use the correct format. Let me start over:

Thought: I need to search for hotels using the search_vector_database tool
Action: search_vector_database
Action Input: """

        agent = create_react_agent(llm, tools, custom_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=handle_parsing_error,  # Use custom error handler
            max_iterations=8,  # Increased from 5
            max_execution_time=120,  # Increased from 60
            early_stopping_method="force",  # Changed from "generate" to "force"
            return_intermediate_steps=True,  # For better debugging
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
        logger.info("Available commands:")
        logger.info(
            "- Enter hotel search queries (e.g., 'Find luxury hotels with spa')"
        )
        logger.info("- 'quit' - Exit the demo")
        logger.info(
            "Try asking: 'Find me a hotel in San Francisco' or 'Show me hotels in New York'"
        )
        logger.info("─" * 40)

        while True:
            query = input("🔍 Enter hotel search query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using Hotel Support Agent!")
                break

            if not query:
                logger.warning("Please enter a search query")
                continue

            try:
                response = agent_executor.invoke({"input": query})
                result = response.get("output", "No response generated")

                print(f"\n🏨 Agent Response:\n{result}\n")
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
    """Run comprehensive test of hotel support agent with 3 test queries."""
    logger.info("Hotel Support Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        agent_executor, application_span = setup_hotel_support_agent()

        # Import shared queries
        from data.queries import get_simple_queries

        # Test scenarios covering different types of hotel searches
        test_queries = get_simple_queries()

        logger.info(f"Running {len(test_queries)} test queries...")

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}: {query}")
            try:
                response = agent_executor.invoke({"input": query})
                result = response.get("output", "No response generated")

                # Display the response
                logger.info(f"🤖 AI Response: {result}")
                logger.info(f"✅ Test {i} completed successfully")

            except Exception as e:
                logger.exception(f"❌ Test {i} failed: {e}")

            logger.info("-" * 50)

        logger.info("All tests completed!")

    except Exception as e:
        logger.exception(f"Test error: {e}")


def test_data_loading():
    """Test data loading from travel-sample independently."""
    logger.info("Testing Hotel Data Loading from travel-sample")
    logger.info("=" * 50)

    try:
        from data.hotel_data import get_hotel_count, get_hotel_texts

        # Test hotel count
        count = get_hotel_count()
        logger.info(f"✅ Hotel count in travel-sample.inventory.hotel: {count}")

        # Test hotel text generation
        texts = get_hotel_texts()
        logger.info(f"✅ Generated {len(texts)} hotel texts for embeddings")

        if texts:
            logger.info(f"✅ First hotel text sample: {texts[0][:200]}...")

        logger.info("✅ Data loading test completed successfully")

    except Exception as e:
        logger.exception(f"❌ Data loading test failed: {e}")


def main():
    """Main entry point - runs interactive demo by default."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        elif sys.argv[1] == "test-data":
            test_data_loading()
        else:
            run_interactive_demo()
    else:
        run_interactive_demo()


if __name__ == "__main__":
    main()
