#!/usr/bin/env python3
"""
Landmark Search Agent - Agent Catalog + LlamaIndex Implementation

A streamlined landmark search agent demonstrating Agent Catalog integration
with LlamaIndex and Couchbase vector search for landmark discovery assistance.
"""

import base64
import getpass
import logging
import os
import sys

import dotenv
from llama_index.core import Settings

# Import shared modules using robust project root discovery
def find_project_root():
    """Find the project root by looking for the shared directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Look for the shared directory as the definitive marker
        shared_path = os.path.join(current, 'shared')
        if os.path.exists(shared_path) and os.path.isdir(shared_path):
            return current
        current = os.path.dirname(current)
    return None

# Add project root to Python path
project_root = find_project_root()
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import agentc and other modules after path is set
import agentc
from shared.agent_setup import setup_ai_services, setup_environment, test_capella_connectivity
from shared.couchbase_client import create_couchbase_client

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

        logger.info(
            f"Using Capella AI endpoint for embeddings: {os.environ['CAPELLA_API_ENDPOINT']}"
        )
        logger.info(
            f"Using NVIDIA NIMs for LLM with API key: {os.environ['NVIDIA_API_KEY'][:10]}..."
        )

    # Validate configuration consistency
    logger.info(f"✅ Configuration loaded:")
    logger.info(f"   Bucket: {os.environ['CB_BUCKET']}")
    logger.info(f"   Scope: {os.environ['CB_SCOPE']}")
    logger.info(f"   Collection: {os.environ['CB_COLLECTION']}")
    logger.info(f"   Index: {os.environ['CB_INDEX']}")


def create_llamaindex_agent(catalog, span):
    """Create LlamaIndex ReAct agent with landmark search tool from Agent Catalog."""
    try:
        from llama_index.core.agent import ReActAgent
        from llama_index.core.tools import FunctionTool

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

        # Create ReAct agent with reasonable iteration limit
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=Settings.llm,
            verbose=True,  # Keep on for debugging
            system_prompt=system_prompt,
            max_iterations=4,  # Allow one tool call and finalization without warnings
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
    span = catalog.Span(name="Landmark Search Agent Setup", blacklist=set())

    # Setup database client using shared module
    client = create_couchbase_client()
    client.connect()

    # Setup LLM and embeddings using shared module
    embeddings, llm = setup_ai_services(framework="llamaindex", temperature=0.1, application_span=span)

    # Set global LlamaIndex settings
    Settings.llm = llm
    Settings.embed_model = embeddings

    # Setup collection
    client.setup_collection(
        os.environ["CB_SCOPE"], 
        os.environ["CB_COLLECTION"],
        clear_existing_data=False  # Let data loader decide based on count check
    )

    # Setup vector search index
    index_definition = client.load_index_definition()
    if index_definition:
        client.setup_vector_search_index(index_definition, os.environ["CB_SCOPE"])

    # Setup vector store with landmark data
    vector_store = client.setup_vector_store_llamaindex(
        scope_name=os.environ["CB_SCOPE"],
        collection_name=os.environ["CB_COLLECTION"],
        index_name=os.environ["CB_INDEX"],
    )

    # Load landmark data
    load_landmark_data_to_couchbase(
        cluster=client.cluster,
        bucket_name=client.bucket_name,
        scope_name=os.environ["CB_SCOPE"],
        collection_name=os.environ["CB_COLLECTION"],
        embeddings=embeddings,
        index_name=os.environ["CB_INDEX"],
    )

    # Create LlamaIndex ReAct agent
    agent = create_llamaindex_agent(catalog, span)

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

                logger.info(f"\n🏛️ Agent Response:\n{result}\n")
                logger.info("─" * 40)

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                logger.error(f"❌ Error: {e}")
                logger.info("─" * 40)

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
