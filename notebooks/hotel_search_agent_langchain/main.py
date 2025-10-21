#!/usr/bin/env python3
"""
Hotel Support Agent - Agent Catalog + LangChain Implementation

A streamlined hotel support agent demonstrating Agent Catalog integration
with LangChain and Couchbase vector search for hotel booking assistance.
Uses real hotel data from travel-sample.inventory.hotel collection.
"""

import json
import logging
import os
import sys

import agentc
import agentc_langchain
import dotenv

# Import hotel data from the data module
from data.hotel_data import load_hotel_data_to_couchbase
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool


# Import shared modules using robust project root discovery
def find_project_root():
    """Find the project root by looking for the shared directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Look for the shared directory as the definitive marker
        shared_path = os.path.join(current, "shared")
        if os.path.exists(shared_path) and os.path.isdir(shared_path):
            return current
        current = os.path.dirname(current)
    return None


# Add project root to Python path
project_root = find_project_root()
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.agent_setup import (
    setup_ai_services,
    setup_environment,
    test_capella_connectivity,
)
from shared.couchbase_client import create_couchbase_client

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

# Priority 1 setup uses direct API keys from environment only
if os.getenv("CAPELLA_API_EMBEDDINGS_KEY") or os.getenv("CAPELLA_API_LLM_KEY"):
    logger.info("Using direct Capella API keys from environment")

# Set default values for travel-sample bucket configuration
DEFAULT_BUCKET = "travel-sample"
DEFAULT_SCOPE = "agentc_data"
DEFAULT_COLLECTION = "hotel_data"
DEFAULT_INDEX = "hotel_data_index"





# Simplified setup functions using shared Priority 1 AI services
def setup_embeddings_service(input_type="query"):
    """Setup embeddings service using Priority 1 (OpenAI wrappers + Capella)."""
    embeddings, _ = setup_ai_services(framework="langchain")
    return embeddings


def setup_llm_service(application_span=None):
    """Setup LLM service using Priority 1 (OpenAI wrappers + Capella)."""
    callbacks = (
        [agentc_langchain.chat.Callback(span=application_span)]
        if application_span
        else None
    )
    _, llm = setup_ai_services(framework="langchain", callbacks=callbacks)
    return llm


def setup_hotel_support_agent():
    """Setup the hotel support agent with Agent Catalog integration."""
    try:
        # Initialize Agent Catalog with single application span
        catalog = agentc.catalog.Catalog()
        application_span = catalog.Span(name="Hotel Support Agent", blacklist=set())

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

        # Setup Couchbase connection and collections using shared client
        couchbase_client = create_couchbase_client()
        couchbase_client.connect()
        couchbase_client.setup_collection(
            os.getenv("CB_SCOPE", DEFAULT_SCOPE),
            os.getenv("CB_COLLECTION", DEFAULT_COLLECTION),
            clear_existing_data=False,  # Keep existing data, let data loader handle it
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

        # Try to setup vector search index, but continue if it fails
        try:
            couchbase_client.setup_vector_search_index(
                index_definition, os.getenv("CB_SCOPE", DEFAULT_SCOPE)
            )
            logger.info("✅ Vector search index setup completed")
        except Exception as e:
            logger.warning(f"⚠️ Vector search index setup failed: {e}")
            logger.info(
                "🔄 Continuing without vector search index - basic functionality will still work"
            )

        # Setup embeddings with priority order
        embeddings = setup_embeddings_service(input_type="passage")

        # Setup vector store with hotel data loading
        couchbase_client.setup_vector_store_langchain(
            scope_name=os.getenv("CB_SCOPE", DEFAULT_SCOPE),
            collection_name=os.getenv("CB_COLLECTION", DEFAULT_COLLECTION),
            index_name=os.getenv("CB_INDEX", DEFAULT_INDEX),
            embeddings=embeddings,
            data_loader_func=load_hotel_data_to_couchbase,
        )

        # Setup LLM with priority order
        llm = setup_llm_service(application_span)

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
            return """I need to use the correct format. Let me try again with the proper ReAct format.

Thought: I need to search for hotels matching the user's requirements
Action: search_vector_database
Action Input: hotels matching user search criteria"""

        agent = create_react_agent(llm, tools, custom_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=handle_parsing_error,  # Use custom error handler
            max_iterations=2,  # STRICT: 1 tool call + 1 Final Answer only
            early_stopping_method="force",  # Force stop
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
