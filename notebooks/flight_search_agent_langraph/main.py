#!/usr/bin/env python3
"""
Flight Search Agent - Agent Catalog + LangGraph Implementation

A streamlined flight search agent demonstrating Agent Catalog integration
with LangGraph and Couchbase vector search for flight booking assistance.
"""

import json
import logging
import os
import sys
from datetime import timedelta

import agentc
import agentc_langgraph.agent
import agentc_langgraph.graph
import dotenv
import langchain_core.messages
import langchain_core.runnables
import langchain_openai.chat_models
import langgraph.graph
import langgraph.prebuilt
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import KeyspaceNotFoundException
from couchbase.options import ClusterOptions
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pydantic import SecretStr


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

from shared.agent_setup import setup_ai_services, setup_environment, test_capella_connectivity
from shared.couchbase_client import create_couchbase_client

# Setup logging with essential level only
# Note: Agent Catalog does not integrate with Python logging
# The main application span will generate meaningful logs
# These logs can be queried from the Agent Catalog bucket (see query_agent_catalog_logs)
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



class CapellaCompatibleChatModel(langchain_core.runnables.Runnable):
    """Wrapper for chat models that disables function calling for Capella AI compatibility."""

    def __init__(self, chat_model):
        super().__init__()
        self.chat_model = chat_model

    def bind_tools(self, *args, **kwargs):
        """Disabled bind_tools to force traditional ReAct format."""
        return self

    def invoke(self, input, config=None, **kwargs):
        """Delegate invoke to the original model."""
        return self.chat_model.invoke(input, config, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generate to the original model."""
        return self.chat_model.generate(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate any missing attributes to the original model."""
        return getattr(self.chat_model, name)


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations - single user system."""

    query: str
    resolved: bool
    search_results: list[dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and ReActAgent framework."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span, chat_model=None):
        """Initialize the flight search agent."""

        if chat_model is None:
            # Fallback to OpenAI if no chat model provided
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

        # Wrap the chat model to disable function calling for Capella AI compatibility
        chat_model = CapellaCompatibleChatModel(chat_model)
        logger.info("Wrapped chat model to disable function calling for Capella AI compatibility")

        super().__init__(
            chat_model=chat_model, catalog=catalog, span=span, prompt_name="flight_search_assistant"
        )

    def create_react_agent(self, span: agentc.Span, **kwargs):
        """Simplified approach: Use Agent Catalog's built-in methods directly."""

        # Use Agent Catalog's direct tool discovery methods
        tools = []
        tool_names = ["lookup_flight_info", "save_flight_booking", "retrieve_flight_bookings", "search_airline_reviews"]

        for tool_name in tool_names:
            try:
                # Direct Agent Catalog tool lookup as suggested by user
                catalog_tool = self.catalog.find("tool", name=tool_name)
                if catalog_tool:
                    langchain_tool = Tool(
                        name=tool_name,
                        description=f"Tool: {tool_name}",
                        func=catalog_tool.func
                    )
                    tools.append(langchain_tool)
                    logger.info(f"✅ Added tool from Agent Catalog: {tool_name}")
            except Exception as e:
                logger.error(f"❌ Failed to add tool {tool_name}: {e}")

        # Fallback: Use direct catalog search if no tools found
        if not tools:
            try:
                all_tools = self.catalog.find_tools()
                for catalog_tool in all_tools:
                    if hasattr(catalog_tool, 'meta') and hasattr(catalog_tool.meta, 'name'):
                        tool_name = catalog_tool.meta.name
                        if tool_name in tool_names:
                            langchain_tool = Tool(
                                name=tool_name,
                                description=f"Tool: {tool_name}",
                                func=catalog_tool.func
                            )
                            tools.append(langchain_tool)
                            logger.info(f"✅ Added tool from fallback search: {tool_name}")
            except Exception as e:
                logger.error(f"❌ Fallback tool discovery failed: {e}")

        # Use existing Agent Catalog prompt structure directly
        if self.prompt_content:
            import datetime
            current_date = datetime.date.today().strftime("%Y-%m-%d")

            # Agent Catalog prompt already has ReAct format and all placeholders
            react_template = str(self.prompt_content.content)
            react_template = react_template.replace("{current_date}", current_date)

            # Create PromptTemplate using existing Agent Catalog structure
            react_prompt = PromptTemplate(
                template=react_template,
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"]
            )
        else:
            raise ValueError("Agent Catalog prompt not loaded")

        # Create traditional ReAct agent for beautiful verbose output
        agent = create_react_agent(self.chat_model, tools, react_prompt)

        # Return AgentExecutor with verbose=True for beautiful real-time output
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # This gives us the beautiful Action/Observation format!
            handle_parsing_errors=True,
            max_iterations=3,
            return_intermediate_steps=True
        )


    def _invoke(
        self,
        span: agentc.Span,
        state: FlightSearchState,
        config: langchain_core.runnables.RunnableConfig,
    ) -> FlightSearchState:
        """Handle flight search conversation with comprehensive debug logging."""

        logger.info("=" * 60)
        logger.info("🔍 STARTING FLIGHT SEARCH AGENT EXECUTION")
        logger.info("=" * 60)

        # Initialize conversation if this is the first message
        if not state["messages"]:
            initial_msg = langchain_core.messages.HumanMessage(content=state["query"])
            state["messages"].append(initial_msg)
            logger.info(f"📝 Flight Query: {state['query']}")
            logger.info(f"📨 Initial messages count: {len(state['messages'])}")

        # Debug: Log state before agent execution
        logger.info(f"🏷️  Current state keys: {list(state.keys())}")
        logger.info(f"📊 Messages in state: {len(state.get('messages', []))}")

        # Create hybrid AgentExecutor with Agent Catalog tools and beautiful verbose output
        logger.info("🔧 Creating hybrid AgentExecutor...")
        agent_executor = self.create_react_agent(span)
        logger.info(f"🤖 AgentExecutor created: {type(agent_executor).__name__}")

        # Execute with beautiful real-time Action/Observation/Thought display
        logger.info("⚡ Invoking AgentExecutor with verbose output...")
        try:
            response = agent_executor.invoke({"input": state["query"]})
            logger.info("✅ AgentExecutor invocation completed successfully!")
        except Exception as e:
            logger.error(f"❌ AgentExecutor invocation failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

        # COMPREHENSIVE RESPONSE ANALYSIS
        logger.info("🔍 ANALYZING AGENT RESPONSE")
        logger.info("-" * 40)
        logger.info(f"📦 Response type: {type(response)}")
        logger.info(f"🗝️  Response keys: {list(response.keys()) if hasattr(response, 'keys') else 'No keys method'}")

        # Log each key-value pair in detail
        if hasattr(response, 'keys'):
            for key in response.keys():
                value = response[key]
                logger.info(f"🔑 {key}: {type(value).__name__} = {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")

        # Look for verbose execution data
        verbose_keys = ['intermediate_steps', 'agent_scratchpad', 'actions', 'observations', 'thoughts', 'steps', 'trace']
        for key in verbose_keys:
            if key in response:
                logger.info(f"🎯 FOUND VERBOSE KEY '{key}': {type(response[key])} = {response[key]}")

        # Handle AgentExecutor response format
        # Extract tool outputs from intermediate_steps for search_results tracking
        if "intermediate_steps" in response and response["intermediate_steps"]:
            logger.info(f"🔧 Found {len(response['intermediate_steps'])} intermediate steps")
            tool_outputs = []
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) >= 2:
                    # step[1] is the tool output/observation
                    tool_output = str(step[1])
                    if tool_output and tool_output.strip():
                        tool_outputs.append(tool_output)
            state["search_results"] = tool_outputs
        else:
            # Fallback: use the final output as search results
            state["search_results"] = [response.get("output", "")]

        # Add the agent's final response as an AI message
        if "output" in response:
            logger.info(f"📤 Final output: {response['output'][:200]}...")
            assistant_msg = langchain_core.messages.AIMessage(content=response["output"])
            state["messages"].append(assistant_msg)

        # Final state logging
        logger.info("🏁 FINAL STATE")
        logger.info("-" * 40)
        logger.info(f"📊 Final messages count: {len(state.get('messages', []))}")
        logger.info(f"🔍 Search results count: {len(state.get('search_results', []))}")
        logger.info(f"✅ Resolved: {state.get('resolved', False)}")

        state["resolved"] = True
        logger.info("=" * 60)
        logger.info("🏁 FLIGHT SEARCH AGENT EXECUTION COMPLETED")
        logger.info("=" * 60)

        return state


class FlightSearchGraph(agentc_langgraph.graph.GraphRunnable):
    """Flight search conversation graph using Agent Catalog."""

    def __init__(self, catalog, span, chat_model=None):
        """Initialize the flight search graph with optional chat model."""
        super().__init__(catalog=catalog, span=span)
        self.chat_model = chat_model

    @staticmethod
    def build_starting_state(query: str) -> FlightSearchState:
        """Build the initial state for the flight search - single user system."""
        return FlightSearchState(
            messages=[],
            query=query,
            resolved=False,
            search_results=[],
        )

    def compile(self):
        """Compile the LangGraph workflow."""

        # Build the flight search agent with catalog integration
        search_agent = FlightSearchAgent(
            catalog=self.catalog, span=self.span, chat_model=self.chat_model
        )

        # Create a wrapper function for the ReActAgent
        def flight_search_node(state: FlightSearchState) -> FlightSearchState:
            """Wrapper function for the flight search ReActAgent."""
            return search_agent._invoke(
                span=self.span,
                state=state,
                config={},  # Empty config for now
            )

        # Create a simple workflow graph for flight search
        workflow = langgraph.graph.StateGraph(FlightSearchState)

        # Add the flight search agent node using the wrapper function
        workflow.add_node("flight_search", flight_search_node)

        # Set entry point and simple flow
        workflow.set_entry_point("flight_search")
        workflow.add_edge("flight_search", langgraph.graph.END)

        return workflow.compile()


def clear_bookings_and_reviews():
    """Clear existing flight bookings to start fresh for demo."""
    try:
        client = create_couchbase_client()
        client.connect()

        # Clear bookings scope using environment variables
        bookings_scope = "agentc_bookings"
        client.clear_scope(bookings_scope)
        logger.info(
            f"✅ Cleared existing flight bookings for fresh test run: {os.environ['CB_BUCKET']}.{bookings_scope}"
        )

        # Check if airline reviews collection needs clearing by comparing expected vs actual document count
        try:
            # Import to get expected document count without loading all data
            from data.airline_reviews_data import _data_manager

            # Get expected document count (this uses cached data if available)
            expected_docs = _data_manager.process_to_texts()
            expected_count = len(expected_docs)

            # Check current document count in collection
            try:
                count_query = f"SELECT COUNT(*) as count FROM `{os.environ['CB_BUCKET']}`.`{os.environ['CB_SCOPE']}`.`{os.environ['CB_COLLECTION']}`"
                count_result = client.cluster.query(count_query)
                count_row = next(iter(count_result))
                existing_count = count_row["count"]

                logger.info(
                    f"📊 Airline reviews collection: {existing_count} existing, {expected_count} expected"
                )

                if existing_count == expected_count:
                    logger.info(
                        f"✅ Collection already has correct document count ({existing_count}), skipping clear"
                    )
                else:
                    logger.info(
                        f"🗑️  Clearing airline reviews collection: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                    )
                    client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                    logger.info(
                        f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                    )

            except KeyspaceNotFoundException:
                # Collection doesn't exist yet - this is expected for fresh setup
                logger.info(
                    f"📊 Collection doesn't exist yet, will create and load fresh data"
                )
            except Exception as count_error:
                # Other query errors - clear anyway to ensure fresh start
                logger.info(
                    f"📊 Collection query failed, will clear and reload: {count_error}"
                )
                client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                logger.info(
                    f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                )

        except Exception as e:
            logger.warning(f"⚠️  Could not check collection count, clearing anyway: {e}")
            client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
            logger.info(
                f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
            )

    except Exception as e:
        logger.warning(f"❌ Could not clear bookings: {e}")


def query_agent_catalog_logs():
    """Query and display Agent Catalog activity logs."""
    try:
        # Connect to Agent Catalog cluster
        auth = PasswordAuthenticator(
            os.environ["AGENT_CATALOG_USERNAME"], os.environ["AGENT_CATALOG_PASSWORD"]
        )
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ["AGENT_CATALOG_CONN_STRING"], options)
        cluster.wait_until_ready(timedelta(seconds=10))

        bucket = os.environ["AGENT_CATALOG_BUCKET"]

        logger.info("Querying Agent Catalog activity logs...")
        logger.info("=" * 50)

        query = cluster.query(f"""
            FROM
                `{bucket}`.agent_activity.Sessions() s
            SELECT
                s.sid,
                s.cid,
                s.root,
                s.start_t,
                s.content,
                s.ann
            ORDER BY s.start_t DESC
            LIMIT 10;
        """)

        for result in query:
            print(f"Session ID: {result.get('sid', 'N/A')}")
            print(f"Content ID: {result.get('cid', 'N/A')}")
            print(f"Root: {result.get('root', 'N/A')}")
            print(f"Start Time: {result.get('start_t', 'N/A')}")
            print(f"Content: {result.get('content', 'N/A')}")
            print(f"Annotations: {result.get('ann', 'N/A')}")
            print("-" * 30)

    except Exception as e:
        logger.warning(f"Could not query Agent Catalog logs: {e}")


def setup_flight_search_agent():
    """Common setup function for flight search agent - returns all necessary components."""
    try:
        # Setup environment first
        setup_environment()

        # Initialize Agent Catalog
        catalog = agentc.Catalog(
            conn_string=os.environ["AGENT_CATALOG_CONN_STRING"],
            username=os.environ["AGENT_CATALOG_USERNAME"],
            password=SecretStr(os.environ["AGENT_CATALOG_PASSWORD"]),
            bucket=os.environ["AGENT_CATALOG_BUCKET"],
        )
        application_span = catalog.Span(name="Flight Search Agent", blacklist=set())

        # Test Capella AI connectivity
        if os.getenv("CAPELLA_API_ENDPOINT"):
            if not test_capella_connectivity():
                logger.warning("❌ Capella AI connectivity test failed. Will use OpenAI fallback.")
        else:
            logger.info("ℹ️ Capella API not configured - will use OpenAI models")

        # Create CouchbaseClient for all operations
        client = create_couchbase_client()

        # Setup everything in one call - bucket, scope, collection
        client.setup_collection(
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            clear_existing_data=False,  # Let data loader decide based on count check
        )

        # Setup vector search index
        try:
            with open("agentcatalog_index.json") as file:
                index_definition = json.load(file)
            logger.info("Loaded vector search index definition from agentcatalog_index.json")
            client.setup_vector_search_index(index_definition, os.environ["CB_SCOPE"])
        except Exception as e:
            logger.warning(f"Error loading index definition: {e!s}")
            logger.info("Continuing without vector search index...")

        # Setup embeddings using shared 4-case priority ladder
        embeddings, _ = setup_ai_services(framework="langgraph")

        # Import data loader function
        from data.airline_reviews_data import load_airline_reviews_to_couchbase

        # Setup vector store with airline reviews data
        vector_store = client.setup_vector_store_langchain(
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            index_name=os.environ["CB_INDEX"],
            embeddings=embeddings,
            data_loader_func=load_airline_reviews_to_couchbase,
        )

        # Setup LLM using shared 4-case priority ladder
        _, chat_model = setup_ai_services(framework="langgraph", temperature=0.1)

        # Create the flight search graph with the chat model
        flight_graph = FlightSearchGraph(
            catalog=catalog, span=application_span, chat_model=chat_model
        )
        # Compile the graph
        compiled_graph = flight_graph.compile()

        logger.info("Agent Catalog integration successful")

        return compiled_graph, application_span

    except Exception as e:
        logger.exception(f"Setup error: {e}")
        logger.info("Ensure Agent Catalog is published: agentc index . && agentc publish")
        raise


def run_interactive_demo():
    """Run an interactive flight search demo."""
    logger.info("Flight Search Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        compiled_graph, application_span = setup_flight_search_agent()

        # Interactive flight search loop
        logger.info("Available commands:")
        logger.info("- Enter flight search queries (e.g., 'Find flights from NYC to LAX')")
        logger.info("- 'logs' - View Agent Catalog activity logs")
        logger.info("- 'quit' - Exit the demo")
        logger.info(
            "Try asking: 'Find cheap flights to Miami' or 'Book a business class flight to Boston'"
        )
        logger.info("─" * 40)

        while True:
            query = input("🔍 Enter flight search query (or 'quit'/'logs'): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using Flight Search Agent!")
                break

            if query.lower() == "logs":
                logger.info("\n" + "=" * 50)
                logger.info("AGENT CATALOG ACTIVITY LOGS")
                logger.info("=" * 50)
                query_agent_catalog_logs()
                continue

            if not query:
                continue

            try:
                logger.info(f"Flight Query: {query}")

                # Build starting state - single user system
                state = FlightSearchGraph.build_starting_state(query=query)

                # Run the flight search
                result = compiled_graph.invoke(state)

                # Display results summary
                if result.get("search_results"):
                    logger.info(f"Found {len(result['search_results'])} flight options")

                logger.info(f"Search completed: {result.get('resolved', False)}")

            except Exception as e:
                logger.exception(f"Search error: {e}")

    except Exception as e:
        logger.exception(f"Demo initialization error: {e}")


def run_test():
    """Run comprehensive test of flight search agent with booking functionality."""
    logger.info("Flight Search Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        # Clear existing bookings first for a clean test run
        clear_bookings_and_reviews()

        compiled_graph, application_span = setup_flight_search_agent()

        # Comprehensive test scenarios covering all core functionality
        test_queries = [
            "Find flights from JFK to LAX for tomorrow",
            "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
            "Book an economy flight from JFK to MIA for next week, 1 passenger",
            "Show me my current flight bookings",
            "What do passengers say about SpiceJet's service quality?",
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}: {query}")
            try:
                state = FlightSearchGraph.build_starting_state(query=query)
                result = compiled_graph.invoke(state)

                if result.get("search_results"):
                    logger.info(f"✅ Found {len(result['search_results'])} flight options")
                logger.info(f"✅ Test {i} completed: {result.get('resolved', False)}")

            except Exception as e:
                logger.exception(f"❌ Test {i} failed: {e}")

            logger.info("-" * 50)

        logger.info("All tests completed!")
        logger.info("💡 Run 'python main.py logs' to view Agent Catalog activity logs")

    except Exception as e:
        logger.exception(f"Test error: {e}")


def run_flight_search_demo():
    """Legacy function - redirects to interactive demo for compatibility."""
    run_interactive_demo()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        elif sys.argv[1] == "logs":
            setup_environment()
            query_agent_catalog_logs()
        else:
            print("Usage: python main.py [test|logs]")
            print("  test - Run comprehensive test suite")
            print("  logs - Query Agent Catalog activity logs")
            print("  (no args) - Run interactive demo")
            sys.exit(1)
    else:
        run_interactive_demo()

    # Uncomment the following lines to visualize the LangGraph workflow:
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="flight_search_graph.png")
    # compiled_graph.get_graph().draw_ascii()