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

        super().__init__(
            chat_model=chat_model, catalog=catalog, span=span, prompt_name="flight_search_assistant"
        )

    def _invoke(
        self,
        span: agentc.Span,
        state: FlightSearchState,
        config: langchain_core.runnables.RunnableConfig,
    ) -> FlightSearchState:
        """Handle flight search conversation using ReActAgent."""

        # Initialize conversation if this is the first message
        if not state["messages"]:
            initial_msg = langchain_core.messages.HumanMessage(content=state["query"])
            state["messages"].append(initial_msg)
            logger.info(f"Flight Query: {state['query']}")

        # Get prompt resource first - we'll need it for the ReAct agent
        prompt_resource = self.catalog.find("prompt", name="flight_search_assistant")

        # Get tools from Agent Catalog with simplified discovery
        tools = []
        tool_names = [
            "lookup_flight_info",
            "save_flight_booking", 
            "retrieve_flight_bookings",
            "search_airline_reviews",
        ]

        for tool_name in tool_names:
            try:
                # Find tool using Agent Catalog
                catalog_tool = self.catalog.find("tool", name=tool_name)
                if catalog_tool:
                    logger.info(f"✅ Found tool: {tool_name}")
                else:
                    logger.error(f"❌ Tool not found: {tool_name}")
                    continue

            except Exception as e:
                logger.error(f"❌ Failed to find tool {tool_name}: {e}")
                continue

            # Create wrapper function to handle proper parameter parsing
            def create_tool_wrapper(original_tool, name):
                """Create a wrapper for Agent Catalog tools with robust input handling."""

                def wrapper_func(tool_input: str) -> str:
                    """Wrapper function that handles input parsing and error handling."""
                    try:
                        logger.info(f"🔧 Tool {name} called with raw input: {repr(tool_input)}")

                        # Robust input sanitization to handle ReAct format artifacts
                        if isinstance(tool_input, str):
                            # Remove ReAct format artifacts that get mixed into input
                            clean_input = tool_input.strip()
                            
                            # Remove common ReAct artifacts
                            artifacts_to_remove = [
                                '\nObservation', 'Observation', '\nThought:', 'Thought:', 
                                '\nAction:', 'Action:', '\nAction Input:', 'Action Input:',
                                '\nFinal Answer:', 'Final Answer:'
                            ]
                            
                            for artifact in artifacts_to_remove:
                                if artifact in clean_input:
                                    clean_input = clean_input.split(artifact)[0]
                            
                            # Clean up quotes and whitespace
                            clean_input = clean_input.strip().strip("\"'").strip()
                            # Normalize whitespace
                            clean_input = " ".join(clean_input.split())
                            
                            tool_input = clean_input

                        logger.info(f"🧹 Tool {name} cleaned input: {repr(tool_input)}")

                        # Call appropriate tool with proper parameter handling
                        if name == "lookup_flight_info":
                            # Parse airport codes from input
                            import re

                            source = None
                            dest = None

                            # 1) Support key=value style inputs from ReAct (e.g., source_airport="JFK", destination_airport="LAX")
                            try:
                                m_src = re.search(r"source_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
                                m_dst = re.search(r"destination_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
                                if m_src and m_dst:
                                    source = m_src.group(1).upper()
                                    dest = m_dst.group(1).upper()
                            except Exception:
                                pass

                            # 2) Fallback: comma separated codes (e.g., "JFK,LAX")
                            if source is None or dest is None:
                                if ',' in tool_input:
                                    parts = tool_input.split(',')
                                    if len(parts) >= 2:
                                        source = parts[0].strip().upper()
                                        dest = parts[1].strip().upper()

                            # 3) Fallback: natural language (e.g., "JFK to LAX")
                            if source is None or dest is None:
                                words = tool_input.upper().split()
                                airport_codes = [w for w in words if len(w) == 3 and w.isalpha()]
                                if len(airport_codes) >= 2:
                                    source, dest = airport_codes[0], airport_codes[1]

                            if not source or not dest:
                                return "Error: Please provide source and destination airports (e.g., JFK,LAX or JFK to LAX)"
                            
                            result = original_tool.func(source_airport=source, destination_airport=dest)

                        elif name == "save_flight_booking":
                            result = original_tool.func(booking_input=tool_input)

                        elif name == "retrieve_flight_bookings":
                            # Handle empty input for "all bookings"
                            if not tool_input or tool_input.lower() in ["", "all", "none"]:
                                result = original_tool.func(booking_query="")
                            else:
                                result = original_tool.func(booking_query=tool_input)

                        elif name == "search_airline_reviews":
                            if not tool_input:
                                return "Error: Please provide a search query for airline reviews"
                            result = original_tool.func(query=tool_input)

                        else:
                            # Generic fallback - pass as first positional argument
                            result = original_tool.func(tool_input)

                        logger.info(f"✅ Tool {name} executed successfully")
                        return str(result) if result is not None else "No results found"

                    except Exception as e:
                        error_msg = f"Error in tool {name}: {str(e)}"
                        logger.error(f"❌ {error_msg}")
                        return error_msg

                return wrapper_func

            # Create LangChain tool with descriptive information
            tool_descriptions = {
                "lookup_flight_info": "Find available flights between airports. Input: 'JFK,LAX' or 'JFK to LAX'. Returns flight options with airlines and aircraft.",
                "save_flight_booking": "Create a flight booking. Input: 'JFK,LAX,2025-12-25' or natural language. Handles passenger count and class automatically.",
                "retrieve_flight_bookings": "View existing bookings. Input: empty string for all bookings, or 'JFK,LAX,2025-12-25' for specific booking.",
                "search_airline_reviews": "Search airline customer reviews. Input: 'SpiceJet service' or 'food quality'. Returns passenger reviews and ratings."
            }
            
            langchain_tool = Tool(
                name=tool_name,
                description=tool_descriptions.get(tool_name, f"Tool for {tool_name.replace('_', ' ')}"),
                func=create_tool_wrapper(catalog_tool, tool_name),
            )
            tools.append(langchain_tool)

        # Use the Agent Catalog prompt content directly - get first result if it's a list
        if isinstance(prompt_resource, list):
            prompt_resource = prompt_resource[0]

        # Safely get the content from the prompt resource
        prompt_content = getattr(prompt_resource, "content", "")
        if not prompt_content:
            prompt_content = "You are a helpful flight search assistant. Use the available tools to help users with their flight queries."

        # Inject current date into the prompt content
        import datetime

        current_date = datetime.date.today().strftime("%Y-%m-%d")
        prompt_content = prompt_content.replace("{current_date}", current_date)

        # Use the Agent Catalog prompt content directly - it already has ReAct format
        react_prompt = PromptTemplate.from_template(str(prompt_content))

        # Create ReAct agent with tools and prompt
        agent = create_react_agent(self.chat_model, tools, react_prompt)

        # Custom parsing error handler - force stopping on parsing errors
        def handle_parsing_errors(error):
            """Custom handler for parsing errors - force early termination."""
            error_msg = str(error)
            if "both a final answer and a parse-able action" in error_msg:
                # Force early termination - return a reasonable response
                return "Final Answer: I encountered a parsing error. Please reformulate your request."
            elif "Missing 'Action:'" in error_msg:
                return "I need to use the correct format with Action: and Action Input:"
            else:
                return f"Final Answer: I encountered an error processing your request. Please try again."

        # Create agent executor - very strict: only 2 iterations max
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=handle_parsing_errors,
            max_iterations=2,  # STRICT: 1 tool call + 1 Final Answer only
            early_stopping_method="force",  # Force stop
            return_intermediate_steps=True,
        )

        # Execute the agent
        response = agent_executor.invoke({"input": state["query"]})

        # Extract tool outputs from intermediate_steps and store in search_results
        if "intermediate_steps" in response and response["intermediate_steps"]:
            tool_outputs = []
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) >= 2:
                    # step[0] is the action, step[1] is the tool output/observation
                    tool_output = str(step[1])
                    if tool_output and tool_output.strip():
                        tool_outputs.append(tool_output)
            state["search_results"] = tool_outputs

        # Add response to conversation
        assistant_msg = langchain_core.messages.AIMessage(content=response["output"])
        state["messages"].append(assistant_msg)
        state["resolved"] = True

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
