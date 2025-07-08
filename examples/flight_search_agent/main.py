#!/usr/bin/env python3
"""
Flight Search Agent - Agent Catalog + LangGraph Implementation

A streamlined flight search agent demonstrating Agent Catalog integration
with LangGraph and Couchbase vector search for flight booking assistance.
"""

import getpass
import json
import logging
import os
import sys
import time
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
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings

from parameter_mapper import ParameterMapper

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


def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup required environment variables with defaults."""
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

    os.environ["INDEX_NAME"] = os.getenv("INDEX_NAME", "vector_search_agentcatalog")
    os.environ["SCOPE_NAME"] = os.getenv("SCOPE_NAME", "shared")
    os.environ["COLLECTION_NAME"] = os.getenv("COLLECTION_NAME", "agentcatalog")


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
            self.cluster = Cluster(self.conn_string, options)
            self.cluster.wait_until_ready(timedelta(seconds=10))
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

    def load_flight_data(self):
        """Load flight data from the enhanced flight_data.py file."""
        try:
            # Import flight data
            sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
            from flight_data import get_all_flight_data

            flight_data = get_all_flight_data()

            # Convert to text format for vector store
            flight_texts = []
            for item in flight_data:
                text = f"{item['title']} - {item['content']}"
                flight_texts.append(text)

            return flight_texts
        except Exception as e:
            raise ValueError(f"Error loading flight data: {e!s}")

    def setup_vector_store(
        self, scope_name: str, collection_name: str, index_name: str, embeddings
    ):
        """Setup vector store with flight data."""
        try:
            if not self.cluster:
                raise RuntimeError("Cluster not connected. Call connect first.")

            vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )

            flight_data = self.load_flight_data()

            try:
                vector_store.add_texts(texts=flight_data, batch_size=10)
                logger.info("Flight data loaded into vector store successfully")
            except Exception as e:
                logger.warning(
                    f"Error loading flight data: {e!s}. Vector store created but data not loaded."
                )

            return vector_store
        except Exception as e:
            raise ValueError(f"Error setting up vector store: {e!s}")

    def clear_scope(self, scope_name: str):
        """Clear all collections in the specified scope."""
        try:
            if not self.bucket:
                # Ensure connection and bucket are ready
                if not self.cluster:
                    self.connect()
                self.bucket = self.cluster.bucket(self.bucket_name)

            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()

            # Find the target scope
            target_scope = None
            for scope in scopes:
                if scope.name == scope_name:
                    target_scope = scope
                    break

            if not target_scope:
                logger.info(f"Scope '{scope_name}' does not exist, nothing to clear")
                return

            # Clear all collections in the scope
            for collection in target_scope.collections:
                try:
                    delete_query = (
                        f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection.name}`"
                    )
                    self.cluster.query(delete_query).execute()
                    logger.info(f"Cleared collection '{collection.name}' in scope '{scope_name}'")
                except Exception as e:
                    logger.warning(f"Could not clear collection '{collection.name}': {e}")

            logger.info(f"Cleared all collections in scope '{scope_name}'")

        except Exception as e:
            logger.warning(f"Could not clear scope '{scope_name}': {e}")


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations - single user system."""

    query: str
    resolved: bool
    search_results: list[dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and ReActAgent framework."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the flight search agent."""

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

        # Initialize parameter mapper for intelligent parameter handling
        parameter_mapper = ParameterMapper(self.chat_model)

        # Get tools from Agent Catalog and create wrapper functions for parameter mapping
        from langchain_core.tools import Tool

        tools = []
        for tool_name in [
            "lookup_flight_info",
            "save_flight_booking",
            "retrieve_flight_bookings",
            "search_flight_policies",
        ]:
            catalog_tool = self.catalog.find("tool", name=tool_name)
            logger.info(f"Loaded tool: {tool_name}")

            # Create wrapper function to handle parameter mapping using ParameterMapper
            def create_tool_wrapper(original_tool, name):
                def wrapper_func(tool_input: str) -> str:
                    """Wrapper to handle parameter mapping using ParameterMapper."""
                    try:
                        # Use ParameterMapper to intelligently map string input to parameters
                        mapped_params = parameter_mapper.map_string_input(
                            name, tool_input, original_tool.func
                        )

                        # Call the original tool with mapped parameters
                        return original_tool.func(**mapped_params)

                    except Exception as e:
                        return f"Error calling {name}: {e!s}"

                return wrapper_func

            # Create LangChain tool with wrapper
            langchain_tool = Tool(
                name=tool_name,
                description=f"Tool for {tool_name.replace('_', ' ')}",
                func=create_tool_wrapper(catalog_tool, tool_name),
            )
            tools.append(langchain_tool)

        # Get prompt from Agent Catalog directly
        prompt_resource = self.catalog.find("prompt", name="flight_search_assistant")

        # Convert PromptResult to LangChain compatible format
        from langchain.agents import create_react_agent
        from langchain_core.prompts import PromptTemplate

        # Use the Agent Catalog prompt content directly - it already has ReAct format
        react_prompt = PromptTemplate.from_template(prompt_resource.content)

        # Create ReAct agent with tools and prompt
        agent = create_react_agent(self.chat_model, tools, react_prompt)

        # Create agent executor
        from langchain.agents import AgentExecutor

        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10
        )

        # Execute the agent
        response = agent_executor.invoke({"input": state["query"]})

        # Add response to conversation
        assistant_msg = langchain_core.messages.AIMessage(content=response["output"])
        state["messages"].append(assistant_msg)
        state["resolved"] = True

        return state


class FlightSearchGraph(agentc_langgraph.graph.GraphRunnable):
    """Flight search conversation graph using Agent Catalog."""

    @staticmethod
    def build_starting_state(query: str) -> FlightSearchState:
        """Build the initial state for the flight search - single user system."""
        return FlightSearchState(
            messages=[],
            query=query,
            resolved=False,
            search_results=[],
            previous_node=None,
        )

    def compile(self) -> langgraph.graph.graph.CompiledGraph:
        """Compile the LangGraph workflow."""

        # Build the flight search agent with catalog integration
        search_agent = FlightSearchAgent(catalog=self.catalog, span=self.span)

        # Create a wrapper function for the ReActAgent
        def flight_search_node(state: FlightSearchState) -> FlightSearchState:
            """Wrapper function for the flight search ReActAgent."""
            with self.span.new("Flight Search Node") as node_span:
                return search_agent._invoke(
                    span=node_span,
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


def clear_flight_bookings():
    """Clear existing flight bookings to start fresh for demo."""
    try:
        client = CouchbaseClient(
            conn_string=os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
            bucket_name=os.getenv("CB_BUCKET", "vector-search-testing"),
        )
        client.connect()

        scope_name = "agentc_bookings"

        # Clear entire scope instead of individual collections
        client.clear_scope(scope_name)
        logger.info("Cleared existing flight bookings for fresh test run")

    except Exception as e:
        logger.warning(f"Could not clear bookings: {e}")


def setup_flight_search_agent():
    """Common setup function for flight search agent - returns all necessary components."""
    try:
        # Setup environment first
        setup_environment()

        # Initialize Agent Catalog
        catalog = agentc.Catalog(
            conn_string=os.environ["AGENT_CATALOG_CONN_STRING"],
            username=os.environ["AGENT_CATALOG_USERNAME"],
            password=os.environ["AGENT_CATALOG_PASSWORD"],
            bucket=os.environ["AGENT_CATALOG_BUCKET"],
        )
        application_span = catalog.Span(name="Flight Search Agent")

        with application_span.new("Couchbase Setup"):
            # Create CouchbaseClient for all operations
            client = CouchbaseClient(
                conn_string=os.environ["CB_CONN_STRING"],
                username=os.environ["CB_USERNAME"],
                password=os.environ["CB_PASSWORD"],
                bucket_name=os.environ["CB_BUCKET"],
            )

            # Setup everything in one call - bucket, scope, collection
            client.setup_collection(
                scope_name=os.environ["SCOPE_NAME"], collection_name=os.environ["COLLECTION_NAME"]
            )

        with application_span.new("Vector Index Setup"):
            try:
                with open("agentcatalog_index.json") as file:
                    index_definition = json.load(file)
                logger.info("Loaded vector search index definition from agentcatalog_index.json")
                client.setup_vector_search_index(index_definition, os.environ["SCOPE_NAME"])
            except Exception as e:
                logger.warning(f"Error loading index definition: {e!s}")
                logger.info("Continuing without vector search index...")

        with application_span.new("Vector Store Setup"):
            embeddings = OpenAIEmbeddings(
                api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"
            )
            client.setup_vector_store(
                scope_name=os.environ["SCOPE_NAME"],
                collection_name=os.environ["COLLECTION_NAME"],
                index_name=os.environ["INDEX_NAME"],
                embeddings=embeddings,
            )

        with application_span.new("Agent Graph Creation"):
            # Create the flight search graph
            flight_graph = FlightSearchGraph(catalog=catalog, span=application_span)
            # Compile the graph
            compiled_graph = flight_graph.compile()

        logger.info("Agent Catalog integration successful")

        return compiled_graph, application_span

    except Exception as e:
        logger.error(f"Setup error: {e}")
        logger.info("Ensure Agent Catalog is published: agentc index . && agentc publish")
        raise


def run_interactive_demo():
    """Run an interactive flight search demo."""
    logger.info("Flight Search Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        compiled_graph, application_span = setup_flight_search_agent()

        # Interactive flight search loop
        with application_span.new("Query Execution") as span:
            logger.info("Available commands:")
            logger.info("- Enter flight search queries (e.g., 'Find flights from NYC to LAX')")
            logger.info("- 'quit' - Exit the demo")
            logger.info(
                "Try asking: 'Find cheap flights to Miami' or 'Book a business class flight to Boston'"
            )
            logger.info("─" * 40)

            while True:
                query = input("🔍 Enter flight search query (or 'quit' to exit): ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    logger.info("Thanks for using Flight Search Agent!")
                    break

                if not query:
                    continue

                with span.new(f"Query: {query}") as query_span:
                    try:
                        logger.info(f"Flight Query: {query}")
                        query_span["query"] = query

                        # Build starting state - single user system
                        state = FlightSearchGraph.build_starting_state(query=query)

                        # Run the flight search
                        result = compiled_graph.invoke(state)
                        query_span["result"] = result

                        # Display results summary
                        if result.get("search_results"):
                            logger.info(f"Found {len(result['search_results'])} flight options")

                        logger.info(f"Search completed: {result.get('resolved', False)}")

                    except Exception as e:
                        logger.error(f"Search error: {e}")
                        query_span["error"] = str(e)

    except Exception as e:
        logger.error(f"Demo initialization error: {e}")


def run_test():
    """Run comprehensive test of flight search agent with booking functionality."""
    logger.info("Flight Search Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        # Clear existing bookings first for a clean test run
        clear_flight_bookings()

        compiled_graph, application_span = setup_flight_search_agent()

        # Comprehensive test scenarios covering all core functionality
        test_queries = [
            "Find flights from JFK to LAX for tomorrow",
            "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
            "Book an economy flight from JFK to MIA for next week, 1 passenger",
            "Show me my current flight bookings",
            "What is the baggage policy for international flights?",
        ]

        with application_span.new("Test Queries") as span:
            for i, query in enumerate(test_queries, 1):
                with span.new(f"Test {i}: {query}") as query_span:
                    logger.info(f"\n🔍 Test {i}: {query}")
                    try:
                        query_span["query"] = query
                        state = FlightSearchGraph.build_starting_state(query=query)
                        result = compiled_graph.invoke(state)
                        query_span["result"] = result

                        if result.get("search_results"):
                            logger.info(f"✅ Found {len(result['search_results'])} flight options")
                        logger.info(f"✅ Test {i} completed: {result.get('resolved', False)}")

                    except Exception as e:
                        logger.error(f"❌ Test {i} failed: {e}")
                        query_span["error"] = str(e)

                    logger.info("-" * 50)

        logger.info("All tests completed!")

    except Exception as e:
        logger.error(f"Test error: {e}")


def run_flight_search_demo():
    """Legacy function - redirects to interactive demo for compatibility."""
    run_interactive_demo()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        else:
            run_interactive_demo()
    else:
        run_interactive_demo()

    # Uncomment the following lines to visualize the LangGraph workflow:
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="flight_search_graph.png")
    # compiled_graph.get_graph().draw_ascii()
