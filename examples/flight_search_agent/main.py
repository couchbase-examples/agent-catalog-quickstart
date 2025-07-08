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

# Setup logging with debug level for better troubleshooting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def setup_couchbase_connection():
    """Setup Couchbase cluster connection."""
    try:
        auth = PasswordAuthenticator(os.environ["CB_USERNAME"], os.environ["CB_PASSWORD"])
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ["CB_CONN_STRING"], options)
        cluster.wait_until_ready(timedelta(seconds=10))
        logger.info("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {e!s}")


def setup_collection(cluster, bucket_name, scope_name, collection_name):
    """Setup Couchbase bucket, scope and collection."""
    try:
        try:
            bucket = cluster.bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' exists")
        except Exception:
            logger.info(f"Creating bucket '{bucket_name}'...")
            bucket_settings = CreateBucketSettings(
                name=bucket_name,
                bucket_type="couchbase",
                ram_quota_mb=1024,
                flush_enabled=True,
                num_replicas=0,
            )
            cluster.buckets().create_bucket(bucket_settings)
            time.sleep(5)
            bucket = cluster.bucket(bucket_name)
            logger.info(f"Bucket '{bucket_name}' created successfully")

        bucket_manager = bucket.collections()

        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)

        if not scope_exists and scope_name != "_default":
            logger.info(f"Creating scope '{scope_name}'...")
            bucket_manager.create_scope(scope_name)
            logger.info(f"Scope '{scope_name}' created successfully")

        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            logger.info(f"Creating collection '{collection_name}'...")
            bucket_manager.create_collection(scope_name, collection_name)
            logger.info(f"Collection '{collection_name}' created successfully")

        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(3)

        try:
            cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            ).execute()
            logger.info("Primary index created successfully")
        except Exception as e:
            logger.warning(f"Error creating primary index: {e!s}")

        logger.info("Collection setup complete")

        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {e!s}")


def setup_vector_search_index(cluster, index_definition):
    """Setup vector search index for flight data."""
    try:
        scope_index_manager = (
            cluster.bucket(os.environ["CB_BUCKET"]).scope(os.environ["SCOPE_NAME"]).search_indexes()
        )

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


def load_flight_data():
    """Load flight data from our enhanced flight_data.py file."""
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


def setup_vector_store(cluster):
    """Setup vector store and load flight data."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"
        )

        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ["CB_BUCKET"],
            scope_name=os.environ["SCOPE_NAME"],
            collection_name=os.environ["COLLECTION_NAME"],
            embedding=embeddings,
            index_name=os.environ["INDEX_NAME"],
        )

        flight_data = load_flight_data()

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


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations - single user system."""

    query: str
    resolved: bool
    search_results: list[dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and prompts."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the flight search agent."""

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

        # Initialize parameter mapper for intelligent parameter handling
        self.parameter_mapper = ParameterMapper(chat_model)

        super().__init__(
            chat_model=chat_model,
            catalog=catalog,
            span=span,
            prompt_name="flight_search_assistant",
        )

    def _invoke(
        self,
        span: agentc.Span,
        state: FlightSearchState,
        config: langchain_core.runnables.RunnableConfig,
    ) -> FlightSearchState:
        """Handle flight search conversation with Agent Catalog tools."""

        # Initialize conversation if this is the first message
        if not state["messages"]:
            initial_msg = langchain_core.messages.HumanMessage(content=state["query"])
            state["messages"].append(initial_msg)
            logger.info(f"Flight Query: {state['query']}")

        try:
            # Get tools from Agent Catalog using find method like hotel agent
            from langchain_core.tools import Tool

            tools = []
            tool_configs = [
                ("lookup_flight_info", "Find flight routes between airports"),
                ("save_flight_booking", "Save flight booking requests"),
                ("retrieve_flight_bookings", "Retrieve existing flight bookings"),
                ("search_flight_policies", "Search flight policies and guidelines"),
            ]

            for tool_name, description in tool_configs:
                try:
                    tool_obj = self.catalog.find("tool", name=tool_name)
                    if tool_obj and hasattr(tool_obj, "func"):
                        # Create a wrapper function using intelligent parameter mapping
                        def create_tool_wrapper(func, tool_name):
                            def wrapper(*args, **kwargs):
                                try:
                                    logger.debug(
                                        f"Tool {tool_name} called with args: {args}, kwargs: {kwargs}"
                                    )

                                    # Case 1: Handle ReAct agent format with multiple positional args
                                    if len(args) > 1:
                                        # ReAct agent passes multiple string arguments like ("SFO", "LAX")
                                        mapped_args = self.parameter_mapper.map_positional_args(
                                            tool_name, args, func
                                        )
                                        return func(**mapped_args)

                                    # Case 2: Handle single string argument (could be JSON or plain string)
                                    elif len(args) == 1 and isinstance(args[0], str) and not kwargs:
                                        input_str = args[0].strip()

                                        # Handle empty string or "None" for tools that take no args
                                        if not input_str or input_str.lower() in [
                                            "none",
                                            "null",
                                            "",
                                        ]:
                                            if tool_name == "retrieve_flight_bookings":
                                                return func()  # Call with no arguments
                                            else:
                                                # For other tools, try with empty dict
                                                mapped_args = (
                                                    self.parameter_mapper.map_parameters_smart(
                                                        tool_name, {}, func
                                                    )
                                                )
                                                return func(**mapped_args)

                                        # Try to parse as JSON first
                                        try:
                                            import json

                                            parsed_args = json.loads(input_str)

                                            if isinstance(parsed_args, dict):
                                                mapped_args = (
                                                    self.parameter_mapper.map_parameters_smart(
                                                        tool_name, parsed_args, func
                                                    )
                                                )
                                                return func(**mapped_args)
                                        except (json.JSONDecodeError, TypeError):
                                            pass  # Not JSON, continue to other parsing methods

                                        # Handle comma-separated format like "SFO", "LAX"
                                        if "," in input_str and '"' in input_str:
                                            try:
                                                # Parse comma-separated quoted strings
                                                import ast

                                                # Convert string like '"SFO", "LAX"' to tuple
                                                formatted_str = f"({input_str})"
                                                parsed_tuple = ast.literal_eval(formatted_str)
                                                if isinstance(parsed_tuple, tuple):
                                                    mapped_args = (
                                                        self.parameter_mapper.map_positional_args(
                                                            tool_name, parsed_tuple, func
                                                        )
                                                    )
                                                    return func(**mapped_args)
                                            except (ValueError, SyntaxError):
                                                pass

                                        # Fallback: treat as single parameter for search tools
                                        mapped_args = self.parameter_mapper.map_string_input(
                                            tool_name, input_str, func
                                        )
                                        return func(**mapped_args)

                                    # Case 3: Normal function call with kwargs
                                    elif kwargs:
                                        mapped_args = self.parameter_mapper.map_parameters_smart(
                                            tool_name, kwargs, func
                                        )
                                        return func(**mapped_args)

                                    # Case 4: No arguments - for tools like retrieve_flight_bookings
                                    elif len(args) == 0:
                                        if tool_name == "retrieve_flight_bookings":
                                            return func()
                                        else:
                                            mapped_args = (
                                                self.parameter_mapper.map_parameters_smart(
                                                    tool_name, {}, func
                                                )
                                            )
                                            return func(**mapped_args)

                                    # Default fallback
                                    return func(*args, **kwargs)

                                except Exception as e:
                                    logger.error(f"Tool {tool_name} execution error: {e}")
                                    logger.debug(f"Failed with args: {args}, kwargs: {kwargs}")
                                    return f"Error executing {tool_name}: {e!s}"

                            return wrapper

                        # Convert to LangChain Tool format with closure fix
                        wrapped_func = create_tool_wrapper(tool_obj.func, tool_name)
                        lc_tool = Tool(
                            name=tool_obj.meta.name,
                            description=tool_obj.meta.description or description,
                            func=wrapped_func,
                        )
                        tools.append(lc_tool)
                        logger.info(f"Loaded tool: {tool_name}")
                except Exception as e:
                    logger.warning(f"Could not load tool {tool_name}: {e}")

            if not tools:
                logger.error("No tools loaded from catalog")
                error_msg = langchain_core.messages.AIMessage(
                    content="I'm unable to access my flight search tools right now. Please try again later."
                )
                state["messages"].append(error_msg)
                state["resolved"] = True
                return state

            # Get prompt from Agent Catalog instead of hardcoded ReAct
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_core.prompts import PromptTemplate

            try:
                # Get prompt from Agent Catalog
                flight_prompt = self.catalog.find("prompt", name="flight_search_assistant")
                if not flight_prompt:
                    raise ValueError(
                        "Could not find flight_search_assistant prompt in catalog. Make sure it's indexed with 'agentc index prompts/'"
                    )

                # Create a custom prompt using the catalog prompt content
                catalog_prompt_content = flight_prompt.content

                # Create prompt template that includes the catalog content and ReAct format
                react_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

{catalog_content}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

                react_prompt = PromptTemplate.from_template(react_template).partial(
                    catalog_content=catalog_prompt_content
                )

                # Create ReAct agent with tools and catalog prompt
                agent = create_react_agent(self.chat_model, tools, react_prompt)
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=3,  # Limit iterations to prevent loops
                    return_intermediate_steps=True,  # Better error handling
                )

                # Execute the agent with the query
                query = state["query"]
                result = agent_executor.invoke({"input": query})

                # Create response message
                response_content = result.get(
                    "output", "I was unable to process your flight search request."
                )
                response_msg = langchain_core.messages.AIMessage(content=response_content)
                state["messages"].append(response_msg)

                logger.info(f"Agent Response: {response_content}")

                # Mark as resolved
                state["resolved"] = True

            except Exception as agent_error:
                logger.error(f"ReAct agent error: {agent_error}")
                # Fallback to simple model response
                fallback_response = self.chat_model.invoke(state["messages"])
                state["messages"].append(fallback_response)

                if hasattr(fallback_response, "content"):
                    logger.info(f"Fallback Response: {fallback_response.content}")

                state["resolved"] = True

        except Exception as e:
            logger.error(f"Agent invocation error: {e}")
            error_msg = langchain_core.messages.AIMessage(
                content=f"I encountered an error while processing your request: {e!s}"
            )
            state["messages"].append(error_msg)
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

        # Create a simple workflow graph for flight search
        workflow = langgraph.graph.StateGraph(FlightSearchState)

        # Add the flight search agent node
        workflow.add_node("flight_search", search_agent)

        # Set entry point and simple flow
        workflow.set_entry_point("flight_search")
        workflow.add_edge("flight_search", langgraph.graph.END)

        return workflow.compile()


def run_flight_search_demo():
    """Run an interactive flight search demo."""

    logger.info("Flight Search Agent - Agent Catalog Demo")
    logger.info("=" * 50)

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

        with application_span.new("Couchbase Connection"):
            # Setup Couchbase infrastructure
            cluster = setup_couchbase_connection()

        with application_span.new("Couchbase Collection Setup"):
            setup_collection(
                cluster,
                os.environ["CB_BUCKET"],
                os.environ["SCOPE_NAME"],
                os.environ["COLLECTION_NAME"],
            )

        with application_span.new("Vector Index Setup"):
            try:
                with open("agentcatalog_index.json") as file:
                    index_definition = json.load(file)
                logger.info("Loaded vector search index definition from agentcatalog_index.json")
            except Exception as e:
                logger.warning(f"Error loading index definition: {e!s}")
                logger.info("Continuing without vector search index...")

            if "index_definition" in locals():
                setup_vector_search_index(cluster, index_definition)

        with application_span.new("Vector Store Setup"):
            setup_vector_store(cluster)

        with application_span.new("Agent Graph Creation"):
            # Create the flight search graph
            flight_graph = FlightSearchGraph(catalog=catalog, span=application_span)
            # Compile the graph
            compiled_graph = flight_graph.compile()

        logger.info("Agent Catalog integration successful")

        # Interactive flight search loop
        with application_span.new("Query Execution") as span:
            while True:
                logger.info("─" * 40)
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
        logger.error(f"Initialization error: {e}")
        logger.info("Ensure Agent Catalog is published: agentc index . && agentc publish")


if __name__ == "__main__":
    run_flight_search_demo()

    # Uncomment the following lines to visualize the LangGraph workflow:
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="flight_search_graph.png")
    # compiled_graph.get_graph().draw_ascii()
