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
from couchbase.management.buckets import BucketType, CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

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


def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup required environment variables with defaults."""
    required_vars = [
        "OPENAI_API_KEY",
        "CB_CONN_STRING",
        "CB_USERNAME",
        "CB_PASSWORD",
        "CB_BUCKET",
        "AGENT_CATALOG_CONN_STRING",
        "AGENT_CATALOG_USERNAME",
        "AGENT_CATALOG_PASSWORD",
        "AGENT_CATALOG_BUCKET",
    ]
    for var in required_vars:
        _set_if_undefined(var)

    # Set non-sensitive defaults for bucket names and Agent Catalog local development
    non_sensitive_defaults = {
        "CB_BUCKET": "travel-sample",
        "AGENT_CATALOG_CONN_STRING": "couchbase://127.0.0.1",
        "AGENT_CATALOG_USERNAME": "Administrator",
        "AGENT_CATALOG_PASSWORD": "password",
        "AGENT_CATALOG_BUCKET": "travel-sample",
    }

    for key, default_value in non_sensitive_defaults.items():
        if not os.environ.get(key):
            os.environ[key] = input(f"Enter {key} (default: {default_value}): ") or default_value

    os.environ["CB_INDEX"] = os.getenv("CB_INDEX", "airline_reviews_index")
    os.environ["CB_SCOPE"] = os.getenv("CB_SCOPE", "agentc_data")
    os.environ["CB_COLLECTION"] = os.getenv("CB_COLLECTION", "airline_reviews")


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
            if not self.bucket and self.cluster is not None:
                try:
                    self.bucket = self.cluster.bucket(self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' exists")
                except Exception:
                    logger.info(f"Creating bucket '{self.bucket_name}'...")
                    bucket_settings = CreateBucketSettings(
                        name=self.bucket_name,
                        bucket_type=BucketType.COUCHBASE,
                        ram_quota_mb=1024,
                        flush_enabled=True,
                        num_replicas=0,
                    )
                    self.cluster.buckets().create_bucket(bucket_settings)
                    time.sleep(5)
                    self.bucket = self.cluster.bucket(self.bucket_name)
                    logger.info(f"Bucket '{self.bucket_name}' created successfully")

            if not self.bucket:
                raise RuntimeError("Failed to initialize bucket")

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
            if self.cluster:
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

    def setup_vector_store(
        self, scope_name: str, collection_name: str, index_name: str, embeddings
    ):
        """Setup vector store with airline reviews data using unified data manager."""
        try:
            if not self.cluster:
                raise RuntimeError("Cluster not connected. Call connect first.")

            # Import the unified data manager
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
            from airline_reviews_data import load_airline_reviews_to_couchbase

            # NOTE: Commented out as data is already loaded in Couchbase
            logger.info("🔄 Setting up vector store with airline reviews data...")

            # Use the unified data loading approach
            load_airline_reviews_to_couchbase(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embeddings=embeddings,
                index_name=index_name,
            )

            logger.info(
                f"✅ Vector store setup complete: {self.bucket_name}.{scope_name}.{collection_name}"
            )

            # Create and return the vector store instance
            vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )

            logger.info(
                f"✅ Vector store setup complete: {self.bucket_name}.{scope_name}.{collection_name}"
            )
            return vector_store

        except Exception as e:
            logger.exception(f"Error setting up vector store: {e!s}")
            raise

    def clear_scope(self, scope_name: str):
        """Clear all collections in the specified scope."""
        try:
            if not self.bucket:
                # Ensure connection and bucket are ready
                if not self.cluster:
                    self.connect()
                if self.cluster:
                    self.bucket = self.cluster.bucket(self.bucket_name)

            if not self.bucket:
                logger.warning("Cannot clear scope - bucket not available")
                return

            logger.info(f"🗑️  Clearing scope: {self.bucket_name}.{scope_name}")
            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()

            # Find the target scope
            target_scope = None
            for scope in scopes:
                if scope.name == scope_name:
                    target_scope = scope
                    break

            if not target_scope:
                logger.info(
                    f"Scope '{self.bucket_name}.{scope_name}' does not exist, nothing to clear"
                )
                return

            # Clear all collections in the scope
            for collection in target_scope.collections:
                try:
                    delete_query = (
                        f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection.name}`"
                    )
                    if self.cluster:
                        self.cluster.query(delete_query).execute()
                        logger.info(
                            f"✅ Cleared collection: {self.bucket_name}.{scope_name}.{collection.name}"
                        )
                except Exception as e:
                    logger.warning(
                        f"❌ Could not clear collection {self.bucket_name}.{scope_name}.{collection.name}: {e}"
                    )

            logger.info(f"✅ Completed clearing scope: {self.bucket_name}.{scope_name}")

        except Exception as e:
            logger.warning(f"❌ Could not clear scope {self.bucket_name}.{scope_name}: {e}")

    def clear_collection(self, scope_name: str, collection_name: str):
        """Clear a specific collection in the specified scope."""
        try:
            if not self.bucket:
                # Ensure connection and bucket are ready
                if not self.cluster:
                    self.connect()
                if self.cluster:
                    self.bucket = self.cluster.bucket(self.bucket_name)

            if not self.bucket:
                logger.warning(f"Cannot clear collection - bucket not available")
                return

            logger.info(
                f"🗑️  Clearing collection: {self.bucket_name}.{scope_name}.{collection_name}"
            )

            # Clear the specific collection
            try:
                delete_query = (
                    f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
                )
                if self.cluster:
                    result = self.cluster.query(delete_query).execute()
                    logger.info(
                        f"✅ Cleared collection: {self.bucket_name}.{scope_name}.{collection_name}"
                    )
            except Exception as e:
                logger.info(
                    f"Collection {self.bucket_name}.{scope_name}.{collection_name} does not exist or is already empty: {e}"
                )

        except Exception as e:
            logger.exception(f"❌ Error clearing collection {scope_name}.{collection_name}: {e}")
            raise


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

        # Get prompt resource first - we'll need it for the ReAct agent
        prompt_resource = self.catalog.find("prompt", name="flight_search_assistant")

        # Get tools from Agent Catalog - primary method: find by name, fallback: find through prompt
        tools = []
        tool_names = [
            "lookup_flight_info",
            "save_flight_booking",
            "retrieve_flight_bookings",
            "search_airline_reviews",
        ]

        for tool_name in tool_names:
            catalog_tool = None
            try:
                # Primary method: Find tool by name using agentc.find("tool")
                catalog_tool = self.catalog.find("tool", name=tool_name)
                if catalog_tool:
                    logger.info(f"✅ Found tool by name: {tool_name}")
                else:
                    logger.warning(
                        f"⚠️  Tool not found by name: {tool_name}, trying prompt fallback"
                    )

            except Exception as e:
                logger.warning(f"❌ Failed to find tool by name {tool_name}: {e}")

            # If tool not found by name, try fallback through prompt
            if not catalog_tool:
                try:
                    logger.info(f"🔄 Trying prompt fallback for tool: {tool_name}")
                    # Use the prompt resource we already found and extract tools from it
                    if prompt_resource:
                        prompt_tools = getattr(prompt_resource, "tools", [])
                        for prompt_tool in prompt_tools:
                            # Check if this is the tool we're looking for
                            tool_meta_name = (
                                getattr(prompt_tool.meta, "name", "")
                                if hasattr(prompt_tool, "meta")
                                else ""
                            )
                            if tool_meta_name == tool_name:
                                catalog_tool = prompt_tool
                                logger.info(f"✅ Found tool through prompt: {tool_name}")
                                break

                    if not catalog_tool:
                        logger.error(f"❌ Tool {tool_name} not found by name or through prompt")
                        continue

                except Exception as e:
                    logger.error(f"❌ Prompt fallback failed for tool {tool_name}: {e}")
                    continue

            # Create wrapper function to handle proper parameter parsing
            def create_tool_wrapper(original_tool, name):
                def wrapper_func(tool_input: str) -> str:
                    """Wrapper to handle proper parameter parsing for each tool."""
                    try:
                        logger.info(f"🔧 Tool {name} called with input: '{tool_input}'")

                        # Parse input based on tool requirements
                        if name == "lookup_flight_info":
                            # Expected format: "JFK,LAX" - parse and pass as separate parameters
                            parts = tool_input.replace(" to ", ",").replace("from ", "").split(",")
                            if len(parts) >= 2:
                                source_airport = parts[0].strip()
                                destination_airport = parts[1].strip()
                                result = original_tool.func(
                                    source_airport=source_airport,
                                    destination_airport=destination_airport,
                                )
                            else:
                                return f"Error: lookup_flight_info requires format 'SOURCE,DESTINATION' (e.g., 'JFK,LAX')"

                        elif name == "save_flight_booking":
                            # Pass the full string directly - tool expects "source,dest,date" format
                            result = original_tool.func(booking_input=tool_input)

                        elif name == "retrieve_flight_bookings":
                            # Pass the full string directly - tool expects string input
                            result = original_tool.func(booking_query=tool_input)

                        elif name == "search_airline_reviews":
                            # Pass the full string directly - tool expects query string
                            result = original_tool.func(query=tool_input)

                        else:
                            # Generic fallback
                            result = original_tool.func(tool_input)

                        logger.info(f"✅ Tool {name} executed successfully")
                        return str(result)

                    except Exception as e:
                        error_msg = f"Error calling {name}: {e!s}"
                        logger.error(error_msg)
                        return error_msg

                return wrapper_func

            # Create LangChain tool with wrapper
            langchain_tool = Tool(
                name=tool_name,
                description=f"Tool for {tool_name.replace('_', ' ')}",
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

        # Create agent executor
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
        )

    def compile(self):
        """Compile the LangGraph workflow."""

        # Build the flight search agent with catalog integration
        search_agent = FlightSearchAgent(catalog=self.catalog, span=self.span)

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
        client = CouchbaseClient(
            conn_string=os.environ["CB_CONN_STRING"],
            username=os.environ["CB_USERNAME"],
            password=os.environ["CB_PASSWORD"],
            bucket_name=os.environ["CB_BUCKET"],
        )
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
            import sys

            sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
            from airline_reviews_data import _data_manager

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
                    client.clear_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                    logger.info(
                        f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                    )

            except Exception as count_error:
                # Collection doesn't exist or query failed - clear anyway to ensure fresh start
                logger.info(
                    f"📊 Collection doesn't exist or query failed, will clear and reload: {count_error}"
                )
                client.clear_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                logger.info(
                    f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                )

        except Exception as e:
            logger.warning(f"⚠️  Could not check collection count, clearing anyway: {e}")
            client.clear_collection(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
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
        application_span = catalog.Span(name="Flight Search Agent")

        # Create CouchbaseClient for all operations
        client = CouchbaseClient(
            conn_string=os.environ["CB_CONN_STRING"],
            username=os.environ["CB_USERNAME"],
            password=os.environ["CB_PASSWORD"],
            bucket_name=os.environ["CB_BUCKET"],
        )

        # Setup everything in one call - bucket, scope, collection
        client.setup_collection(
            scope_name=os.environ["CB_SCOPE"], collection_name=os.environ["CB_COLLECTION"]
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

        # Setup embeddings and vector store
        # Use Capella AI embeddings if available, fallback to OpenAI
        if (
            os.environ.get("CB_USERNAME")
            and os.environ.get("CB_PASSWORD")
            and os.environ.get("CAPELLA_API_ENDPOINT")
            and os.environ.get("CAPELLA_API_EMBEDDING_MODEL")
        ):
            logger.info("🔄 Using Capella AI embeddings for main application")
            import base64

            api_key = base64.b64encode(
                f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()
            ).decode()

            embeddings = OpenAIEmbeddings(
                model=os.environ["CAPELLA_API_EMBEDDING_MODEL"],
                api_key=api_key,
                base_url=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
            )
        else:
            logger.info(
                "🔄 Using OpenAI embeddings for main application (Capella AI not configured)"
            )
            embeddings = OpenAIEmbeddings(
                api_key=SecretStr(os.environ["OPENAI_API_KEY"]), model="text-embedding-3-small"
            )
        client.setup_vector_store(
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            index_name=os.environ["CB_INDEX"],
            embeddings=embeddings,
        )

        # Create the flight search graph
        flight_graph = FlightSearchGraph(catalog=catalog, span=application_span)
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
