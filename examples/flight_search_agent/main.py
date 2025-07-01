#!/usr/bin/env python3
"""
Flight Search Agent - Agent Catalog + LangGraph Implementation

A streamlined flight search agent demonstrating Agent Catalog integration
with LangGraph and Couchbase vector search for flight booking assistance.
"""

import json
import os
import time
import typing
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

# Load environment variables
dotenv.load_dotenv()


def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        import getpass

        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup required environment variables with defaults."""
    required_vars = ["OPENAI_API_KEY", "CB_HOST", "CB_USERNAME", "CB_PASSWORD", "CB_BUCKET_NAME"]
    for var in required_vars:
        _set_if_undefined(var)

    defaults = {
        "CB_HOST": "couchbase://localhost",
        "CB_USERNAME": "Administrator",
        "CB_PASSWORD": "password",
        "CB_BUCKET_NAME": "vector-search-testing",
        "INDEX_NAME": "vector_search_agentcatalog",
        "SCOPE_NAME": "shared",
        "COLLECTION_NAME": "agentcatalog",
    }

    for key, default_value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = input(f"Enter {key} (default: {default_value}): ") or default_value


def setup_couchbase_connection():
    """Setup Couchbase cluster connection."""
    try:
        auth = PasswordAuthenticator(os.environ["CB_USERNAME"], os.environ["CB_PASSWORD"])
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ["CB_HOST"], options)
        cluster.wait_until_ready(timedelta(seconds=10))
        print("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")


def setup_collection(cluster, bucket_name, scope_name, collection_name):
    """Setup Couchbase bucket, scope and collection."""
    try:
        try:
            bucket = cluster.bucket(bucket_name)
            print(f"Bucket '{bucket_name}' exists")
        except Exception:
            print(f"Creating bucket '{bucket_name}'...")
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
            print(f"Bucket '{bucket_name}' created successfully")

        bucket_manager = bucket.collections()

        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)

        if not scope_exists and scope_name != "_default":
            print(f"Creating scope '{scope_name}'...")
            bucket_manager.create_scope(scope_name)
            print(f"Scope '{scope_name}' created successfully")

        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            print(f"Creating collection '{collection_name}'...")
            bucket_manager.create_collection(scope_name, collection_name)
            print(f"Collection '{collection_name}' created successfully")

        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(3)

        try:
            cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            ).execute()
            print("Primary index created successfully")
        except Exception as e:
            print(f"Warning: Error creating primary index: {str(e)}")

        print(f"Collection setup complete")

        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")


def setup_vector_search_index(cluster, index_definition):
    """Setup vector search index for flight data."""
    try:
        scope_index_manager = (
            cluster.bucket(os.environ["CB_BUCKET_NAME"])
            .scope(os.environ["SCOPE_NAME"])
            .search_indexes()
        )

        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]

        if index_name not in [index.name for index in existing_indexes]:
            print(f"Creating vector search index '{index_name}'...")
            search_index = SearchIndex.from_json(index_definition)
            scope_index_manager.upsert_index(search_index)
            print(f"Vector search index '{index_name}' created successfully")
        else:
            print(f"Vector search index '{index_name}' already exists")
    except Exception as e:
        raise RuntimeError(f"Error setting up vector search index: {str(e)}")


def load_flight_data():
    """Load flight data from our enhanced flight_data.py file."""
    try:
        # Import flight data
        import sys

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
        raise ValueError(f"Error loading flight data: {str(e)}")


def setup_vector_store(cluster):
    """Setup vector store and load flight data."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"
        )

        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ["CB_BUCKET_NAME"],
            scope_name=os.environ["SCOPE_NAME"],
            collection_name=os.environ["COLLECTION_NAME"],
            embedding=embeddings,
            index_name=os.environ["INDEX_NAME"],
        )

        flight_data = load_flight_data()

        try:
            vector_store.add_texts(texts=flight_data, batch_size=10)
            print("Flight data loaded into vector store successfully")
        except Exception as e:
            print(
                f"Warning: Error loading flight data: {str(e)}. Vector store created but data not loaded."
            )

        return vector_store
    except Exception as e:
        raise ValueError(f"Error setting up vector store: {str(e)}")


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations."""

    customer_id: str
    query: str
    resolved: bool
    search_results: typing.List[typing.Dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and prompts."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the flight search agent."""

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

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
            print(f"🔍 Flight Query: {state['query']}")

        # Create the ReAct agent with tools from Agent Catalog
        agent = self.create_react_agent(span)

        # Run the agent with the current state
        response = agent.invoke(input=state, config=config)

        # Extract the assistant's response
        if response.get("messages"):
            assistant_message = response["messages"][-1]
            state["messages"].append(assistant_message)

            if hasattr(assistant_message, "content"):
                print(f"✈️ Response: {assistant_message.content}")

        # Check if search is complete
        if response.get("structured_response"):
            structured = response["structured_response"]
            state["resolved"] = structured.get("search_complete", True)

            # Store search results
            if "flight_results" in structured:
                state["search_results"] = structured["flight_results"]

        return state


class FlightSearchGraph(agentc_langgraph.graph.GraphRunnable):
    """Flight search conversation graph using Agent Catalog."""

    @staticmethod
    def build_starting_state(customer_id: str, query: str) -> FlightSearchState:
        """Build the initial state for the flight search."""
        return FlightSearchState(
            messages=[],
            customer_id=customer_id,
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

    print("\n🛫 Flight Search Agent - Agent Catalog Demo")
    print("=" * 50)

    try:
        # Setup environment
        setup_environment()

        # Setup Couchbase infrastructure
        cluster = setup_couchbase_connection()

        setup_collection(
            cluster,
            os.environ["CB_BUCKET_NAME"],
            os.environ["SCOPE_NAME"],
            os.environ["COLLECTION_NAME"],
        )

        try:
            with open("agentcatalog_index.json", "r") as file:
                index_definition = json.load(file)
            print("Loaded vector search index definition from agentcatalog_index.json")
        except Exception as e:
            print(f"Warning: Error loading index definition: {str(e)}")
            print("Continuing without vector search index...")

        if "index_definition" in locals():
            setup_vector_search_index(cluster, index_definition)

        setup_vector_store(cluster)

        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Flight Search Agent")

        # Create the flight search graph
        flight_graph = FlightSearchGraph(catalog=catalog, span=application_span)

        # Compile the graph
        compiled_graph = flight_graph.compile()

        print("✅ Agent Catalog integration successful")

        # Interactive flight search loop
        while True:
            print("\n" + "─" * 40)
            query = input("🔍 Enter flight search query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("✈️ Thanks for using Flight Search Agent!")
                break

            if not query:
                continue

            try:
                # Build starting state
                state = FlightSearchGraph.build_starting_state(customer_id="demo_user", query=query)

                # Run the flight search
                result = compiled_graph.invoke(state)

                # Display results summary
                if result.get("search_results"):
                    print(f"\n📋 Found {len(result['search_results'])} flight options")

                print(f"✅ Search completed: {result.get('resolved', False)}")

            except Exception as e:
                print(f"❌ Search error: {e}")

    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("💡 Ensure Agent Catalog is published: agentc index . && agentc publish")


if __name__ == "__main__":
    run_flight_search_demo()

    # Uncomment the following lines to visualize the LangGraph workflow:
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="flight_search_graph.png")
    # compiled_graph.get_graph().draw_ascii()
