#!/usr/bin/env python3
"""
Route Planner Agent - Simplified Implementation

A basic route planning agent demonstrating Agent Catalog tools
for intelligent travel planning.
"""

import json
import logging
import os
import sys
import time
from datetime import timedelta
from typing import Dict, List, Any, Optional

import agentc
import dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from couchbase.exceptions import CouchbaseException

from llama_index.core import Settings, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(override=True)


class RouteDataLoader:
    """Load route data for vector search."""

    def __init__(self):
        self.data_path = "data/route_data.py"

    def load_route_data(self) -> List[str]:
        """Load route data from the data file."""
        try:
            # Import the route data
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, "data")
            sys.path.insert(0, data_dir)
            from route_data import get_travel_knowledge_base

            travel_data = get_travel_knowledge_base()
            route_texts = []

            for item in travel_data:
                # Create a comprehensive text description for each route/POI
                text = f"{item['title']}. {item['content']}"

                # Add metadata information
                metadata = item.get("metadata", {})
                if metadata.get("distance"):
                    text += f" Distance: {metadata['distance']}."
                if metadata.get("duration"):
                    text += f" Duration: {metadata['duration']}."
                if metadata.get("cities"):
                    text += f" Cities: {', '.join(metadata['cities'])}."
                if metadata.get("transport_mode"):
                    text += f" Transportation: {metadata['transport_mode']}."
                if metadata.get("difficulty"):
                    text += f" Difficulty: {metadata['difficulty']}."
                if metadata.get("region"):
                    text += f" Region: {metadata['region']}."

                route_texts.append(text.strip())

            logger.info(f"Loaded {len(route_texts)} route descriptions")
            return route_texts

        except Exception as e:
            logger.error(f"Error loading route data: {e}")
            return []


class CouchbaseSetup:
    """Handle Couchbase cluster setup and configuration."""

    def __init__(self):
        self.cluster = None
        self.collection = None

    def setup_environment(self):
        """Setup required environment variables."""
        required_vars = [
            "OPENAI_API_KEY",
            "CB_HOST",
            "CB_USERNAME",
            "CB_PASSWORD",
            "CB_BUCKET_NAME",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

        # Set defaults for optional variables
        if not os.getenv("INDEX_NAME"):
            os.environ["INDEX_NAME"] = "vector_search_agentcatalog"
        if not os.getenv("SCOPE_NAME"):
            os.environ["SCOPE_NAME"] = "shared"
        if not os.getenv("COLLECTION_NAME"):
            os.environ["COLLECTION_NAME"] = "agentcatalog"

        logger.info(f"Using Couchbase connection: {os.getenv('CB_HOST')}")
        logger.info(f"Using bucket: {os.getenv('CB_BUCKET_NAME')}")
        logger.info(f"Using scope: {os.getenv('SCOPE_NAME')}")
        logger.info(f"Using collection: {os.getenv('COLLECTION_NAME')}")
        logger.info(f"Using index: {os.getenv('INDEX_NAME')}")

    def setup_couchbase_connection(self):
        """Setup Couchbase cluster connection."""
        try:
            auth = PasswordAuthenticator(os.environ["CB_USERNAME"], os.environ["CB_PASSWORD"])
            options = ClusterOptions(auth)
            self.cluster = Cluster(os.environ["CB_HOST"], options)
            self.cluster.wait_until_ready(timedelta(seconds=10))
            logger.info("Successfully connected to Couchbase cluster")
            return self.cluster
        except CouchbaseException as e:
            raise ConnectionError(f"Failed to connect to Couchbase: {e}")

    def setup_bucket_scope_collection(self):
        """Setup bucket, scope, and collection."""
        try:
            bucket_name = os.environ["CB_BUCKET_NAME"]
            scope_name = os.environ["SCOPE_NAME"]
            collection_name = os.environ["COLLECTION_NAME"]

            # Check bucket
            try:
                bucket = self.cluster.bucket(bucket_name)
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
                self.cluster.buckets().create_bucket(bucket_settings)
                time.sleep(5)
                bucket = self.cluster.bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' created successfully")

            # Setup scope and collection
            bucket_manager = bucket.collections()

            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                logger.info(f"Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                logger.info(f"Scope '{scope_name}' created successfully")

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

            self.collection = bucket.scope(scope_name).collection(collection_name)
            time.sleep(3)

            # Create primary index
            try:
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
                ).execute()
                logger.info("Primary index created successfully")
            except Exception as e:
                logger.warning(f"Error creating primary index: {e}")

            return self.collection

        except Exception as e:
            raise RuntimeError(f"Error setting up bucket/scope/collection: {e}")

    def setup_vector_search_index(self):
        """Setup vector search index."""
        try:
            # Load index definition from agentcatalog_index.json
            index_file = "agentcatalog_index.json"
            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    index_definition = json.load(f)

                scope_index_manager = (
                    self.cluster.bucket(os.environ["CB_BUCKET_NAME"])
                    .scope(os.environ["SCOPE_NAME"])
                    .search_indexes()
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
            else:
                logger.warning(f"Index definition file {index_file} not found")

        except Exception as e:
            logger.error(f"Error setting up vector search index: {e}")


class RouteplannerAgent:
    """Route planner agent using Agent Catalog tools and Couchbase vector store."""

    def __init__(self, span: agentc.Span):
        """Initialize the route planner agent."""
        self.catalog = None
        self.vector_store = None
        self.couchbase_setup = None
        self.data_loader = None
        self.application_span = span
        self.setup()

    def setup(self):
        """Setup the route planner agent."""
        try:
            with self.application_span.new("Environment Setup"):
                # Setup environment
                self.couchbase_setup = CouchbaseSetup()
                self.couchbase_setup.setup_environment()

            with self.application_span.new("Couchbase Connection"):
                # Setup Couchbase
                cluster = self.couchbase_setup.setup_couchbase_connection()
                collection = self.couchbase_setup.setup_bucket_scope_collection()
                self.couchbase_setup.setup_vector_search_index()

            with self.application_span.new("LLM and Embeddings Setup"):
                # Setup LLM and embeddings with OpenAI
                Settings.llm = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o", temperature=0.1
                )

                embed_model = OpenAIEmbedding(
                    api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small"
                )
                Settings.embed_model = embed_model

            with self.application_span.new("Vector Store Setup"):
                # Setup vector store (LlamaIndex pattern - no embedding parameter)
                self.vector_store = CouchbaseSearchVectorStore(
                    cluster=cluster,
                    bucket_name=os.environ["CB_BUCKET_NAME"],
                    scope_name=os.environ["SCOPE_NAME"],
                    collection_name=os.environ["COLLECTION_NAME"],
                    index_name=os.environ["INDEX_NAME"],
                    # Note: LlamaIndex uses Settings.embed_model globally, not embedding_key
                )

            with self.application_span.new("Data Ingestion"):
                # Load and ingest route data
                self.data_loader = RouteDataLoader()
                self.ingest_route_data()

            with self.application_span.new("Agent Catalog Setup"):
                # Setup Agent Catalog
                self.catalog = agentc.Catalog()
            logger.info("Route planner setup complete")

        except Exception as e:
            logger.error(f"Error setting up route planner: {e}")
            raise

    def ingest_route_data(self):
        """Ingest route data into the vector store."""
        try:
            route_texts = self.data_loader.load_route_data()

            if not route_texts:
                logger.warning("No route data loaded")
                return

            # Check if data already exists by trying to query
            try:
                # Try a simple search to see if data exists
                from llama_index.core.vector_stores import VectorStoreQuery

                embed_model = Settings.embed_model
                query_embedding = embed_model.get_query_embedding("route")

                query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=1)
                search_results = self.vector_store.query(query_obj)
                if search_results and len(search_results.nodes) > 0:
                    logger.info("Route data already exists in vector store")
                    return
            except Exception:
                # If search fails, assume no data exists
                pass

            logger.info("Ingesting route data into vector store...")

            # Create documents
            documents = [Document(text=text) for text in route_texts]

            # Setup ingestion pipeline
            pipeline = IngestionPipeline(
                transformations=[
                    SentenceSplitter(chunk_size=512, chunk_overlap=50),
                    Settings.embed_model,
                ],
                vector_store=self.vector_store,
            )

            # Ingest documents
            pipeline.run(documents=documents)
            logger.info(f"Successfully ingested {len(documents)} route documents")

        except Exception as e:
            logger.error(f"Error ingesting route data: {e}")

    def search_routes_with_vector_store(self, query: str) -> str:
        """Search routes using vector store."""
        try:
            if not self.vector_store:
                return "Vector store not initialized"

            from llama_index.core.vector_stores import VectorStoreQuery

            # Generate query embedding
            embed_model = Settings.embed_model
            query_embedding = embed_model.get_query_embedding(query)

            query_obj = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=3)
            search_results = self.vector_store.query(query_obj)

            if not search_results or not search_results.nodes:
                return "No routes found matching your query"

            response = f"Found {len(search_results.nodes)} relevant routes:\n\n"
            for i, node in enumerate(search_results.nodes, 1):
                response += f"{i}. {node.text}\n"
                if hasattr(node, "score"):
                    response += f"   (Relevance score: {node.score:.3f})\n"
                response += "\n"

            return response

        except Exception as e:
            return f"Error searching routes with vector store: {e}"

    def plan_route(self, query: str) -> str:
        """Plan a route based on user query."""
        try:
            if not self.catalog:
                return "Route planner not properly initialized. Please check your configuration."

            # First try to search using vector store
            vector_result = self.search_routes_with_vector_store(query)

            # Then try to use the search_routes tool from Agent Catalog
            try:
                tool_obj = self.catalog.find("tool", name="search_routes")
                if tool_obj and hasattr(tool_obj, "func"):
                    tool_result = tool_obj.func(query=query)
                    combined_result = f"Vector Search Results:\n{vector_result}\n\nTool Search Results:\n{tool_result}"
                    return combined_result
                else:
                    return vector_result
            except Exception as e:
                logger.warning(f"Error using search_routes tool: {e}")
                return vector_result

        except Exception as e:
            return f"Error planning route: {e!s}"

    def calculate_distance(self, origin: str, destination: str) -> str:
        """Calculate distance between two locations."""
        try:
            if not self.catalog:
                return "Route planner not properly initialized. Please check your configuration."

            # Try to find and use the calculate_distance tool
            try:
                tool_obj = self.catalog.find("tool", name="calculate_distance")
                if tool_obj and hasattr(tool_obj, "func"):
                    result = tool_obj.func(origin=origin, destination=destination)
                    return str(result)
                else:
                    # Use find_tools with proper parameters to avoid None name issue
                    try:
                        available_tools = list(self.catalog.find_tools(query="", limit=20))
                        tool_names = []
                        for t in available_tools:
                            try:
                                if hasattr(t, "meta") and t.meta and hasattr(t.meta, "name"):
                                    tool_names.append(t.meta.name)
                                elif hasattr(t, "__name__"):
                                    tool_names.append(t.__name__)
                                else:
                                    tool_names.append(str(t))
                            except Exception:
                                tool_names.append("<unknown_tool>")
                        return f"calculate_distance tool not found. Available tools: {tool_names}"
                    except Exception as list_error:
                        return (
                            f"calculate_distance tool not found. Error listing tools: {list_error}"
                        )
            except Exception as e:
                return f"Error calculating distance: {e!s}"

        except Exception as e:
            return f"Error calculating distance: {e!s}"

    def list_available_tools(self) -> str:
        """List available tools in the catalog."""
        try:
            if not self.catalog:
                return "Catalog not initialized"

            # Use find_tools with empty query and reasonable limit to get all tools
            # This avoids the None name issue that caused embedding problems
            try:
                tools = list(self.catalog.find_tools(query="", limit=20))
                if not tools:
                    return "No tools found in catalog"

                tool_names = []
                for tool in tools:
                    try:
                        if hasattr(tool, "meta") and tool.meta and hasattr(tool.meta, "name"):
                            tool_names.append(tool.meta.name)
                        elif hasattr(tool, "__name__"):
                            tool_names.append(tool.__name__)
                        else:
                            tool_names.append(str(tool))
                    except Exception as e:
                        tool_names.append(f"<unknown_tool: {e}>")
                        continue

                return f"Available tools ({len(tool_names)}): {', '.join(tool_names)}"

            except Exception as e:
                logger.error(f"Error calling find_tools: {e}")

                # Fallback: try to access tools directly from the local catalog
                try:
                    if hasattr(self.catalog, "_tool_provider") and hasattr(
                        self.catalog._tool_provider, "catalog"
                    ):
                        tool_catalog = self.catalog._tool_provider.catalog
                        if hasattr(tool_catalog, "_tools"):
                            tools = tool_catalog._tools
                            tool_names = []
                            for tool_id, tool_data in tools.items():
                                try:
                                    if isinstance(tool_data, dict):
                                        name = (
                                            tool_data.get("name")
                                            or tool_data.get("meta", {}).get("name")
                                            or tool_id
                                        )
                                    else:
                                        name = getattr(tool_data, "name", str(tool_data))
                                    tool_names.append(str(name))
                                except Exception:
                                    tool_names.append(f"<tool_{tool_id}>")

                            if tool_names:
                                return (
                                    f"Available tools ({len(tool_names)}): {', '.join(tool_names)}"
                                )

                    # Final fallback: scan local tools directory
                    tool_names = []
                    import os

                    tools_dir = os.path.join(os.path.dirname(__file__), "tools")
                    if os.path.exists(tools_dir):
                        for filename in os.listdir(tools_dir):
                            if filename.endswith(".py") and not filename.startswith("__"):
                                tool_name = filename[:-3]  # Remove .py extension
                                tool_names.append(tool_name)

                    if tool_names:
                        return f"Local tools found ({len(tool_names)}): {', '.join(tool_names)}"

                    return f"Error accessing tools: {e}"

                except Exception as inner_e:
                    logger.error(f"Fallback method also failed: {inner_e}")
                    return f"Error accessing tools: {e}"

        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return f"Error listing tools: {e}"


def run_interactive_demo():
    """Run interactive demo of the route planner."""
    logger.info("Starting Route Planner Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Route Planner Agent")
        planner = RouteplannerAgent(span=application_span)
        logger.info("Route planner initialized successfully!")

        logger.info(planner.list_available_tools())

        logger.info("Available commands:")
        logger.info("- 'plan <query>' - Plan a route (e.g., 'plan route from New York to Boston')")
        logger.info("- 'distance <origin> to <destination>' - Calculate distance")
        logger.info("- 'tools' - List available tools")
        logger.info("- 'quit' - Exit the demo")
        logger.info(
            "Try asking: 'plan scenic route in California' or 'distance San Francisco to Los Angeles'"
        )
        logger.info("-" * 50)

        with application_span.new("Interactive Demo") as span:
            while True:
                user_input = input("\nEnter your request: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    logger.info("Thanks for using the Route Planner!")
                    break

                with span.new(f"Request: {user_input}") as request_span:
                    if user_input.lower() == "tools":
                        logger.info(planner.list_available_tools())
                        continue

                    if user_input.lower().startswith("plan "):
                        query = user_input[5:]  # Remove 'plan ' prefix
                        result = planner.plan_route(query)
                        request_span["result"] = result
                        logger.info(f"Route Planning Results:\n{result}")

                    elif " to " in user_input.lower() and any(
                        word in user_input.lower() for word in ["distance", "from"]
                    ):
                        # Parse distance query
                        parts = (
                            user_input.lower()
                            .replace("distance", "")
                            .replace("from", "")
                            .strip()
                            .split(" to ")
                        )
                        if len(parts) == 2:
                            origin = parts[0].strip()
                            destination = parts[1].strip()
                            result = planner.calculate_distance(origin, destination)
                            request_span["result"] = result
                            logger.info(f"Distance Calculation Results:\n{result}")
                        else:
                            logger.info("Please use format: 'distance <origin> to <destination>'")

                    else:
                        # Default to route planning
                        result = planner.plan_route(user_input)
                        request_span["result"] = result
                        logger.info(f"Route Planning Results:\n{result}")

    except Exception as e:
        logger.error(f"Error in interactive demo: {e}")


def run_test():
    """Run a quick test of the route planner."""
    logger.info("Running Route Planner Test")
    logger.info("=" * 30)

    try:
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Route Planner Agent Test")
        planner = RouteplannerAgent(span=application_span)

        with application_span.new("Distance Calculation Test") as span:
            # Test distance calculation
            logger.info("Testing distance calculation...")
            result = planner.calculate_distance("San Francisco", "Los Angeles")
            span["result"] = result
            logger.info(f"Distance Test Result: {result}")

        with application_span.new("Route Planning Test") as span:
            # Test route planning
            logger.info("Testing route planning...")
            result = planner.plan_route("scenic route in California")
            span["result"] = result
            logger.info(f"Route Planning Test Result: {result}")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_test()
    else:
        run_interactive_demo()
