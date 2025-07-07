"""
Route Search Tool for Agent Catalog

Performs semantic search for route information using Couchbase vector store.
"""

import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

dotenv.load_dotenv(override=True)

# Global Couchbase connection
cluster = None
try:
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        couchbase.options.ClusterOptions(
            authenticator=couchbase.auth.PasswordAuthenticator(
                username=os.getenv("CB_USERNAME", "Administrator"),
                password=os.getenv("CB_PASSWORD", "password"),
            )
        ),
    )
    cluster.wait_until_ready(timedelta(seconds=5))
    print("Successfully connected to Couchbase cluster")
except couchbase.exceptions.CouchbaseException as e:
    print(f"Could not connect to Couchbase cluster: {e}")
    cluster = None
except Exception as e:
    print(f"Unexpected error connecting to Couchbase: {e}")
    cluster = None


def _get_vector_store():
    """Get vector store instance for route searching."""
    try:
        if cluster is None:
            raise RuntimeError(
                "Couchbase cluster is not connected. Please check your connection settings and credentials."
            )

        # Create embedding model for Capella AI
        import base64

        capella_ai_key = base64.b64encode(
            f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
        ).decode("utf-8")

        embed_model = OpenAIEmbedding(
            api_key=capella_ai_key,
            api_base=os.getenv("CAPELLA_AI_ENDPOINT"),
            model_name="intfloat/e5-mistral-7b-instruct",
            embed_batch_size=30,
        )

        # Configure LlamaIndex to use this embedding model globally
        Settings.embed_model = embed_model

        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=os.getenv("CB_BUCKET", "route-planner-bucket"),
            scope_name=os.getenv("SCOPE_NAME", "shared"),
            collection_name=os.getenv("COLLECTION_NAME", "route_data"),
            index_name=os.getenv("INDEX_NAME", "route_planner_vector_index"),
            # Note: LlamaIndex uses Settings.embed_model, not embedding parameter
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}")


@agentc.catalog.tool
def search_routes(
    query: str,
    route_type: str = None,
    region: str = None,
    transport_mode: str = None,
    max_results: int = 5,
) -> str:
    """
    Search for route information using semantic search.

    Args:
        query: Natural language description of the route or travel need
        route_type: Optional filter for route type (scenic_drive, urban_corridor, etc.)
        region: Optional filter for geographic region
        transport_mode: Optional filter for transportation mode
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Formatted string with route search results and recommendations
    """
    try:
        vector_store = _get_vector_store()

        # Perform semantic search
        search_results = vector_store.query(query_str=query, similarity_top_k=max_results)

        if not search_results.nodes:
            return f"No routes found matching '{query}'. Try a different search term or broader criteria."

        # Format results
        formatted_results = []
        for i, node in enumerate(search_results.nodes, 1):
            content = node.text
            metadata = node.metadata

            # Apply filters if specified
            if route_type and metadata.get("route_type") != route_type:
                continue
            if region and region.lower() not in metadata.get("region", "").lower():
                continue
            if transport_mode and transport_mode not in metadata.get("transport_mode", ""):
                continue

            result = f"\n**Route {i}: {metadata.get('title', 'Route Information')}**\n"
            result += f"📍 Region: {metadata.get('region', 'Not specified')}\n"
            result += f"🚗 Transport: {metadata.get('transport_mode', 'Various')}\n"
            result += f"⏱️ Duration: {metadata.get('duration', 'Variable')}\n"

            if metadata.get("distance"):
                result += f"📏 Distance: {metadata['distance']}\n"
            if metadata.get("difficulty"):
                result += f"💪 Difficulty: {metadata['difficulty']}\n"

            result += f"\n{content[:300]}{'...' if len(content) > 300 else ''}\n"
            result += "─" * 50

            formatted_results.append(result)

        if not formatted_results:
            return f"No routes found matching the specified filters for '{query}'."

        header = f"🗺️ **Route Search Results for:** '{query}'\n"
        header += f"Found {len(formatted_results)} relevant routes:\n"
        header += "=" * 60

        return header + "\n".join(formatted_results)

    except Exception as e:
        return f"Error searching routes: {e!s}. Please check your connection and try again."


@agentc.catalog.tool
def search_routes_by_cities(origin: str, destination: str, preferences: str = "") -> str:
    """
    Search for specific routes between two cities with optional preferences.

    Args:
        origin: Starting city or location
        destination: Ending city or location
        preferences: Optional preferences like "scenic", "fastest", "cheapest", etc.

    Returns:
        Formatted route options between the specified cities
    """
    try:
        # Construct search query
        search_query = f"route from {origin} to {destination}"
        if preferences:
            search_query += f" {preferences}"

        # Use the main search function
        results = search_routes(query=search_query, max_results=3)

        # Add city-specific header
        header = f"🏙️ **Routes from {origin} to {destination}**\n"
        if preferences:
            header += f"Preferences: {preferences}\n"
        header += "=" * 50 + "\n"

        return header + results

    except Exception as e:
        return f"Error finding routes between {origin} and {destination}: {e!s}"


@agentc.catalog.tool
def search_scenic_routes(region: str = "", activity: str = "") -> str:
    """
    Search for scenic and recreational routes.

    Args:
        region: Geographic region to search in (optional)
        activity: Type of activity like "hiking", "driving", "cycling" (optional)

    Returns:
        Scenic route recommendations with descriptions
    """
    try:
        query = "scenic beautiful route drive"
        if activity:
            query += f" {activity}"
        if region:
            query += f" {region}"

        results = search_routes(
            query=query, route_type="scenic_drive", region=region if region else None, max_results=4
        )

        header = "🌄 **Scenic Route Recommendations**\n"
        if region:
            header += f"Region: {region}\n"
        if activity:
            header += f"Activity: {activity}\n"
        header += "=" * 50 + "\n"

        return header + results

    except Exception as e:
        return f"Error finding scenic routes: {e!s}"
