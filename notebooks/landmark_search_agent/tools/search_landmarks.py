"""
Landmark Search Tool - AgentC Integration for Travel Sample Landmarks

This tool demonstrates:
- Using AgentC for tool registration (@agentc.catalog.tool)
- Semantic search using the vector store for landmark data from travel-sample.inventory.landmark
- Integration with Couchbase travel-sample database
"""

import os
import logging
import dotenv
from datetime import timedelta

import agentc
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        auth = PasswordAuthenticator(
            username=os.environ.get("CB_USERNAME", "kaustavcluster"),
            password=os.environ.get("CB_PASSWORD", "Password@123"),
        )
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")

        # Set timeout configurations
        timeout_options = ClusterTimeoutOptions(
            kv_timeout=timedelta(seconds=10),
            kv_durable_timeout=timedelta(seconds=15),
            query_timeout=timedelta(seconds=30),
            search_timeout=timedelta(seconds=30),
            management_timeout=timedelta(seconds=30),
            bootstrap_timeout=timedelta(seconds=20),
        )
        options.timeout_options = timeout_options

        conn_string = os.environ.get(
            "CB_CONN_STRING", "couchbases://cb.hlcup4o4jmjr55yf.cloud.couchbase.com"
        )
        cluster = Cluster(conn_string, options)
        cluster.wait_until_ready(timedelta(seconds=20))

        return cluster

    except Exception as e:
        logger.error(f"Failed to connect to Couchbase: {e}")
        raise


@agentc.catalog.tool()
def search_landmarks(query: str, limit: int = 5) -> str:
    """
    Search for landmark information using semantic vector search.

    This tool searches through landmark data from the travel-sample database,
    including attractions, monuments, museums, and points of interest.

    Args:
        query (str): The search query describing what landmarks to find
        limit (int): Maximum number of results to return (default: 5)

    Returns:
        str: Formatted landmark search results with details like name, location,
             description, activities, and practical information
    """
    try:
        # Get cluster connection
        cluster = get_cluster_connection()

        # Create vector store using the landmark collection
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get("CB_BUCKET", "travel-sample"),
            scope_name=os.environ.get("CB_SCOPE", "inventory"),
            collection_name=os.environ.get("CB_COLLECTION", "landmark"),
            index_name=os.environ.get("CB_INDEX", "landmark_vector_index"),
        )

        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=Settings.embed_model
        )

        # Perform semantic search
        query_engine = index.as_query_engine(
            similarity_top_k=limit,
            response_mode="no_text",  # We just want the source documents
        )

        response = query_engine.query(query)

        # Format results
        if not response.source_nodes:
            return f"No landmarks found matching your query: '{query}'"

        results = []
        for i, node in enumerate(response.source_nodes, 1):
            landmark_data = node.metadata
            content = node.text

            # Extract key information from the landmark data
            name = landmark_data.get("name", "Unknown")
            title = landmark_data.get("title", name)
            city = landmark_data.get("city", "Unknown")
            country = landmark_data.get("country", "Unknown")
            address = landmark_data.get("address", "Address not available")
            activity = landmark_data.get("activity", "General")

            # Format the result
            result = f"{i}. **{title}** ({name})\n"
            result += f"   📍 Location: {city}, {country}\n"
            result += f"   🏠 Address: {address}\n"
            result += f"   🎯 Activity: {activity.title()}\n"

            # Add additional details if available
            if landmark_data.get("phone"):
                result += f"   📞 Phone: {landmark_data['phone']}\n"
            if landmark_data.get("url"):
                result += f"   🌐 Website: {landmark_data['url']}\n"
            if landmark_data.get("hours"):
                result += f"   🕐 Hours: {landmark_data['hours']}\n"
            if landmark_data.get("price"):
                result += f"   💰 Price: {landmark_data['price']}\n"

            # Add description/content
            if content:
                result += (
                    f"   📝 Description: {content[:200]}{'...' if len(content) > 200 else ''}\n"
                )

            results.append(result)

        return f"Found {len(results)} landmarks matching '{query}':\n\n" + "\n".join(results)

    except Exception as e:
        logger.error(f"Error searching landmarks: {e}")
        return f"Error searching landmarks: {str(e)}"
