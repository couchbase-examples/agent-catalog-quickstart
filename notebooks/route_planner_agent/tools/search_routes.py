"""
Route Search Tool - Simplified for AgentC Integration

This tool demonstrates:
- Using AgentC for tool registration (@agentc.catalog.tool)
- Simple tool function that can be discovered by AgentC
- Semantic search using the vector store set up in main.py
"""

import os
import logging
import dotenv
from datetime import timedelta

import agentc
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        from couchbase.auth import PasswordAuthenticator
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions, ClusterTimeoutOptions
        
        auth = PasswordAuthenticator(
            username=os.environ.get('CB_USERNAME', 'kaustavcluster'),
            password=os.environ.get('CB_PASSWORD', 'Password@123')
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
        
        cluster = Cluster(
            os.environ.get('CB_CONN_STRING', 'couchbases://cb.hlcup4o4jmjr55yf.cloud.couchbase.com'),
            options
        )
        cluster.wait_until_ready(timedelta(seconds=10))
        return cluster
    except Exception as e:
        logger.error(f"Failed to connect to Couchbase: {e}")
        raise


@agentc.catalog.tool
def search_routes(query: str) -> str:
    """
    Search for route information using semantic vector search.
    
    This tool performs semantic search against the route database to find
    relevant travel routes, scenic drives, and transportation information.
    
    Args:
        query: Natural language query about routes (e.g., "routes from New York to Boston", "scenic drives in California")
        
    Returns:
        Formatted string with relevant route information and travel recommendations
    """
    try:
        # Get configuration from environment
        cluster_config = {
            'bucket': os.environ.get('CB_BUCKET', 'vector-search-testing'),
            'scope': os.environ.get('CB_SCOPE', 'agentc_data'),
            'collection': os.environ.get('CB_COLLECTION', 'route_data'),
            'index': os.environ.get('CB_INDEX', 'route_data_index')
        }
        
        # Get fresh cluster connection
        cluster = get_cluster_connection()
        
        # Create vector store
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=cluster_config['bucket'],
            scope_name=cluster_config['scope'],
            collection_name=cluster_config['collection'],
            index_name=cluster_config['index']
        )
        
        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=3,
            response_mode="compact"
        )
        
        # Perform semantic search
        logger.info(f"🔍 Searching routes for: {query}")
        response = query_engine.query(query)
        
        # Format response
        result = f"🗺️ **Route Search Results**\n"
        result += "=" * 50 + "\n"
        result += f"**Query:** {query}\n\n"
        result += f"**Results:**\n{response}\n\n"
        result += "💡 **Tip:** Use calculate_distance for specific distance calculations between cities."
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error searching routes: {str(e)}"
        logger.error(error_msg)
        return error_msg


