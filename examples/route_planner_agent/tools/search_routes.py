"""
Route Search Tool - Simplified for AgentC Integration

This tool demonstrates:
- Using AgentC for tool registration (@agentc.catalog.tool)
- Simple tool function that can be discovered by AgentC
- Semantic search using the vector store set up in main.py
"""

import os
import logging

import agentc
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

# Setup logging
logger = logging.getLogger(__name__)

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
        # Import here to avoid circular imports
        from couchbase.auth import PasswordAuthenticator
        from couchbase.cluster import Cluster
        from couchbase.options import ClusterOptions
        from datetime import timedelta
        
        # Get configuration from environment (set by main.py)
        cluster_config = {
            'host': os.environ.get('CB_HOST', 'couchbase://localhost'),
            'username': os.environ.get('CB_USERNAME', 'Administrator'),
            'password': os.environ.get('CB_PASSWORD', 'password'),
            'bucket': os.environ.get('CB_BUCKET_NAME', 'route_planner'),
            'scope': os.environ.get('SCOPE_NAME', 'shared'),
            'collection': os.environ.get('COLLECTION_NAME', 'routes'),
            'index': os.environ.get('INDEX_NAME', 'route_search_index')
        }
        
        # Connect to Couchbase (quick connection for tool usage)
        auth = PasswordAuthenticator(cluster_config['username'], cluster_config['password'])
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")
        cluster = Cluster(cluster_config['host'], options)
        cluster.wait_until_ready(timedelta(seconds=10))
        
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


