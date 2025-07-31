import os
import sys
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

# Import shared AI services module - robust path handling
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Go up 2 levels to reach shared/
from shared.agent_setup import setup_ai_services

from langchain_couchbase.vectorstores import CouchbaseVectorStore

dotenv.load_dotenv()

def setup_embeddings_service_for_tool():
    """Setup embeddings service using shared 4-case priority ladder."""
    embeddings, _ = setup_ai_services(framework="langchain")
    return embeddings


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        auth = couchbase.auth.PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
        )
        options = couchbase.options.ClusterOptions(authenticator=auth)
        # Use WAN profile for better timeout handling with remote clusters
        options.apply_profile("wan_development")

        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"), options
        )
        cluster.wait_until_ready(timedelta(seconds=15))
        return cluster
    except couchbase.exceptions.CouchbaseException as e:
        print(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


@agentc.catalog.tool
def search_vector_database(query: str) -> str:
    """
    Search for hotels using semantic similarity. Returns raw hotel information for agent processing.

    Args:
        query: Search query for hotels (location, amenities, etc.)

    Returns:
        Hotel search results or error message
    """
    try:
        # Get cluster connection
        cluster = get_cluster_connection()
        if not cluster:
            return "ERROR: Could not connect to database"

        # Setup embeddings with priority order
        embeddings = setup_embeddings_service_for_tool()
        if not embeddings:
            return "ERROR: No embeddings service available"

        # Setup vector store
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name="travel-sample",
            scope_name="agentc_data",
            collection_name="hotel_data",
            embedding=embeddings,
            index_name="hotel_data_index",
        )

        # Perform similarity search
        try:
            search_results = vector_store.similarity_search_with_score(query, k=10)
        except Exception as search_error:
            return f"ERROR: Search failed - {str(search_error)}"

        if not search_results:
            return f"NO_RESULTS: No hotels found for '{query}'"

        # Simple deduplication based on content similarity
        unique_results = []
        seen_content = set()

        for doc, score in search_results:
            # Use first 60 characters as deduplication key
            content_key = doc.page_content[:60].strip()

            if content_key not in seen_content:
                unique_results.append((doc, score))
                seen_content.add(content_key)

            # Limit to top 6 unique results
            if len(unique_results) >= 6:
                break

        # Format results as simple list
        results = []
        for i, (doc, score) in enumerate(unique_results, 1):
            results.append(f"HOTEL_{i}: {doc.page_content} (Score: {score:.3f})")

        return f"FOUND_{len(unique_results)}_HOTELS:\n" + "\n\n".join(results)

    except Exception as e:
        return f"ERROR: Unexpected error - {str(e)}"
