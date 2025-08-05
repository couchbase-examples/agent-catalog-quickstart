import logging
import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

# Simple import - agent_setup should be accessible via shared module
try:
    from shared.agent_setup import setup_ai_services
except ImportError:
    # Fallback: Add parent directories to path to find shared module
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dirs = [os.path.join(current_dir, '..', '..', '..'), os.path.join(current_dir, '..', '..')]
    for parent_dir in parent_dirs:
        if parent_dir not in sys.path and os.path.exists(os.path.join(parent_dir, 'shared')):
            sys.path.insert(0, parent_dir)
            break
    from shared.agent_setup import setup_ai_services

from langchain_couchbase.vectorstores import CouchbaseVectorStore

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
try:
    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password"),
    )
    options = couchbase.options.ClusterOptions(auth)

    # Use WAN profile for better timeout handling with remote clusters
    options.apply_profile("wan_development")

    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"), options
    )
    cluster.wait_until_ready(timedelta(seconds=20))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"


def create_vector_store():
    """Create vector store instance for searching airline reviews."""
    try:
        # Setup embeddings using shared module
        embeddings, _ = setup_ai_services(framework="langgraph")

        # Create vector store
        return CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.getenv("CB_BUCKET", "travel-sample"),
            scope_name=os.getenv("CB_SCOPE", "agentc_data"),
            collection_name=os.getenv("CB_COLLECTION", "airline_reviews"),
            embedding=embeddings,
            index_name=os.getenv("CB_INDEX", "airline_reviews_index"),
        )
        
    except Exception as e:
        msg = f"Failed to create vector store: {e}"
        logger.error(msg)
        raise RuntimeError(msg)


def format_review_results(results: list, query: str) -> str:
    """Format search results for display."""
    if not results:
        return "No relevant airline reviews found for your query. Please try different search terms like 'food', 'seats', 'service', 'delays', 'check-in', or 'baggage'."

    formatted_results = []
    for i, doc in enumerate(results, 1):
        content = doc.page_content
        
        # Simple metadata formatting
        metadata_info = ""
        if hasattr(doc, "metadata") and doc.metadata:
            meta_parts = []
            for key in ["airline", "rating", "route", "seat_type"]:
                if key in doc.metadata:
                    value = doc.metadata[key]
                    if key == "rating":
                        meta_parts.append(f"Rating: {value}/5")
                    else:
                        meta_parts.append(f"{key.title()}: {value}")
            
            if meta_parts:
                metadata_info = f"[{' | '.join(meta_parts)}]\n"

        # Limit content length for readability
        if len(content) > 300:
            content = content[:300] + "..."

        formatted_results.append(f"Review {i}:\n{metadata_info}{content}")

    summary = f"Found {len(results)} relevant airline reviews for '{query}':\n\n"
    return summary + "\n\n".join(formatted_results)


@agentc.catalog.tool
def search_airline_reviews(query: str) -> str:
    """
    Search airline reviews using vector similarity search.
    Finds relevant customer reviews based on semantic similarity to the query.

    Args:
        query: Search query about airline experiences (e.g., 'food quality', 'seat comfort', 'service', 'delay experience')

    Returns:
        Formatted string with relevant airline reviews
    """
    try:
        # Validate query input
        if not query or not query.strip():
            return "Please provide a search query for airline reviews (e.g., 'food quality', 'seat comfort', 'service experience', 'delays')."

        query = query.strip()
        
        # Create vector store and search
        vector_store = create_vector_store()
        
        logger.info(f"Searching for airline reviews with query: '{query}'")
        results = vector_store.similarity_search(query=query, k=5)
        logger.info(f"Found {len(results)} results for query: '{query}'")
        
        return format_review_results(results, query)

    except couchbase.exceptions.CouchbaseException as e:
        logger.exception("Database error in search_airline_reviews")
        return "Unable to search airline reviews due to a database error. Please try again later."
    except Exception as e:
        logger.exception("Unexpected error in search_airline_reviews")
        return f"Error searching airline reviews: {str(e)}. Please try again."
