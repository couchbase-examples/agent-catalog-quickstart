import logging
import os
import sys
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

# Import shared AI services module using robust project root discovery
def find_project_root():
    """Find the project root by looking for the shared directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Look for the shared directory as the definitive marker
        shared_path = os.path.join(current, 'shared')
        if os.path.exists(shared_path) and os.path.isdir(shared_path):
            return current
        current = os.path.dirname(current)
    return None

# Add project root to Python path
project_root = find_project_root()
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

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


# Cache vector store instance to avoid repeated AI service setup
_vector_store_cache = None

def _get_vector_store():
    """Get vector store instance for searching airline reviews with caching."""
    global _vector_store_cache
    
    if _vector_store_cache is not None:
        return _vector_store_cache
    
    try:
        # Setup embeddings using shared module (only once)
        embeddings, _ = setup_ai_services(framework="langgraph")

        # Create and cache vector store
        _vector_store_cache = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.getenv("CB_BUCKET", "travel-sample"),
            scope_name=os.getenv("CB_SCOPE", "agentc_data"),
            collection_name=os.getenv("CB_COLLECTION", "airline_reviews"),
            embedding=embeddings,
            index_name=os.getenv("CB_INDEX", "airline_reviews_index"),
        )
        
        logger.info("✅ Vector store initialized and cached")
        return _vector_store_cache
        
    except Exception as e:
        msg = f"Failed to create vector store: {e!s}"
        logger.error(msg)
        raise RuntimeError(msg)


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

        # Get vector store
        vector_store = _get_vector_store()

        # Perform vector similarity search for airline reviews
        try:
            logger.info(f"Searching for airline reviews with query: '{query.strip()}'")
            results = vector_store.similarity_search(
                query=query.strip(),
                k=5,  # Return top 5 most similar reviews for better coverage
            )
            logger.info(f"Found {len(results)} results for query: '{query.strip()}'")
        except Exception as search_error:
            logger.exception(f"Search failed: {search_error}")
            return f"Search error: {search_error}. Please try again or contact customer service."

        if not results:
            return "No relevant airline reviews found for your query. Please try different search terms (e.g., 'food', 'seats', 'service', 'delays', 'check-in', 'baggage') or be more specific about the airline or service aspect you're interested in."

        # Format results for display
        formatted_results = []
        for i, doc in enumerate(results, 1):
            # Extract review information from document content
            content = doc.page_content

            # Include metadata if available
            metadata_info = ""
            if hasattr(doc, "metadata") and doc.metadata:
                parts = []
                if "airline" in doc.metadata:
                    parts.append(f"Airline: {doc.metadata['airline']}")
                if "rating" in doc.metadata:
                    parts.append(f"Rating: {doc.metadata['rating']}/5")
                if "travel_date" in doc.metadata:
                    parts.append(f"Travel Date: {doc.metadata['travel_date']}")
                if "aircraft" in doc.metadata:
                    parts.append(f"Aircraft: {doc.metadata['aircraft']}")
                if "seat_type" in doc.metadata:
                    parts.append(f"Class: {doc.metadata['seat_type']}")
                if "route" in doc.metadata:
                    parts.append(f"Route: {doc.metadata['route']}")

                if parts:
                    metadata_info = f"[{' | '.join(parts)}]\n"

            # Limit content length for readability
            if len(content) > 400:
                content = content[:400] + "..."

            formatted_results.append(f"Review {i}:\n{metadata_info}{content}")

        # Add helpful summary
        total_reviews = len(results)
        summary = f"Found {total_reviews} relevant airline reviews for '{query.strip()}':\n\n"

        return summary + "\n\n".join(formatted_results)

    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Database error while searching reviews: {e!s}"
        logger.exception("Database error in search_airline_reviews")
        return "Unable to search airline reviews due to a database error. Please try again later."
    except Exception as e:
        error_msg = f"Error searching airline reviews: {e!s}"
        logger.exception("Unexpected error in search_airline_reviews")
        return f"Error searching airline reviews: {error_msg}. Please try again or contact customer service."
