import logging
import os
from datetime import timedelta

import agentc
import dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
_cluster = None


def _get_cluster():
    """Lazy connection to Couchbase cluster - only connects when needed."""
    global _cluster
    if _cluster is not None:
        return _cluster

    try:
        auth = PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
        )
        options = ClusterOptions(auth)

        # Use WAN profile for better timeout handling with remote clusters
        options.apply_profile("wan_development")

        _cluster = Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"), options
        )
        _cluster.wait_until_ready(timedelta(seconds=20))
        return _cluster
    except CouchbaseException as e:
        logger.error(f"Could not connect to Couchbase cluster: {e!s}")
        raise


def create_vector_store():
    """Create vector store instance for searching airline reviews."""
    try:
        # Setup embeddings directly - using Capella AI with OpenAI wrapper (priority 1)
        embeddings = OpenAIEmbeddings(
            model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
            api_key=os.getenv("CAPELLA_API_EMBEDDINGS_KEY"),
            base_url=f"{os.getenv('CAPELLA_API_ENDPOINT')}/v1",
            check_embedding_ctx_length=False,  # Fix for asymmetric models
        )

        # Create vector store
        return CouchbaseSearchVectorStore(
            cluster=_get_cluster(),
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

        # Show full content for comprehensive reviews (removed 300 character limit)

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

    except CouchbaseException as e:
        logger.exception("Database error in search_airline_reviews")
        return "Unable to search airline reviews due to a database error. Please try again later."
    except Exception as e:
        logger.exception("Unexpected error in search_airline_reviews")
        return f"Error searching airline reviews: {str(e)}. Please try again."
