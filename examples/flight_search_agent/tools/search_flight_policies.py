import logging
import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
import openai

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
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        options
    )
    cluster.wait_until_ready(timedelta(seconds=20))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"
    print(error_msg)


def _get_vector_store():
    """Get vector store instance for searching flight data - optimized for Capella AI."""
    try:
        # Use Capella AI embeddings if available, fallback to OpenAI
        if (
            os.getenv("CB_USERNAME")
            and os.getenv("CB_PASSWORD")
            and os.getenv("CAPELLA_API_ENDPOINT")
            and os.getenv("CAPELLA_API_EMBEDDING_MODEL")
        ):
            import base64

            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()

            embeddings = OpenAIEmbeddings(
                model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                api_key=api_key,
                base_url=f"{os.getenv('CAPELLA_API_ENDPOINT')}/v1",
            )
        else:
            embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
            )

        # Updated to use correct environment variable names and default values
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.getenv("CB_BUCKET", "vector-search-testing"),
            scope_name=os.getenv("CB_SCOPE", "agentc_data"),
            collection_name=os.getenv("CB_COLLECTION", "flight_policies"),
            embedding=embeddings,
            index_name=os.getenv("CB_INDEX", "flight_policies_index"),
        )
        return vector_store
    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e!s}")


@agentc.catalog.tool
def search_flight_policies(query: str) -> str:
    """
    Search flight policies using vector similarity search.
    Finds relevant policies based on semantic similarity to the query.
    """
    try:
        # Validate query input
        if not query or not query.strip():
            return "Please provide a search query for flight policies (e.g., 'baggage policy', 'cancellation rules')."
        
        # Get vector store
        vector_store = _get_vector_store()

        # Perform vector similarity search specifically for flight policies
        # First try without filter to see if we can find any documents
        try:
            logger.info(f"Searching for flight policies with query: '{query.strip()}'")
            results = vector_store.similarity_search(
                query=query.strip(),
                k=5,  # Return top 5 most similar results for better coverage
            )
            logger.info(f"Found {len(results)} results for query: '{query.strip()}'")
        except Exception as search_error:
            logger.error(f"Search failed: {search_error}")
            return f"Search error: {search_error}. Please try again or contact customer service."

        if not results:
            return "No relevant flight policies found for your query. Please try different search terms (e.g., 'baggage', 'carry-on', 'checked bags', 'fees', 'weight limits') or contact customer service for specific policy questions."

        # Format results for display
        formatted_results = []
        for i, doc in enumerate(results, 1):
            # Extract policy information from document content
            content = doc.page_content[:500]  # Limit content length for readability
            if len(doc.page_content) > 500:
                content += "..."
            
            # Include metadata if available
            metadata_info = ""
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'title' in doc.metadata:
                    metadata_info = f"Title: {doc.metadata['title']}\n"
                elif 'source' in doc.metadata:
                    metadata_info = f"Source: {doc.metadata['source']}\n"
            
            formatted_results.append(f"Policy {i}:\n{metadata_info}{content}")

        return "\n\n".join(formatted_results)

    except openai.OpenAIError as e:
        # Handle OpenAI service errors (model unavailable, health errors, etc.)
        logger.warning(f"OpenAI service error in search_flight_policies: {e}")
        return "The flight policy search service is temporarily unavailable. Please try again in a few minutes or contact customer service for assistance."
    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Database error while searching policies: {e!s}"
        logger.exception("Database error in search_flight_policies")
        return "Unable to search flight policies due to a database error. Please try again later."
    except Exception as e:
        error_msg = f"Error searching flight policies: {e!s}"
        logger.exception("Unexpected error in search_flight_policies")
        return f"Error searching flight policies: {error_msg}. Please try again or contact customer service."
