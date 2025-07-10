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

        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.getenv("CB_BUCKET", "vector-search-testing"),
            scope_name=os.getenv("SCOPE_NAME", "shared"),
            collection_name=os.getenv("COLLECTION_NAME", "agentcatalog"),
            embedding=embeddings,
            index_name=os.getenv("INDEX_NAME", "vector_search_agentcatalog"),
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

        # Perform vector similarity search with filter for policy documents
        # Search for documents that are semantically similar to the query
        # Filter to only include policy documents, not routes/airlines/airports
        try:
            results = vector_store.similarity_search(
                query=query.strip(),
                k=3,  # Return top 3 most similar results
                filter={"type": {"$in": ["policy", "airline", "booking_class"]}},  # Only policy-related docs
            )
        except Exception as filter_error:
            logger.warning(f"Filter search failed: {filter_error}, trying without filter")
            # Fallback to unfiltered search if filter fails
            results = vector_store.similarity_search(
                query=query.strip(),
                k=6,  # Get more results to manually filter
                filter=None,
            )
            # Manually filter results by content patterns
            policy_results = []
            for doc in results:
                content = doc.page_content.lower()
                # Look for policy-related keywords in content
                if any(keyword in content for keyword in [
                    "policy", "baggage", "cancellation", "check-in", "restriction", 
                    "fee", "allowance", "prohibited", "carry-on", "checked"
                ]):
                    policy_results.append(doc)
                    if len(policy_results) >= 3:
                        break
            results = policy_results

        if not results:
            return "No relevant flight policies found for your query. Please try different search terms or contact customer service for specific policy questions."

        # Format results for display
        formatted_results = []
        for i, doc in enumerate(results, 1):
            # Extract policy information from document content
            content = doc.page_content
            formatted_results.append(f"Policy {i}:\n{content}")

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
        return "Error searching flight policies. Please try again or contact customer service."
