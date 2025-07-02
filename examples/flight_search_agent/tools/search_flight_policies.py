import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv(override=True)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
try:
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        couchbase.options.ClusterOptions(
            authenticator=couchbase.auth.PasswordAuthenticator(
                username=os.getenv("CB_USERNAME", "Administrator"),
                password=os.getenv("CB_PASSWORD", "password")
            )
        ),
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"
    print(error_msg)


def _get_vector_store():
    """Get vector store instance for searching flight data."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small"
        )

        vector_store = CouchbaseSearchVectorStore(
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
        # Get vector store
        vector_store = _get_vector_store()
        
        # Perform vector similarity search
        # Search for documents that are semantically similar to the query
        results = vector_store.similarity_search(
            query=query,
            k=3,  # Return top 3 most similar results
            filter=None  # No additional filtering
        )
        
        if not results:
            return "No relevant flight policies found for your query. Please contact customer service for specific policy questions."
        
        # Format results for display
        formatted_results = []
        for i, doc in enumerate(results, 1):
            # Extract policy information from document content
            content = doc.page_content
            formatted_results.append(f"Policy {i}:\n{content}")
        
        return "\n\n".join(formatted_results)
        
    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Database error while searching policies: {e!s}"
        return "Unable to search flight policies due to a database error. Please try again later."
    except Exception as e:
        error_msg = f"Error searching flight policies: {e!s}"
        return "Error searching flight policies. Please try again or contact customer service."