import agentc
import base64
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseVectorStore
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


from datetime import timedelta

dotenv.load_dotenv()

# Generate Capella AI API key if endpoint is provided
if (
    os.getenv("CAPELLA_API_ENDPOINT")
    and os.getenv("CB_USERNAME")
    and os.getenv("CB_PASSWORD")
):
    os.environ["CAPELLA_API_KEY"] = base64.b64encode(
        f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode("utf-8")
    ).decode("utf-8")


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

        # Setup embeddings
        try:
            # Capella AI embeddings
            if os.getenv("CAPELLA_API_ENDPOINT") and os.getenv("CAPELLA_API_KEY"):
                embeddings = OpenAIEmbeddings(
                    model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                    api_key=os.getenv("CAPELLA_API_KEY"),
                    base_url=os.getenv("CAPELLA_API_ENDPOINT"),
                )

            # NVIDIA embeddings
            # embeddings = NVIDIAEmbeddings(
            #     model="nvidia/nv-embedqa-e5-v5",
            #     api_key=os.getenv('NVIDIA_API_KEY'),
            #     truncate="END",
            # )

            # OpenAI embeddings
            # embeddings = OpenAIEmbeddings(
            #     model="text-embedding-3-small",
            #     api_key=os.getenv('OPENAI_API_KEY'),
            #     base_url=os.getenv('OPENAI_API_ENDPOINT'),
            # )

        except Exception as e:
            return f"ERROR: Search system unavailable - {str(e)}"

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
