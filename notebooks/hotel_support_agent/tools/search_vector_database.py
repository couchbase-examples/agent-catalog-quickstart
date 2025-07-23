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
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


from datetime import timedelta

dotenv.load_dotenv()

# Generate Capella AI API key if endpoint is provided
if os.getenv('CAPELLA_API_ENDPOINT') and os.getenv('CB_USERNAME') and os.getenv('CB_PASSWORD'):
    os.environ['CAPELLA_API_KEY'] = base64.b64encode(
        f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode("utf-8")
    ).decode("utf-8")


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        auth = couchbase.auth.PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"), 
            password=os.getenv("CB_PASSWORD", "password")
        )
        options = couchbase.options.ClusterOptions(authenticator=auth)
        # Use WAN profile for better timeout handling with remote clusters
        options.apply_profile("wan_development")
        
        # Additional timeout configurations for Capella cloud connections
        from couchbase.options import ClusterTimeoutOptions
        
        # Configure extended timeouts for cloud connectivity
        timeout_options = ClusterTimeoutOptions(
            kv_timeout=timedelta(seconds=10),  # Key-value operations
            kv_durable_timeout=timedelta(seconds=15),  # Durable writes
            query_timeout=timedelta(seconds=30),  # N1QL queries
            search_timeout=timedelta(seconds=30),  # Search operations
            management_timeout=timedelta(seconds=30),  # Management operations
            bootstrap_timeout=timedelta(seconds=20),  # Initial connection
        )
        options.timeout_options = timeout_options
        
        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            options
        )
        cluster.wait_until_ready(timedelta(seconds=15))
        return cluster
    except couchbase.exceptions.CouchbaseException as e:
        print(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


@agentc.catalog.tool
def search_vector_database(query: str) -> str:
    """
    Search for hotels in the vector database using semantic similarity.
    Returns raw search results for the agent to intelligently process.
    
    Args:
        query: The search query describing the desired hotel characteristics
        
    Returns:
        A formatted string containing the search results
    """
    try:
        # Get cluster connection
        cluster = get_cluster_connection()
        if not cluster:
            return "Error: Could not connect to Couchbase cluster"
        
        # Setup embeddings
        try:
            # COMMENTED OUT - Capella AI embeddings (testing NVIDIA embeddings instead)
            # if os.getenv('CAPELLA_API_ENDPOINT') and os.getenv('CAPELLA_API_KEY'):
            #     embeddings = OpenAIEmbeddings(
            #         model=os.getenv('CAPELLA_API_EMBEDDING_MODEL', 'intfloat/e5-mistral-7b-instruct'),
            #         api_key=os.getenv('CAPELLA_API_KEY'),
            #         base_url=os.getenv('CAPELLA_API_ENDPOINT')
            #     )

            # COMMENTED OUT - OpenAI embeddings (testing NVIDIA embeddings instead)
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_API_ENDPOINT'),
            )

            # embeddings = NVIDIAEmbeddings(
            #     model="nvidia/nv-embedqa-e5-v5", 
            #     api_key=os.getenv('NVIDIA_API_KEY'),
            #     truncate="END", 
            # )

        except Exception as e:
            return f"Error setting up embeddings: {str(e)}"
        
        # Setup vector store
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name="travel-sample",
            scope_name="agentc_data",
            collection_name="hotel_data",
            embedding=embeddings,
            index_name="hotel_data_index",
        )
        
        # Search for hotels - return raw results for agent to process
        try:
            search_results = vector_store.similarity_search_with_score(query, k=10)
        except Exception as search_error:
            return f"Search failed: {str(search_error)}. Please try again with different terms."
        
        if not search_results:
            return f"No hotels found matching '{query}'. Please try broader search terms or different locations."
        
        # Remove duplicates but don't filter by location - let agent handle that
        unique_results = []
        seen_hotels = set()
        
        for doc, score in search_results:
            hotel_content = doc.page_content
            hotel_identifier = hotel_content.split('.')[0]
            
            if hotel_identifier not in seen_hotels:
                unique_results.append((doc, score))
                seen_hotels.add(hotel_identifier)
                
            # Stop at 8 unique results to give agent good variety
            if len(unique_results) >= 8:
                break
        
        # Format results for agent to intelligently process
        if len(unique_results) == 1:
            result = f"Found 1 hotel:\n\n{unique_results[0][0].page_content}"
        else:
            formatted_results = []
            for i, (doc, score) in enumerate(unique_results, 1):
                formatted_results.append(f"Hotel {i}: {doc.page_content}")
            
            result = f"Found {len(unique_results)} hotels:\n\n" + "\n\n".join(formatted_results)
        
        return result
        
    except Exception as e:
        return f"Search error: {str(e)}. Please try again with different search terms."
