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
            if os.getenv('CAPELLA_API_ENDPOINT') and os.getenv('CAPELLA_API_KEY'):
                embeddings = OpenAIEmbeddings(
                    model=os.getenv('CAPELLA_API_EMBEDDING_MODEL', 'intfloat/e5-mistral-7b-instruct'),
                    api_key=os.getenv('CAPELLA_API_KEY'),
                    base_url=os.getenv('CAPELLA_API_ENDPOINT')
                )
            else:
                embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    api_key=os.getenv('OPENAI_API_KEY')
                )
        except Exception as e:
            return f"Error setting up embeddings: {str(e)}"
        
        # Setup vector store with updated configuration
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name="travel-sample",  # Changed to travel-sample
            scope_name="agentc_data",     # Changed to agentc_data
            collection_name="hotel_data", # Changed to hotel_data
            embedding=embeddings,
            index_name="hotel_data_index",
        )
        
        # Search without filter to avoid issues, get more results initially
        try:
            search_results = vector_store.similarity_search_with_score(query, k=10)
        except Exception as search_error:
            return f"Search failed: {str(search_error)}. Please try again with different terms."
        
        if not search_results:
            return f"No hotels found matching '{query}'. Please try broader search terms or different locations."
        
        # Remove duplicates and get unique hotels
        unique_results = []
        seen_hotels = set()
        
        for doc, score in search_results:
            hotel_content = doc.page_content
            # Use first part of content as identifier to avoid duplicates
            hotel_identifier = hotel_content.split('.')[0]
            
            if hotel_identifier not in seen_hotels:
                unique_results.append((doc, score))
                seen_hotels.add(hotel_identifier)
                
            # Stop at 5 unique results for better coverage
            if len(unique_results) >= 5:
                break
        
        # Simple, clean format - no confusing guidance text
        if len(unique_results) == 1:
            return f"Found 1 hotel:\n\n{unique_results[0][0].page_content}"
        else:
            formatted_results = []
            for i, (doc, score) in enumerate(unique_results, 1):
                formatted_results.append(f"Hotel {i}: {doc.page_content}")
            
            return f"Found {len(unique_results)} hotels:\n\n" + "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Search error: {str(e)}. Please try again with different search terms."
