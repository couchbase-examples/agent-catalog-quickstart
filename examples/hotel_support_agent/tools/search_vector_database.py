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
    """Search for hotels using semantic vector similarity based on user query preferences."""
    try:
        # Get fresh cluster connection
        cluster = get_cluster_connection()
        if not cluster:
            return "Database connection failed. Please try again or contact support for assistance."
        
        # Use Capella AI embeddings to match the data loading embeddings
        try:
            if os.environ.get('CAPELLA_API_KEY') and os.environ.get('CAPELLA_API_ENDPOINT'):
                embeddings = OpenAIEmbeddings(
                    api_key=os.environ['CAPELLA_API_KEY'],
                    base_url=os.environ['CAPELLA_API_ENDPOINT'],
                    model=os.environ['CAPELLA_API_EMBEDDING_MODEL']
                )
            else:
                raise Exception("Capella credentials not available")
        except Exception as e:
            return f"Search service unavailable: {str(e)}. Please try again later."
        
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get('CB_BUCKET', 'vector-search-testing'),
            scope_name=os.environ.get('CB_SCOPE', 'agentc_data'),
            collection_name=os.environ.get('CB_COLLECTION', 'hotel_data'),
            embedding=embeddings,
            index_name=os.environ.get('CB_INDEX', 'hotel_data_index'),
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
                
            # Stop at 3 unique results for simplicity
            if len(unique_results) >= 3:
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
