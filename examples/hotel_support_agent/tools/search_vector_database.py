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

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
try:
    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"), 
        password=os.getenv("CB_PASSWORD", "password")
    )
    options = couchbase.options.ClusterOptions(authenticator=auth)
    # Use WAN profile for better timeout handling with remote clusters
    options.apply_profile("wan_development")
    
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_HOST", "couchbase://localhost"),
        options
    )
    cluster.wait_until_ready(timedelta(seconds=15))
except couchbase.exceptions.CouchbaseException as e:
    print(f"Could not connect to Couchbase cluster: {str(e)}")

@agentc.catalog.tool
def search_vector_database(query: str) -> str:
    """Search for hotels using semantic vector similarity based on user query preferences."""
    try:
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
            raise RuntimeError(f"⚠️ Capella AI not available {str(e)}")
        
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get('CB_BUCKET_NAME', 'vector-search-testing'),
            scope_name=os.environ.get('SCOPE_NAME', 'shared'),
            collection_name=os.environ.get('COLLECTION_NAME', 'agentcatalog'),
            embedding=embeddings,
            index_name=os.environ.get('INDEX_NAME', 'vector_search_agentcatalog'),
        )
        
        search_results = vector_store.similarity_search_with_score(query, k=10)  # Get more results for filtering
        
        if not search_results:
            return "No hotels found matching your criteria. Please try a different search query."
        
        # Deduplicate results based on content
        seen_content = set()
        unique_results = []
        
        for doc, score in search_results:
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                unique_results.append((doc, score))
                
                # Limit to top 3 unique results for cleaner output
                if len(unique_results) >= 3:
                    break
        
        # Format results with clear separators and numbering
        formatted_results = []
        for i, (doc, score) in enumerate(unique_results, 1):
            result_text = f"""HOTEL {i}:
Match Score: {score:.3f}
Hotel Details: {doc.page_content}
{'='*60}"""
            formatted_results.append(result_text)
        
        # Add summary at the end
        summary = f"\nSEARCH SUMMARY: Found {len(unique_results)} unique hotels matching your criteria."
        
        return "\n\n".join(formatted_results) + summary
        
    except Exception as e:
        raise RuntimeError(f"Failed to search hotel database: {str(e)}")
