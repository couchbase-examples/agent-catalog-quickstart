import os
import logging
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from langchain_openai import OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from agentc.core import tool

@tool
def search_vector_database(query: str) -> str:
    """Searches the Couchbase vector database for hotels matching the user's query.
    
    Args:
        query: The search query describing hotel preferences, location, amenities, etc.
        
    Returns:
        A formatted string containing relevant hotel search results with details.
    """
    try:
        auth = PasswordAuthenticator(
            os.environ.get('CB_USERNAME', 'Administrator'), 
            os.environ.get('CB_PASSWORD', 'password')
        )
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ.get('CB_HOST', 'couchbase://localhost'), options)
        cluster.wait_until_ready(timedelta(seconds=5))
        
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model="text-embedding-3-small"
        )
        
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get('CB_BUCKET_NAME', 'hotel-search'),
            scope_name=os.environ.get('SCOPE_NAME', 'shared'),
            collection_name=os.environ.get('COLLECTION_NAME', 'hotels'),
            embedding=embeddings,
            index_name=os.environ.get('INDEX_NAME', 'hotel_vector_search'),
        )
        
        search_results = vector_store.similarity_search_with_score(query, k=5)
        
        if not search_results:
            return "No hotels found matching your criteria. Please try a different search query."
        
        formatted_results = []
        for doc, score in search_results:
            result_text = f"Match Score: {score:.3f}\nHotel Details: {doc.page_content}\n"
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logging.error(f"Error searching vector database: {str(e)}")
        raise RuntimeError(f"Failed to search hotel database: {str(e)}")
