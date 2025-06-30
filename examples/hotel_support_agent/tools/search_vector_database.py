import agentc
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

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
try:
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_HOST", "couchbase://localhost"),
        couchbase.options.ClusterOptions(
            authenticator=couchbase.auth.PasswordAuthenticator(
                username=os.getenv("CB_USERNAME", "Administrator"), 
                password=os.getenv("CB_PASSWORD", "password")
            )
        ),
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    print(f"Could not connect to Couchbase cluster: {str(e)}")

@agentc.catalog.tool
def search_vector_database(query: str) -> str:
    """Search for hotels using semantic vector similarity based on user query preferences."""
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model="text-embedding-3-small"
        )
        
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get('CB_BUCKET_NAME', 'vector-search-testing'),
            scope_name=os.environ.get('SCOPE_NAME', 'shared'),
            collection_name=os.environ.get('COLLECTION_NAME', 'deepseek'),
            embedding=embeddings,
            index_name=os.environ.get('INDEX_NAME', 'vector_search_deepseek'),
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
        raise RuntimeError(f"Failed to search hotel database: {str(e)}")
