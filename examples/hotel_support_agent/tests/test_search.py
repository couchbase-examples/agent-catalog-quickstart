#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
import couchbase.auth
import couchbase.cluster
import couchbase.options
from datetime import timedelta

load_dotenv()

def test_vector_search():
    """Test vector search functionality directly"""
    
    print("🔍 Testing vector search functionality...")
    
    try:
        # Connect to Couchbase
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
        print("✅ Connected to Couchbase")
        
        # Setup embeddings
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model="text-embedding-3-small"
        )
        print("✅ OpenAI embeddings initialized")
        
        # Setup vector store
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get('CB_BUCKET_NAME', 'vector-search-testing'),
            scope_name=os.environ.get('SCOPE_NAME', 'shared'),
            collection_name=os.environ.get('COLLECTION_NAME', 'agentcatalog'),
            embedding=embeddings,
            index_name=os.environ.get('INDEX_NAME', 'vector_search_agentcatalog'),
        )
        print("✅ Vector store initialized")
        
        # Test different queries with different parameters
        test_queries = [
            "luxury hotel with pool",
            "beach resort",
            "boutique hotel San Francisco", 
            "business hotel Chicago",
            "mountain lodge"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Try with different k values and see if we get results
                for k in [1, 3, 5, 10]:
                    print(f"  Testing with k={k}...")
                    search_results = vector_store.similarity_search_with_score(query, k=k)
                    
                    if search_results:
                        print(f"  ✅ Found {len(search_results)} results with k={k}")
                        for i, (doc, score) in enumerate(search_results[:2]):  # Show first 2
                            print(f"    Result {i+1}: Score={score:.4f}, Content: {doc.page_content[:100]}...")
                        break
                    else:
                        print(f"  ❌ No results with k={k}")
                        
                if not any(vector_store.similarity_search_with_score(query, k=k) for k in [1, 3, 5, 10]):
                    print("  ❌ No results found for any k value")
                    
            except Exception as e:
                print(f"  ❌ Error during search: {e}")
        
        # Test without score to see if that works
        print("\n🔍 Testing similarity_search (without score)...")
        try:
            results = vector_store.similarity_search("luxury hotel", k=5)
            if results:
                print(f"✅ Found {len(results)} results without score")
                for i, doc in enumerate(results[:2]):
                    print(f"  Result {i+1}: {doc.page_content[:100]}...")
            else:
                print("❌ No results without score either")
        except Exception as e:
            print(f"❌ Error during similarity_search: {e}")
                
    except Exception as e:
        print(f"❌ Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_search()