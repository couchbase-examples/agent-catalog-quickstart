#!/usr/bin/env python3

import os
import json
import requests
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def test_search_api():
    """Test vector search using direct REST API"""
    
    print("🔍 Testing vector search via REST API...")
    
    try:
        # Get embeddings for test query
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model="text-embedding-3-small"
        )
        
        query_text = "luxury hotel with pool"
        query_vector = embeddings.embed_query(query_text)
        vector_preview = query_vector[:10] + ["..."] if len(query_vector) > 10 else query_vector
        print(f"✅ Generated embedding for query: '{query_text}' (dimension: {len(query_vector)})")
        print(f"   Vector preview: {vector_preview}")
        
        # Prepare search request
        search_request = {
            "query": {
                "match_none": {}
            },
            "knn": [
                {
                    "field": "embedding",
                    "vector": query_vector,
                    "k": 5
                }
            ],
            "size": 5
        }
        
        # Couchbase search endpoint
        cb_host = os.environ.get('CB_HOST', 'couchbase://localhost')
        
        if 'localhost' in cb_host:
            # Local Couchbase
            search_url = "http://localhost:8094/api/index/vector_search_agentcatalog/query"
        else:
            # Capella - extract the host and use proper FTS endpoint
            host = cb_host.replace('couchbase://', '').replace('couchbases://', '')
            search_url = f"https://{host}/api/index/vector_search_agentcatalog/query"
        
        print(f"🔍 Calling search API: {search_url}")
        
        # Make request
        response = requests.post(
            search_url,
            json=search_request,
            auth=(os.environ['CB_USERNAME'], os.environ['CB_PASSWORD']),
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"✅ Search successful!")
            print(f"📊 Total hits: {result.get('total_hits', 0)}")
            print(f"📊 Max score: {result.get('max_score', 'N/A')}")
            
            hits = result.get('hits', [])
            if hits:
                print(f"📋 Found {len(hits)} results:")
                for i, hit in enumerate(hits[:3]):  # Show first 3
                    print(f"  Result {i+1}:")
                    print(f"    Score: {hit.get('score', 'N/A')}")
                    print(f"    ID: {hit.get('id', 'N/A')}")
                    if 'fields' in hit:
                        text_content = hit['fields'].get('text', 'N/A')
                        print(f"    Text: {text_content[:100]}...")
            else:
                print("❌ No results found")
                
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_api()