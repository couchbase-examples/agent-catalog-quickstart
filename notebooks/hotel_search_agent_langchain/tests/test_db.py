#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions

load_dotenv()

def check_database():
    """Check what's actually in the database"""
    
    # Connect to Couchbase
    auth = PasswordAuthenticator(
        os.environ['CB_USERNAME'],
        os.environ['CB_PASSWORD']
    )
    
    cluster = Cluster(
        os.environ['CB_HOST'],
        ClusterOptions(auth)
    )
    
    bucket = cluster.bucket(os.environ['CB_BUCKET_NAME'])
    collection = bucket.scope('_default').collection('_default')
    
    print("🔍 Checking database contents...")
    
    # Check if any documents exist
    try:
        # Query to get all documents
        query = "SELECT META().id, * FROM `vector-search-testing` LIMIT 10"
        result = cluster.query(query)
        
        docs = list(result)
        print(f"📊 Found {len(docs)} documents in database")
        
        for i, doc in enumerate(docs):
            print(f"\n📄 Document {i+1}:")
            print(f"  ID: {doc.get('id', 'N/A')}")
            
            # Print document structure
            for key, value in doc.items():
                if key != 'id':
                    if isinstance(value, dict):
                        print(f"  {key}: {type(value)} with keys: {list(value.keys())}")
                    elif isinstance(value, list):
                        print(f"  {key}: list with {len(value)} items")
                    else:
                        print(f"  {key}: {str(value)[:100]}...")
        
        if not docs:
            print("❌ No documents found in database!")
            print("🔧 Let's check if the collection exists...")
            
            # Try to insert a test document
            test_doc = {
                "name": "Test Hotel",
                "description": "A test hotel for debugging",
                "embedding": [0.1] * 1536  # OpenAI embedding dimensions
            }
            
            collection.upsert("test_hotel", test_doc)
            print("✅ Successfully inserted test document")
            
            # Try to retrieve it
            retrieved = collection.get("test_hotel")
            print("✅ Successfully retrieved test document")
            print(f"📄 Retrieved: {retrieved.content_as[dict]}")
            
    except Exception as e:
        print(f"❌ Error querying database: {e}")
        
    # Check vector search index
    print("\n🔍 Checking vector search index...")
    try:
        # Try to perform a simple vector search
        search_request = {
            "knn": [
                {
                    "field": "embedding",
                    "vector": [0.1] * 1536,  # Test vector
                    "k": 5
                }
            ]
        }
        
        # This would normally be done through the search endpoint
        print("🔧 Vector search index check would require search endpoint")
        
    except Exception as e:
        print(f"❌ Error checking vector search: {e}")

if __name__ == "__main__":
    check_database()