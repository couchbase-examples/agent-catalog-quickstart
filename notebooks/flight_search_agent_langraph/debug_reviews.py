#!/usr/bin/env python3
"""
Debug script to check what airline reviews are in the Couchbase database
and test the search functionality.
"""

import os
import sys
from datetime import timedelta
import couchbase
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.auth import PasswordAuthenticator
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import OpenAIEmbeddings
import base64

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")
    print("Using system environment variables")


def setup_environment():
    """Setup environment variables if not already set"""
    # Check for required environment variables
    required_vars = [
        "CB_CONN_STRING",
        "CB_USERNAME",
        "CB_PASSWORD",
        "CB_BUCKET",
        "CB_SCOPE",
        "CB_COLLECTION",
        "CB_INDEX",
        "CAPELLA_API_ENDPOINT",
        "CAPELLA_API_EMBEDDING_MODEL",
    ]

    missing_vars = []
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)

    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("Will try to continue with available variables...")

    # Print current environment variables for debugging
    print("\n📝 Current environment variables:")
    for var in required_vars:
        value = os.environ.get(var, "NOT_SET")
        if "PASSWORD" in var or "KEY" in var:
            print(f"  {var}: {'***' if value != 'NOT_SET' else 'NOT_SET'}")
        else:
            print(f"  {var}: {value}")

    return True


def connect_to_couchbase():
    """Connect to Couchbase cluster"""
    try:
        auth = PasswordAuthenticator(
            username=os.environ["CB_USERNAME"], password=os.environ["CB_PASSWORD"]
        )
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")

        cluster = Cluster(os.environ["CB_CONN_STRING"], options)
        cluster.wait_until_ready(timedelta(seconds=10))

        print("✅ Connected to Couchbase successfully")
        return cluster
    except Exception as e:
        print(f"❌ Failed to connect to Couchbase: {e}")
        return None


def check_collection_stats(cluster):
    """Check basic stats about the airline reviews collection"""
    try:
        bucket_name = os.environ["CB_BUCKET"]
        scope_name = os.environ["CB_SCOPE"]
        collection_name = os.environ["CB_COLLECTION"]

        # Count total documents
        count_query = (
            f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        )
        result = cluster.query(count_query)
        count = next(iter(result))["count"]

        print(f"📊 Total documents in {bucket_name}.{scope_name}.{collection_name}: {count}")

        # Get sample documents
        sample_query = f"SELECT * FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` LIMIT 5"
        result = cluster.query(sample_query)

        print("\n📝 Sample documents:")
        for i, doc in enumerate(result, 1):
            print(f"\n--- Document {i} ---")
            print(f"Keys: {list(doc.keys()) if hasattr(doc, 'keys') else 'No keys'}")
            if hasattr(doc, "get"):
                text = doc.get("text", doc.get("content", "No text field"))
                print(f"Text preview: {text[:200]}...")

        return count
    except Exception as e:
        print(f"❌ Error checking collection stats: {e}")
        return 0


def search_reviews_sample(cluster):
    """Test searching for sample airline reviews"""
    try:
        # Setup embeddings
        if (
            os.environ.get("CB_USERNAME")
            and os.environ.get("CB_PASSWORD")
            and os.environ.get("CAPELLA_API_ENDPOINT")
            and os.environ.get("CAPELLA_API_EMBEDDING_MODEL")
        ):
            api_key = base64.b64encode(
                f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode()
            ).decode()

            embeddings = OpenAIEmbeddings(
                model=os.environ["CAPELLA_API_EMBEDDING_MODEL"],
                api_key=api_key,
                base_url=f"{os.environ['CAPELLA_API_ENDPOINT']}/v1",
            )
        else:
            print("❌ Missing Capella AI credentials")
            return

        # Setup vector store
        index_name = os.environ.get("CB_INDEX", "airline_reviews_index")
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ["CB_BUCKET"],
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            embedding=embeddings,
            index_name=index_name,
        )

        # Test various search queries
        test_queries = [
            "IndiGo",
            "service quality",
            "customer service",
            "food",
            "seats",
            "delays",
            "Air India",
            "SpiceJet",
            "Vistara",
            "good",
            "bad",
            "excellent",
            "poor",
        ]

        print("\n🔍 Testing search queries:")
        for query in test_queries:
            try:
                results = vector_store.similarity_search(query, k=3)
                print(f"\n--- Query: '{query}' ---")
                print(f"Found {len(results)} results")
                for i, doc in enumerate(results, 1):
                    content = (
                        doc.page_content[:100] if hasattr(doc, "page_content") else str(doc)[:100]
                    )
                    print(f"  {i}. {content}...")
            except Exception as e:
                print(f"❌ Search error for '{query}': {e}")

    except Exception as e:
        print(f"❌ Error in search testing: {e}")


def inspect_raw_documents(cluster):
    """Inspect raw documents to understand their structure"""
    try:
        bucket_name = os.environ["CB_BUCKET"]
        scope_name = os.environ["CB_SCOPE"]
        collection_name = os.environ["CB_COLLECTION"]

        # Get documents with all fields
        query = f"""
        SELECT META().id, airline_reviews.text as text_content
        FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` 
        LIMIT 5
        """

        result = cluster.query(query)

        print("\n🔍 Raw document inspection:")
        for i, doc in enumerate(result, 1):
            print(f"\n--- Raw Document {i} ---")
            doc_id = doc.get("id", "unknown")
            text_content = doc.get("text_content", "No text content")
            print(f"ID: {doc_id}")
            print(f"Text: {text_content[:300]}...")

            # Look for airline names in the text
            airlines = ["IndiGo", "Air India", "SpiceJet", "Vistara", "Go Air", "Jet Airways"]
            found_airlines = [
                airline for airline in airlines if airline.lower() in text_content.lower()
            ]
            if found_airlines:
                print(f"Airlines mentioned: {found_airlines}")

    except Exception as e:
        print(f"❌ Error inspecting documents: {e}")


def main():
    """Main debug function"""
    print("🔍 Airline Reviews Database Debug Script")
    print("=" * 50)

    # Setup environment
    setup_environment()

    # Connect to Couchbase
    cluster = connect_to_couchbase()
    if not cluster:
        return

    # Check collection stats
    count = check_collection_stats(cluster)
    if count == 0:
        print("❌ No documents found in collection")
        return

    # Inspect raw documents
    inspect_raw_documents(cluster)

    # Test search functionality
    search_reviews_sample(cluster)

    print("\n✅ Debug complete!")


if __name__ == "__main__":
    main()
