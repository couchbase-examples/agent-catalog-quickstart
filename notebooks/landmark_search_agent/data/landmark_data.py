#!/usr/bin/env python3
"""
Landmark data module for the landmark search agent demo.
Loads real landmark data from travel-sample.inventory.landmark collection.
"""

import os
import json
import logging
from datetime import timedelta
from typing import List, Dict, Any

import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        auth = couchbase.auth.PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
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
            os.getenv("CB_CONN_STRING", "couchbase://localhost"), options
        )
        cluster.wait_until_ready(timedelta(seconds=15))
        return cluster
    except couchbase.exceptions.CouchbaseException as e:
        logger.error(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


def load_landmark_data_from_travel_sample():
    """Load landmark data from travel-sample.inventory.landmark collection."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        # Query to get all landmark documents from travel-sample.inventory.landmark
        query = """
            SELECT l.*, META(l).id as doc_id
            FROM `travel-sample`.inventory.landmark l
            ORDER BY l.name
        """

        logger.info("Loading landmark data from travel-sample.inventory.landmark...")
        result = cluster.query(query)

        landmarks = []
        logger.info("Processing landmark documents...")

        # Convert to list to get total count for progress bar
        landmark_rows = list(result)

        # Use tqdm for progress bar
        for row in tqdm(landmark_rows, desc="Loading landmarks", unit="landmarks"):
            landmark = row
            landmarks.append(landmark)

        logger.info(f"Loaded {len(landmarks)} landmarks from travel-sample.inventory.landmark")
        return landmarks

    except Exception as e:
        logger.error(f"Error loading landmark data: {str(e)}")
        raise


def get_landmark_texts():
    """Returns formatted landmark texts for vector store embedding from travel-sample data."""
    landmarks = load_landmark_data_from_travel_sample()
    landmark_texts = []

    logger.info("Generating landmark text embeddings...")

    # Use tqdm for progress bar while processing landmarks
    for landmark in tqdm(landmarks, desc="Processing landmarks", unit="landmarks"):
        # Start with basic info
        name = landmark.get("name", "Unknown Landmark")
        title = landmark.get("title", name)
        city = landmark.get("city", "Unknown City")
        country = landmark.get("country", "Unknown Country")

        # Build comprehensive text with all available fields
        text_parts = [f"{title} ({name}) in {city}, {country}"]

        # Add all fields dynamically instead of manual selection
        field_mappings = {
            "content": "Description",
            "address": "Address",
            "directions": "Directions",
            "phone": "Phone",
            "tollfree": "Toll-free",
            "email": "Email",
            "url": "Website",
            "hours": "Hours",
            "price": "Price",
            "activity": "Activity type",
            "type": "Type",
            "state": "State",
            "alt": "Alternative name",
            "image": "Image",
        }

        # Add all available fields
        for field, label in field_mappings.items():
            value = landmark.get(field)
            if value is not None and value != "" and value != "None":
                if isinstance(value, bool):
                    text_parts.append(f"{label}: {'Yes' if value else 'No'}")
                else:
                    text_parts.append(f"{label}: {value}")

        # Add geographic coordinates if available
        if landmark.get("geo"):
            geo = landmark["geo"]
            if geo.get("lat") and geo.get("lon"):
                accuracy = geo.get("accuracy", "Unknown")
                text_parts.append(f"Coordinates: {geo['lat']}, {geo['lon']} (accuracy: {accuracy})")

        # Add ID for reference
        if landmark.get("id"):
            text_parts.append(f"ID: {landmark['id']}")

        # Join all parts with ". "
        text = ". ".join(text_parts)
        landmark_texts.append(text)

    logger.info(f"Generated {len(landmark_texts)} landmark text embeddings")
    return landmark_texts


def load_landmark_data_to_couchbase(
    cluster, bucket_name: str, scope_name: str, collection_name: str, embeddings, index_name: str
):
    """Load landmark data from travel-sample into the target collection with embeddings."""
    try:
        # Check if data already exists
        count_query = (
            f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        )
        count_result = cluster.query(count_query)
        count_row = list(count_result)[0]
        existing_count = count_row["count"]

        if existing_count > 0:
            logger.info(
                f"Found {existing_count} existing documents in collection, skipping data load"
            )
            return

        # Get the source landmarks from travel-sample
        landmarks = load_landmark_data_from_travel_sample()
        landmark_texts = get_landmark_texts()

        # Setup vector store for the target collection
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            index_name=index_name,
        )

        # Create LlamaIndex Documents
        logger.info(f"Creating {len(landmark_texts)} LlamaIndex Documents...")
        documents = []
        
        for i, (landmark, text) in enumerate(zip(landmarks, landmark_texts)):
            document = Document(
                text=text,
                metadata={
                    "landmark_id": landmark.get("id", f"landmark_{i}"),
                    "name": landmark.get("name", "Unknown"),
                    "city": landmark.get("city", "Unknown"),
                    "country": landmark.get("country", "Unknown"),
                    "activity": landmark.get("activity", ""),
                    "type": landmark.get("type", ""),
                }
            )
            documents.append(document)

        # Use IngestionPipeline to process documents with embeddings
        logger.info(f"Processing documents with ingestion pipeline...")
        pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(), embeddings],
            vector_store=vector_store,
        )

        # Process documents in batches to avoid memory issues
        batch_size = 25  # Well below Capella AI embedding model limit
        total_batches = (len(documents) + batch_size - 1) // batch_size

        logger.info(f"Processing {len(documents)} documents in {total_batches} batches...")
        
        # Process in batches
        for i in tqdm(
            range(0, len(documents), batch_size),
            desc="Loading batches",
            unit="batch",
            total=total_batches,
        ):
            batch = documents[i : i + batch_size]
            pipeline.run(documents=batch)

        logger.info(
            f"Successfully loaded {len(documents)} landmark documents to vector store"
        )

    except Exception as e:
        logger.error(f"Error loading landmark data to Couchbase: {str(e)}")
        raise


def get_landmark_count():
    """Get the count of landmarks in travel-sample.inventory.landmark."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = "SELECT COUNT(*) as count FROM `travel-sample`.inventory.landmark"
        result = cluster.query(query)

        for row in result:
            return row["count"]

        return 0

    except Exception as e:
        logger.error(f"Error getting landmark count: {str(e)}")
        return 0


def get_landmarks_by_city(city: str, limit: int = 10):
    """Get landmarks for a specific city."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = f"""
            SELECT l.*, META(l).id as doc_id
            FROM `travel-sample`.inventory.landmark l
            WHERE LOWER(l.city) = LOWER('{city}')
            ORDER BY l.name
            LIMIT {limit}
        """

        result = cluster.query(query)
        landmarks = []

        for row in result:
            landmarks.append(row)

        return landmarks

    except Exception as e:
        logger.error(f"Error getting landmarks by city: {str(e)}")
        return []


def get_landmarks_by_activity(activity: str, limit: int = 10):
    """Get landmarks for a specific activity type."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = f"""
            SELECT l.*, META(l).id as doc_id
            FROM `travel-sample`.inventory.landmark l
            WHERE LOWER(l.activity) = LOWER('{activity}')
            ORDER BY l.name
            LIMIT {limit}
        """

        result = cluster.query(query)
        landmarks = []

        for row in result:
            landmarks.append(row)

        return landmarks

    except Exception as e:
        logger.error(f"Error getting landmarks by activity: {str(e)}")
        return []


def get_landmarks_by_country(country: str, limit: int = 10):
    """Get landmarks for a specific country."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = f"""
            SELECT l.*, META(l).id as doc_id
            FROM `travel-sample`.inventory.landmark l
            WHERE LOWER(l.country) = LOWER('{country}')
            ORDER BY l.name
            LIMIT {limit}
        """

        result = cluster.query(query)
        landmarks = []

        for row in result:
            landmarks.append(row)

        return landmarks

    except Exception as e:
        logger.error(f"Error getting landmarks by country: {str(e)}")
        return []


def search_landmarks_by_text(search_text: str, limit: int = 10):
    """Search landmarks by text content."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = f"""
            SELECT l.*, META(l).id as doc_id
            FROM `travel-sample`.inventory.landmark l
            WHERE LOWER(l.name) LIKE LOWER('%{search_text}%')
               OR LOWER(l.title) LIKE LOWER('%{search_text}%')
               OR LOWER(l.content) LIKE LOWER('%{search_text}%')
               OR LOWER(l.address) LIKE LOWER('%{search_text}%')
            ORDER BY l.name
            LIMIT {limit}
        """

        result = cluster.query(query)
        landmarks = []

        for row in result:
            landmarks.append(row)

        return landmarks

    except Exception as e:
        logger.error(f"Error searching landmarks: {str(e)}")
        return []


if __name__ == "__main__":
    # Test the data loading
    print("Testing landmark data loading...")
    count = get_landmark_count()
    print(f"Landmark count in travel-sample.inventory.landmark: {count}")

    texts = get_landmark_texts()
    print(f"Generated {len(texts)} landmark texts")

    if texts:
        print("\nFirst landmark text:")
        print(texts[0])

    # Test city search
    print("\n\nTesting city search for 'London':")
    london_landmarks = get_landmarks_by_city("London", 3)
    for landmark in london_landmarks:
        print(f"- {landmark.get('name', 'Unknown')} in {landmark.get('city', 'Unknown')}")

    # Test activity search
    print("\n\nTesting activity search for 'see':")
    see_landmarks = get_landmarks_by_activity("see", 3)
    for landmark in see_landmarks:
        print(f"- {landmark.get('name', 'Unknown')} ({landmark.get('activity', 'Unknown')})")

    # Test text search
    print("\n\nTesting text search for 'museum':")
    museum_landmarks = search_landmarks_by_text("museum", 3)
    for landmark in museum_landmarks:
        print(f"- {landmark.get('name', 'Unknown')} in {landmark.get('city', 'Unknown')}")
