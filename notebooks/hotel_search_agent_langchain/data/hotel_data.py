#!/usr/bin/env python3
"""
Hotel data module for the hotel support agent demo.
Loads real hotel data from travel-sample.inventory.hotel collection.
"""

import os
import logging
import time
from datetime import timedelta

import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from tqdm import tqdm

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_with_backoff(func, retries=3):
    """Simple retry with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except Exception:
            if attempt == retries - 1:
                raise
            delay = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
            time.sleep(delay)


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

        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"), options
        )
        cluster.wait_until_ready(timedelta(seconds=15))
        return cluster
    except couchbase.exceptions.CouchbaseException as e:
        logger.error(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


def load_hotel_data_from_travel_sample():
    """Load hotel data from travel-sample.inventory.hotel collection."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        # Query to get all hotel documents from travel-sample.inventory.hotel
        query = """
            SELECT h.*, META(h).id as doc_id
            FROM `travel-sample`.inventory.hotel h
            ORDER BY h.name
        """

        logger.info("Loading hotel data from travel-sample.inventory.hotel...")
        result = cluster.query(query)

        hotels = []
        for row in result:
            hotel = row
            hotels.append(hotel)

        logger.info(f"Loaded {len(hotels)} hotels from travel-sample.inventory.hotel")
        return hotels

    except Exception as e:
        logger.error(f"Error loading hotel data: {str(e)}")
        raise


def get_hotel_texts():
    """Returns formatted hotel texts for vector store embedding from travel-sample data."""
    hotels = load_hotel_data_from_travel_sample()
    hotel_texts = []

    for hotel in tqdm(hotels, desc="Processing hotels"):
        # Start with basic info
        name = hotel.get("name", "Unknown Hotel")
        city = hotel.get("city", "Unknown City")
        country = hotel.get("country", "Unknown Country")

        # Build text with PRIORITIZED information for search
        text_parts = [f"{name} in {city}, {country}"]

        # PRIORITY 1: Location details (critical for search)
        location_fields = ["address", "state", "directions"]
        for field in location_fields:
            value = hotel.get(field)
            if value and value != "None":
                text_parts.append(f"{field.title()}: {value}")

        # PRIORITY 2: Key amenities (most searched features)
        amenity_fields = [
            ("free_breakfast", "Free breakfast"),
            ("free_internet", "Free internet"), 
            ("free_parking", "Free parking"),
            ("pets_ok", "Pets allowed")
        ]
        for field, label in amenity_fields:
            value = hotel.get(field)
            if value is not None:
                text_parts.append(f"{label}: {'Yes' if value else 'No'}")

        # PRIORITY 3: Hotel description and type
        description_fields = [
            ("description", "Description"),
            ("type", "Type"),
            ("title", "Title")
        ]
        for field, label in description_fields:
            value = hotel.get(field)
            if value and value != "None":
                text_parts.append(f"{label}: {value}")

        # PRIORITY 4: Other details (less critical for search)
        other_fields = [
            ("price", "Price"),
            ("checkin", "Check-in"),
            ("checkout", "Check-out"),
            ("phone", "Phone"),
            ("email", "Email"),
            ("vacancy", "Vacancy"),
            ("alias", "Also known as")
        ]
        for field, label in other_fields:
            value = hotel.get(field)
            if value and value != "None":
                if isinstance(value, bool):
                    text_parts.append(f"{label}: {'Yes' if value else 'No'}")
                else:
                    text_parts.append(f"{label}: {value}")

        # Add geographic coordinates if available
        if hotel.get("geo"):
            geo = hotel["geo"]
            if geo.get("lat") and geo.get("lon"):
                text_parts.append(f"Coordinates: {geo['lat']}, {geo['lon']}")

        # Add review summary if available
        if hotel.get("reviews") and isinstance(hotel["reviews"], list):
            review_count = len(hotel["reviews"])
            if review_count > 0:
                text_parts.append(f"Reviews: {review_count} customer reviews available")

                # Include a sample of review content for better search matching
                sample_reviews = hotel["reviews"][:2]  # First 2 reviews
                for i, review in enumerate(sample_reviews):
                    if review.get("content"):
                        # Truncate long reviews for embedding efficiency
                        content = (
                            review["content"][:200] + "..."
                            if len(review["content"]) > 200
                            else review["content"]
                        )
                        text_parts.append(f"Review {i + 1}: {content}")

        # Add public likes if available
        if hotel.get("public_likes") and isinstance(hotel["public_likes"], list):
            likes_count = len(hotel["public_likes"])
            if likes_count > 0:
                text_parts.append(f"Public likes: {likes_count} likes")

        # Join all parts with ". "
        text = ". ".join(text_parts)
        hotel_texts.append(text)

    logger.info(f"Generated {len(hotel_texts)} hotel text embeddings")
    return hotel_texts


def load_hotel_data_to_couchbase(
    cluster,
    bucket_name: str,
    scope_name: str,
    collection_name: str,
    embeddings,
    index_name: str,
):
    """Load hotel data from travel-sample into the target collection with embeddings."""
    try:
        # Check if data already exists
        count_query = f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        count_result = cluster.query(count_query)
        count_row = list(count_result)[0]
        existing_count = count_row["count"]

        if existing_count > 0:
            logger.info(
                f"Found {existing_count} existing documents in collection, skipping data load"
            )
            return

        # Get hotel texts for embeddings
        hotel_texts = get_hotel_texts()

        # Setup vector store for the target collection
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            embedding=embeddings,
            index_name=index_name,
        )

        # Add hotel texts to vector store with batch processing
        logger.info(
            f"Loading {len(hotel_texts)} hotel embeddings to {bucket_name}.{scope_name}.{collection_name}"
        )

        # Process in batches with simple retry
        batch_size = 10

        with tqdm(total=len(hotel_texts), desc="Loading hotel embeddings") as pbar:
            for i in range(0, len(hotel_texts), batch_size):
                batch = hotel_texts[i : i + batch_size]
                
                def add_batch():
                    return vector_store.add_texts(texts=batch, batch_size=batch_size)
                
                retry_with_backoff(add_batch, retries=3)
                pbar.update(len(batch))

        logger.info(
            f"Successfully loaded {len(hotel_texts)} hotel embeddings to vector store"
        )

    except Exception as e:
        logger.error(f"Error loading hotel data to Couchbase: {str(e)}")
        raise


def get_hotel_count():
    """Get the count of hotels in travel-sample.inventory.hotel."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")

        query = "SELECT COUNT(*) as count FROM `travel-sample`.inventory.hotel"
        result = cluster.query(query)

        for row in result:
            return row["count"]

        return 0

    except Exception as e:
        logger.error(f"Error getting hotel count: {str(e)}")
        return 0


if __name__ == "__main__":
    # Test the data loading
    print("Testing hotel data loading...")
    count = get_hotel_count()
    print(f"Hotel count in travel-sample.inventory.hotel: {count}")

    texts = get_hotel_texts()
    print(f"Generated {len(texts)} hotel texts")

    if texts:
        print("\nFirst hotel text:")
        print(texts[0])
