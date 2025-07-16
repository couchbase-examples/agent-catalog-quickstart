#!/usr/bin/env python3
"""
Airline reviews data module for the flight search agent demo.
Downloads and processes Japanese Airlines Reviews dataset from Kaggle.
"""

import os
import json
import logging
from datetime import timedelta
from typing import List, Dict, Any

import pandas as pd
import kagglehub
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
from langchain_couchbase.vectorstores import CouchbaseVectorStore

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
        logger.error(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


def load_airline_reviews_from_kaggle():
    """Load airline reviews data from Kaggle dataset."""
    try:
        # Download the dataset from Kaggle
        logger.info("Downloading Japanese Airlines Reviews dataset from Kaggle...")
        path = kagglehub.dataset_download("kanchana1990/japanese-airlines-reviews")
        
        # Find the CSV file
        csv_file = None
        for file in os.listdir(path):
            if file.endswith('.csv'):
                csv_file = os.path.join(path, file)
                break
        
        if not csv_file:
            raise FileNotFoundError("No CSV file found in downloaded dataset")
        
        # Load the CSV file
        logger.info(f"Loading reviews from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to list of dictionaries
        reviews = df.to_dict('records')
        
        logger.info(f"Loaded {len(reviews)} airline reviews from Kaggle dataset")
        return reviews
    
    except Exception as e:
        logger.error(f"Error loading airline reviews from Kaggle: {str(e)}")
        raise


def get_airline_review_texts():
    """Returns formatted airline review texts for vector store embedding."""
    reviews = load_airline_reviews_from_kaggle()
    review_texts = []
    
    for review in reviews:
        # Get all available fields and just append them as simple text
        text_parts = []
        
        # Add title if available
        if review.get('title'):
            text_parts.append(f"Title: {review['title']}")
        
        # Add main review text
        if review.get('text'):
            text_parts.append(f"Review: {review['text']}")
        
        # Add rating if available
        if review.get('rating'):
            text_parts.append(f"Rating: {review['rating']}")
        
        # Add travel date if available
        if review.get('travel_date'):
            text_parts.append(f"Travel Date: {review['travel_date']}")
        
        # Add published date if available
        if review.get('published_date'):
            text_parts.append(f"Published: {review['published_date']}")
        
        # Add helpful votes if available
        if review.get('helpful_votes'):
            text_parts.append(f"Helpful Votes: {review['helpful_votes']}")
        
        # Add language if available
        if review.get('lang'):
            text_parts.append(f"Language: {review['lang']}")
        
        # Join all parts with ". "
        text = ". ".join(text_parts)
        review_texts.append(text)
    
    logger.info(f"Generated {len(review_texts)} airline review text embeddings")
    return review_texts


def load_airline_reviews_to_couchbase(
    cluster, 
    bucket_name: str, 
    scope_name: str, 
    collection_name: str, 
    embeddings, 
    index_name: str
):
    """Load airline reviews from Kaggle into the target collection with embeddings."""
    try:
        # Check if data already exists
        count_query = f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        count_result = cluster.query(count_query)
        count_row = list(count_result)[0]
        existing_count = count_row['count']
        
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing documents in collection, skipping data load")
            return
        
        # Get the airline reviews texts
        review_texts = get_airline_review_texts()
        
        # Setup vector store for the target collection
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            embedding=embeddings,
            index_name=index_name,
        )
        
        # Add review texts to vector store with batch processing
        logger.info(f"Loading {len(review_texts)} airline review embeddings to {bucket_name}.{scope_name}.{collection_name}")
        
        # Process in batches to avoid memory issues and respect Capella AI batch limit
        batch_size = 25  # Well below Capella AI embedding model limit of 32
        for i in range(0, len(review_texts), batch_size):
            batch = review_texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(review_texts) - 1)//batch_size + 1}")
            
            vector_store.add_texts(
                texts=batch,
                batch_size=batch_size
            )
        
        logger.info(f"Successfully loaded {len(review_texts)} airline review embeddings to vector store")
        
    except Exception as e:
        logger.error(f"Error loading airline reviews to Couchbase: {str(e)}")
        raise


def load_airline_reviews():
    """Simple function to load airline reviews - called by main.py."""
    try:
        cluster = get_cluster_connection()
        if not cluster:
            raise ConnectionError("Could not connect to Couchbase cluster")
        
        # Get environment variables
        bucket_name = os.getenv("CB_BUCKET_NAME", "vector-search-testing")
        scope_name = os.getenv("CB_SCOPE", "agentc_data")
        collection_name = os.getenv("CB_COLLECTION", "airline_reviews")
        index_name = os.getenv("CB_INDEX", "airline_reviews_index")
        
        # For now, let's just log that we're loading the data
        # The actual embeddings will be set up by the main.py file
        logger.info("Loading airline reviews data...")
        
        # Load the data to verify it works
        reviews = load_airline_reviews_from_kaggle()
        logger.info(f"Successfully loaded {len(reviews)} airline reviews")
        
        return reviews
        
    except Exception as e:
        logger.error(f"Error in load_airline_reviews: {str(e)}")
        raise


if __name__ == "__main__":
    # Test the data loading
    print("Testing airline reviews data loading...")
    reviews = load_airline_reviews()
    print(f"Successfully loaded {len(reviews)} airline reviews")
    
    if reviews:
        print("\nFirst review:")
        print(reviews[0]) 