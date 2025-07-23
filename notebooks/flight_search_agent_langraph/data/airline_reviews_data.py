#!/usr/bin/env python3
"""
Airline reviews data module for the flight search agent demo.
Downloads and processes Indian Airlines Customer Reviews dataset from Kaggle.
"""

import logging
import os
from datetime import timedelta

import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
from couchbase.options import ClusterOptions
import dotenv
import pandas as pd
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from tqdm import tqdm

# Import kagglehub only when needed to avoid import errors during indexing
try:
    import kagglehub
except ImportError:
    kagglehub = None

# Load environment variables
dotenv.load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirlineReviewsDataManager:
    """Manages airline reviews data loading, processing, and embedding."""
    
    def __init__(self):
        self._raw_data_cache = None
        self._processed_texts_cache = None
        
    def load_raw_data(self):
        """Load raw airline reviews data from Kaggle dataset (with caching)."""
        if self._raw_data_cache is not None:
            return self._raw_data_cache
            
        try:
            if kagglehub is None:
                raise ImportError("kagglehub is not available")
            
            # Download the dataset from Kaggle
            logger.info("Downloading Indian Airlines Customer Reviews dataset from Kaggle...")
            path = kagglehub.dataset_download("jagathratchakan/indian-airlines-customer-reviews")

            # Find the CSV file
            csv_file = None
            for file in os.listdir(path):
                if file.endswith(".csv"):
                    csv_file = os.path.join(path, file)
                    break

            if not csv_file:
                msg = "No CSV file found in downloaded dataset"
                raise FileNotFoundError(msg)

            # Load the CSV file
            logger.info(f"Loading reviews from {csv_file}")
            df = pd.read_csv(csv_file)

            # Convert DataFrame to list of dictionaries and cache
            self._raw_data_cache = df.to_dict("records")
            logger.info(f"Loaded {len(self._raw_data_cache)} airline reviews from Kaggle dataset")
            return self._raw_data_cache

        except Exception as e:
            logger.exception(f"Error loading airline reviews from Kaggle: {e!s}")
            raise

    def process_to_texts(self):
        """Process raw data into formatted text strings for embedding (with caching)."""
        if self._processed_texts_cache is not None:
            return self._processed_texts_cache
            
        reviews = self.load_raw_data()
        review_texts = []

        for review in reviews:
            # Get all available fields and format them as text
            text_parts = []

            # Add airline name if available
            if review.get("AirLine_Name"):
                text_parts.append(f"Airline: {review['AirLine_Name']}")

            # Add title if available
            if review.get("Title"):
                text_parts.append(f"Title: {review['Title']}")

            # Add main review text
            if review.get("Review"):
                text_parts.append(f"Review: {review['Review']}")

            # Add rating if available
            if review.get("Rating - 10"):
                text_parts.append(f"Rating: {review['Rating - 10']}/10")

            # Add reviewer name if available
            if review.get("Name"):
                text_parts.append(f"Reviewer: {review['Name']}")

            # Add date if available
            if review.get("Date"):
                text_parts.append(f"Date: {review['Date']}")

            # Add recommendation if available
            if review.get("Recommond"):
                text_parts.append(f"Recommended: {review['Recommond']}")

            # Join all parts with ". "
            text = ". ".join(text_parts)
            review_texts.append(text)

        # Cache the processed texts
        self._processed_texts_cache = review_texts
        logger.info(f"Processed {len(review_texts)} airline reviews into text format")
        return review_texts

    def load_to_vector_store(self, cluster, bucket_name: str, scope_name: str, 
                           collection_name: str, embeddings, index_name: str):
        """Load airline reviews into Couchbase vector store with embeddings."""
        try:
            # Check if data already exists
            count_query = f"SELECT COUNT(*) as count FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            count_result = cluster.query(count_query)
            count_row = next(iter(count_result))
            existing_count = count_row["count"]

            if existing_count > 0:
                logger.info(f"Found {existing_count} existing documents in collection, skipping data load")
                return

            # Get the processed review texts
            review_texts = self.process_to_texts()

            # Setup vector store for the target collection
            vector_store = CouchbaseVectorStore(
                cluster=cluster,
                bucket_name=bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )

            # Add review texts to vector store with batch processing and progress bar
            logger.info(f"Loading {len(review_texts)} airline review embeddings to {bucket_name}.{scope_name}.{collection_name}")

            # Process in batches to avoid memory issues and respect Capella AI batch limit
            batch_size = 10  # Conservative batch size for stability
            total_batches = (len(review_texts) + batch_size - 1) // batch_size
            
            with tqdm(total=len(review_texts), desc="Loading airline reviews", unit="reviews") as pbar:
                for i in range(0, len(review_texts), batch_size):
                    batch_num = i // batch_size + 1
                    batch = review_texts[i:i + batch_size]
                    
                    # Add this batch to vector store
                    vector_store.add_texts(texts=batch, batch_size=len(batch))
                    
                    # Update progress bar
                    pbar.update(len(batch))
                    pbar.set_postfix(batch=f"{batch_num}/{total_batches}")

            logger.info(f"Successfully loaded {len(review_texts)} airline review embeddings to vector store")

        except Exception as e:
            logger.exception(f"Error loading airline reviews to Couchbase: {e!s}")
            raise


# Global instance for reuse
_data_manager = AirlineReviewsDataManager()


def get_airline_review_texts():
    """Get processed airline review texts (uses global cached instance)."""
    return _data_manager.process_to_texts()


def load_airline_reviews_from_kaggle():
    """Load raw airline reviews data from Kaggle (uses global cached instance)."""
    return _data_manager.load_raw_data()


def load_airline_reviews_to_couchbase(cluster, bucket_name: str, scope_name: str, 
                                     collection_name: str, embeddings, index_name: str):
    """Load airline reviews into Couchbase vector store (uses global cached instance)."""
    return _data_manager.load_to_vector_store(
        cluster, bucket_name, scope_name, collection_name, embeddings, index_name
    )


def load_airline_reviews():
    """Simple function to load airline reviews - called by main.py."""
    try:
        # Just return the processed texts for embedding
        # This eliminates the need for separate cluster connection here
        logger.info("Loading airline reviews data...")
        reviews = _data_manager.process_to_texts()
        logger.info(f"Successfully loaded {len(reviews)} airline reviews")
        return reviews

    except Exception as e:
        logger.exception(f"Error in load_airline_reviews: {e!s}")
        raise


if __name__ == "__main__":
    # Test the data loading
    reviews = load_airline_reviews()
    print(f"Successfully loaded {len(reviews)} airline reviews")
    if reviews:
        print(f"First review: {reviews[0][:200]}...")
