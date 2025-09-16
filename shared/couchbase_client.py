#!/usr/bin/env python3
"""
Shared Couchbase Client

Universal Couchbase client with superset of all agent implementations:
- Hotel Agent (LangChain): Collection management, data clearing, vector store setup
- Flight Agent (LangGraph): Bucket creation, scope clearing, comprehensive setup
- Landmark Agent (LlamaIndex): Enhanced timeouts, specialized data loading

Provides consistent database operations across all agent frameworks.
"""

import json
import logging
import os
import time
from datetime import timedelta
from typing import Optional

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import KeyspaceNotFoundException, InternalServerFailureException
from couchbase.management.buckets import BucketType, CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions

logger = logging.getLogger(__name__)


class CouchbaseClient:
    """Universal Couchbase client for all database operations across agent frameworks."""

    def __init__(
        self,
        conn_string: str,
        username: str,
        password: str,
        bucket_name: str,
        wan_profile: bool = True,
        timeout_seconds: int = 20,
    ):
        """
        Initialize Couchbase client with connection details.
        
        Args:
            conn_string: Couchbase connection string
            username: Couchbase username
            password: Couchbase password
            bucket_name: Target bucket name
            wan_profile: Whether to use WAN development profile for remote clusters
            timeout_seconds: Connection timeout in seconds
        """
        self.conn_string = conn_string
        self.username = username
        self.password = password
        self.bucket_name = bucket_name
        self.wan_profile = wan_profile
        self.timeout_seconds = timeout_seconds
        self.cluster = None
        self.bucket = None
        self._collections = {}

    def connect(self):
        """Establish connection to Couchbase cluster."""
        try:
            auth = PasswordAuthenticator(self.username, self.password)
            options = ClusterOptions(auth)

            # Use WAN profile for better timeout handling with remote clusters
            if self.wan_profile:
                options.apply_profile("wan_development")

            self.cluster = Cluster(self.conn_string, options)
            self.cluster.wait_until_ready(timedelta(seconds=self.timeout_seconds))
            logger.info("✅ Successfully connected to Couchbase")
            return self.cluster
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Couchbase: {e!s}")

    def setup_bucket(self, create_if_missing: bool = True):
        """Setup bucket - connect to existing or create if missing."""
        try:
            # Ensure cluster connection
            if not self.cluster:
                self.connect()

            # Try to connect to existing bucket
            try:
                self.bucket = self.cluster.bucket(self.bucket_name)
                logger.info(f"✅ Connected to existing bucket '{self.bucket_name}'")
                return self.bucket
            except Exception as e:
                logger.info(f"⚠️ Bucket '{self.bucket_name}' not accessible: {e}")

            # Create bucket if missing and allowed
            if create_if_missing:
                logger.info(f"🔧 Creating bucket '{self.bucket_name}'...")
                bucket_settings = CreateBucketSettings(
                    name=self.bucket_name,
                    bucket_type=BucketType.COUCHBASE,
                    ram_quota_mb=1024,
                    flush_enabled=True,
                    num_replicas=0,
                )
                self.cluster.buckets().create_bucket(bucket_settings)
                time.sleep(5)  # Allow bucket creation to complete
                self.bucket = self.cluster.bucket(self.bucket_name)
                logger.info(f"✅ Bucket '{self.bucket_name}' created successfully")
                return self.bucket
            else:
                raise RuntimeError(f"❌ Bucket '{self.bucket_name}' not found and creation disabled")

        except Exception as e:
            raise RuntimeError(f"❌ Error setting up bucket: {e!s}")

    def setup_collection(
        self,
        scope_name: str,
        collection_name: str,
        clear_existing_data: bool = True,
        create_primary_index: bool = True,
    ):
        """
        Setup collection with comprehensive options.
        
        Args:
            scope_name: Target scope name
            collection_name: Target collection name
            clear_existing_data: Whether to clear data if collection exists
            create_primary_index: Whether to create primary index
            
        Returns:
            Collection object
        """
        try:
            # Ensure bucket setup
            if not self.bucket:
                self.setup_bucket()

            bucket_manager = self.bucket.collections()

            # Setup scope
            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                logger.info(f"🔧 Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                logger.info(f"✅ Scope '{scope_name}' created successfully")

            # Setup collection
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name
                and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if collection_exists:
                if clear_existing_data:
                    logger.info(f"🗑️ Collection '{collection_name}' exists, clearing data...")
                    self.clear_collection_data(scope_name, collection_name)
                else:
                    logger.info(f"ℹ️ Collection '{collection_name}' exists, keeping existing data")
            else:
                logger.info(f"🔧 Creating collection '{collection_name}'...")
                bucket_manager.create_collection(scope_name, collection_name)
                logger.info(f"✅ Collection '{collection_name}' created successfully")

            time.sleep(3)  # Allow operations to complete

            # Create primary index if requested
            if create_primary_index:
                try:
                    self.cluster.query(
                        f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
                    ).execute()
                    logger.info("✅ Primary index created successfully")
                except Exception as e:
                    logger.warning(f"⚠️ Error creating primary index: {e}")

            # Cache and return collection
            collection_key = f"{scope_name}.{collection_name}"
            collection = self.bucket.scope(scope_name).collection(collection_name)
            self._collections[collection_key] = collection

            logger.info(f"✅ Collection setup complete: {scope_name}.{collection_name}")
            return collection

        except Exception as e:
            raise RuntimeError(f"❌ Error setting up collection: {e!s}")

    def clear_collection_data(self, scope_name: str, collection_name: str, verify_cleared: bool = True):
        """
        Clear all data from a collection with optional verification.
        
        Args:
            scope_name: Target scope name
            collection_name: Target collection name
            verify_cleared: Whether to verify collection is empty after clearing
        """
        try:
            logger.info(f"🗑️ Clearing data from {self.bucket_name}.{scope_name}.{collection_name}...")

            # Use N1QL to delete all documents
            delete_query = f"DELETE FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
            result = self.cluster.query(delete_query)
            list(result)  # Execute the query

            # Wait for deletion to propagate
            time.sleep(2)

            # Verify collection is empty if requested
            if verify_cleared:
                count_query = f"SELECT COUNT(*) as count FROM `{self.bucket_name}`.`{scope_name}`.`{collection_name}`"
                count_result = self.cluster.query(count_query)
                count_row = list(count_result)[0]
                remaining_count = count_row["count"]

                if remaining_count == 0:
                    logger.info(f"✅ Collection cleared successfully, {remaining_count} documents remaining")
                else:
                    logger.warning(f"⚠️ Collection clear incomplete, {remaining_count} documents remaining")

        except KeyspaceNotFoundException:
            logger.info(f"ℹ️ Collection {self.bucket_name}.{scope_name}.{collection_name} doesn't exist, nothing to clear")
            # This is actually success - clearing non-existent collection is successful
        except Exception as e:
            logger.warning(f"⚠️ Error clearing collection data: {e}")
            # Continue anyway - collection might not exist or be empty

    def clear_scope(self, scope_name: str):
        """Clear all collections in the specified scope."""
        try:
            # Ensure bucket setup
            if not self.bucket:
                self.setup_bucket()

            logger.info(f"🗑️ Clearing scope: {self.bucket_name}.{scope_name}")
            bucket_manager = self.bucket.collections()
            scopes = bucket_manager.get_all_scopes()

            # Find the target scope
            target_scope = None
            for scope in scopes:
                if scope.name == scope_name:
                    target_scope = scope
                    break

            if not target_scope:
                logger.info(f"ℹ️ Scope '{self.bucket_name}.{scope_name}' does not exist, nothing to clear")
                return

            # Clear all collections in the scope
            for collection in target_scope.collections:
                try:
                    self.clear_collection_data(scope_name, collection.name, verify_cleared=False)
                    logger.info(f"✅ Cleared collection: {self.bucket_name}.{scope_name}.{collection.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not clear collection {collection.name}: {e}")

            logger.info(f"✅ Completed clearing scope: {self.bucket_name}.{scope_name}")

        except Exception as e:
            logger.warning(f"❌ Could not clear scope {self.bucket_name}.{scope_name}: {e}")

    def get_collection(self, scope_name: str, collection_name: str, auto_create: bool = False):
        """
        Get a collection object with optional auto-creation.
        
        Args:
            scope_name: Target scope name
            collection_name: Target collection name
            auto_create: Whether to create collection if it doesn't exist
            
        Returns:
            Collection object
        """
        collection_key = f"{scope_name}.{collection_name}"
        
        if collection_key not in self._collections:
            if auto_create:
                self.setup_collection(scope_name, collection_name, clear_existing_data=False)
            else:
                # Just cache the collection reference
                if not self.bucket:
                    self.setup_bucket()
                self._collections[collection_key] = self.bucket.scope(scope_name).collection(collection_name)
                
        return self._collections[collection_key]

    def setup_vector_search_index(self, index_definition: dict, scope_name: str):
        """Setup vector search index for the specified scope."""
        try:
            if not self.bucket:
                raise RuntimeError("❌ Bucket not initialized. Call setup_bucket first.")

            scope_index_manager = self.bucket.scope(scope_name).search_indexes()
            index_name = index_definition["name"]

            # Try to get existing indexes, but handle case when no indexes exist yet
            try:
                existing_indexes = scope_index_manager.get_all_indexes()
                index_exists = index_name in [index.name for index in existing_indexes]
            except InternalServerFailureException as get_error:
                # When no indexes exist, get_all_indexes() throws InternalServerFailureException
                logger.info(f"🔍 No existing indexes found (empty scope): {get_error}")
                index_exists = False

            if not index_exists:
                logger.info(f"🔧 Creating vector search index '{index_name}'...")
                search_index = SearchIndex.from_json(index_definition)
                scope_index_manager.upsert_index(search_index)
                logger.info(f"✅ Vector search index '{index_name}' created successfully")
            else:
                logger.info(f"ℹ️ Vector search index '{index_name}' already exists")
        except Exception as e:
            raise RuntimeError(f"❌ Error setting up vector search index: {e!s}")

    def load_index_definition(self, index_file_path: str = "agentcatalog_index.json") -> Optional[dict]:
        """Load vector search index definition from JSON file."""
        try:
            with open(index_file_path) as file:
                index_definition = json.load(file)
            logger.info(f"✅ Loaded vector search index definition from {index_file_path}")
            return index_definition
        except FileNotFoundError:
            logger.warning(f"⚠️ {index_file_path} not found, continuing without vector search index...")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Error parsing index definition JSON: {e!s}")
            return None
        except Exception as e:
            logger.warning(f"⚠️ Error loading index definition: {e!s}")
            return None

    def setup_vector_store_langchain(
        self,
        scope_name: str,
        collection_name: str,
        index_name: str,
        embeddings,
        data_loader_func=None,
        **loader_kwargs,
    ):
        """
        Setup LangChain CouchbaseVectorStore with optional data loading.
        
        Args:
            scope_name: Target scope name
            collection_name: Target collection name
            index_name: Vector search index name
            embeddings: Embeddings model instance
            data_loader_func: Optional function to load data (e.g., load_hotel_data_to_couchbase)
            **loader_kwargs: Additional arguments for data loader function
            
        Returns:
            CouchbaseVectorStore instance
        """
        try:
            from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore

            # Load data if loader function provided
            if data_loader_func:
                logger.info("🔄 Loading data into vector store...")
                data_loader_func(
                    cluster=self.cluster,
                    bucket_name=self.bucket_name,
                    scope_name=scope_name,
                    collection_name=collection_name,
                    embeddings=embeddings,
                    index_name=index_name,
                    **loader_kwargs,
                )
                logger.info("✅ Data loaded into vector store successfully")

            # Create LangChain vector store instance
            vector_store = CouchbaseSearchVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                embedding=embeddings,
                index_name=index_name,
            )

            logger.info(f"✅ LangChain vector store setup complete: {self.bucket_name}.{scope_name}.{collection_name}")
            return vector_store

        except Exception as e:
            raise RuntimeError(f"❌ Error setting up LangChain vector store: {e!s}")

    def setup_vector_store_llamaindex(
        self,
        scope_name: str,
        collection_name: str,
        index_name: str,
    ):
        """
        Setup LlamaIndex CouchbaseSearchVectorStore.
        
        Args:
            scope_name: Target scope name
            collection_name: Target collection name
            index_name: Vector search index name
            
        Returns:
            CouchbaseSearchVectorStore instance
        """
        try:
            from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

            # Create LlamaIndex vector store instance
            vector_store = CouchbaseSearchVectorStore(
                cluster=self.cluster,
                bucket_name=self.bucket_name,
                scope_name=scope_name,
                collection_name=collection_name,
                index_name=index_name,
            )

            logger.info(f"✅ LlamaIndex vector store setup complete: {self.bucket_name}.{scope_name}.{collection_name}")
            return vector_store

        except Exception as e:
            raise RuntimeError(f"❌ Error setting up LlamaIndex vector store: {e!s}")

    def disconnect(self):
        """Clean disconnect from Couchbase cluster."""
        try:
            if self.cluster:
                # Clear cached collections
                self._collections.clear()
                self.bucket = None
                self.cluster = None
                logger.info("✅ Disconnected from Couchbase")
        except Exception as e:
            logger.warning(f"⚠️ Error during disconnect: {e}")

    def __enter__(self):
        """Context manager entry - establish connection."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean disconnect."""
        self.disconnect()


def create_couchbase_client(
    conn_string: str = None,
    username: str = None,
    password: str = None,
    bucket_name: str = None,
    wan_profile: bool = True,
    timeout_seconds: int = 20,
) -> CouchbaseClient:
    """
    Factory function to create CouchbaseClient with environment variable defaults.
    
    Args:
        conn_string: Couchbase connection string (defaults to CB_CONN_STRING env var)
        username: Couchbase username (defaults to CB_USERNAME env var)
        password: Couchbase password (defaults to CB_PASSWORD env var)
        bucket_name: Target bucket name (defaults to CB_BUCKET env var)
        wan_profile: Whether to use WAN development profile
        timeout_seconds: Connection timeout in seconds
        
    Returns:
        CouchbaseClient instance
    """
    return CouchbaseClient(
        conn_string=conn_string or os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        username=username or os.getenv("CB_USERNAME", "Administrator"),
        password=password or os.getenv("CB_PASSWORD", "password"),
        bucket_name=bucket_name or os.getenv("CB_BUCKET", "travel-sample"),
        wan_profile=wan_profile,
        timeout_seconds=timeout_seconds,
    )
