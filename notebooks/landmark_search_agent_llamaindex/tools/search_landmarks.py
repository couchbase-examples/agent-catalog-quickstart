"""
Landmark Search Tool - AgentC Integration for Travel Sample Landmarks

This tool demonstrates:
- Using AgentC for tool registration (@agentc.catalog.tool)
- Semantic search using the vector store for landmark data from travel-sample.inventory.landmark
- Integration with Couchbase travel-sample database
"""

import logging
import os
from datetime import timedelta

import agentc
import dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from llama_index.core import Settings, VectorStoreIndex
from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore

# Setup logging
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


def get_cluster_connection():
    """Get a fresh cluster connection for each request."""
    try:
        auth = PasswordAuthenticator(
            username=os.environ["CB_USERNAME"],
            password=os.environ["CB_PASSWORD"],
        )
        options = ClusterOptions(auth)
        options.apply_profile("wan_development")

        conn_string = os.environ["CB_CONN_STRING"]
        cluster = Cluster(conn_string, options)
        cluster.wait_until_ready(timedelta(seconds=20))

        return cluster

    except Exception as e:
        logger.exception(f"Failed to connect to Couchbase: {e}")
        raise


@agentc.catalog.tool()
def search_landmarks(query: str, limit: int = 5) -> str:
    """
    Search for landmark information using semantic vector search.

    This tool searches through landmark data from the travel-sample database,
    including attractions, monuments, museums, and points of interest.

    Args:
        query (str): The search query describing what landmarks to find
        limit (int): Maximum number of results to return (default: 5)

    Returns:
        str: Formatted landmark search results with details like name, location,
             description, activities, and practical information
    """
    try:
        # Ensure limit is an integer (in case agent passes it as string)
        limit = int(limit) if isinstance(limit, str) else limit
        limit = max(1, min(limit, 20))  # Clamp between 1 and 20
        # Get cluster connection
        cluster = get_cluster_connection()

        # Create vector store using the landmark collection
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=os.environ.get("CB_BUCKET", "travel-sample"),
            scope_name=os.environ.get("CB_SCOPE", "agentc_data"),
            collection_name=os.environ.get("CB_COLLECTION", "landmark_data"),
            index_name=os.environ.get("CB_INDEX", "landmark_data_index"),
        )

        # Create index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=Settings.embed_model
        )

        # Perform semantic search using retriever (more reliable than query engine)
        retriever = index.as_retriever(similarity_top_k=limit)
        nodes = retriever.retrieve(query)

        # Format results
        if not nodes:
            return f"No landmarks found matching your query: '{query}'"

        results = []
        for i, node in enumerate(nodes, 1):
            # Extract metadata - with fallback to text parsing
            metadata = node.metadata or {}
            content = node.text or ""

            # Try to extract information from metadata first, then from text content
            name = metadata.get("name", "")
            city = metadata.get("city", "")
            country = metadata.get("country", "")
            address = metadata.get("address", "")
            phone = metadata.get("phone", "")
            url = metadata.get("url", metadata.get("website", ""))
            activity = metadata.get("activity", "")
            state = metadata.get("state", "")
            hours = metadata.get("hours", "")
            price = metadata.get("price", "")

            # If metadata is missing key info, try to parse from text content
            if not name or name == "Unknown":
                # Extract name from text patterns - more flexible
                import re

                # Pattern 1: "Name (Alternative) in City, Country"
                name_match = re.search(
                    r"^([^(]+?)(?:\s*\([^)]+\))?\s+in\s+([^,]+),\s*(.+?)(?:\.|$)", content
                )
                if name_match:
                    name = name_match.group(1).strip()
                    if not city or city == "Unknown":
                        city = name_match.group(2).strip()
                    if not country or country == "Unknown":
                        country = name_match.group(3).strip()
                
                # Pattern 2: Extract from first line if no "in" pattern
                elif not name_match and content:
                    first_line = content.split('\n')[0].strip()
                    if first_line and len(first_line) < 100:  # Reasonable name length
                        name = first_line

            # Extract additional info from text - simplified and robust
            if not address and "Address:" in content:
                addr_match = re.search(r"Address:\s*([^.\n]+(?:\.[^.\n]*)*?)(?:\s+Phone:|\s+Website:|\s+Hours:|$)", content, re.IGNORECASE)
                if addr_match:
                    address = addr_match.group(1).strip()

            if not phone and "Phone:" in content:
                phone_match = re.search(r"Phone:\s*([+\d\s\-()]+)", content, re.IGNORECASE)
                if phone_match:
                    phone = phone_match.group(1).strip()

            if not url and ("Website:" in content or "http" in content):
                url_match = re.search(r'(?:Website:\s*)?(https?://[^\s<>"]+)', content, re.IGNORECASE)
                if url_match:
                    url = url_match.group(1).strip()

            if not hours and "Hours:" in content:
                hours_match = re.search(r"Hours:\s*([^.\n]+(?:\.[^.\n]*)*?)(?:\s+Price:|\s+Phone:|\s+Website:|$)", content, re.IGNORECASE)
                if hours_match:
                    hours = hours_match.group(1).strip()

            if not price and ("Price:" in content or "€" in content or "$" in content or "£" in content):
                price_match = re.search(r"(?:Price:\s*)?([€$£]\d+[^.\n]*)", content, re.IGNORECASE)
                if price_match:
                    price = price_match.group(1).strip()

            if not activity or activity == "General":
                if "Activity type:" in content:
                    activity_match = re.search(r"Activity type:\s*([^\n.]+)", content, re.IGNORECASE)
                    if activity_match:
                        activity = activity_match.group(1).strip()
                else:
                    # Infer activity from content
                    content_lower = content.lower()
                    if any(word in content_lower for word in ["museum", "gallery", "art", "cathedral", "monument", "attraction"]):
                        activity = "See"
                    elif any(word in content_lower for word in ["restaurant", "dining", "food", "cuisine", "eat", "cafe"]):
                        activity = "Eat"
                    elif any(word in content_lower for word in ["trail", "hiking", "park", "sport", "recreation", "activity"]):
                        activity = "Do"

            # Use fallbacks for essential fields
            name = name or "Landmark"
            city = city or "Unknown City"
            country = country or "Unknown Country"
            activity = activity or "see"

            # Build result with robust formatting
            result_lines = [f"{i}. **{name}**"]
            result_lines.append(f"   📍 Location: {city}, {country}")

            # Add state if available
            if state and state.strip() and state != "Unknown":
                result_lines.append(f"   🗺️ State: {state}")

            result_lines.append(f"   🎯 Activity: {activity.title()}")

            # Only show fields that have actual meaningful data
            if address and address.strip() and address.lower() not in ["unknown", "none"]:
                result_lines.append(f"   🏠 Address: {address}")
            if phone and phone.strip() and phone.lower() not in ["unknown", "none", "null"]:
                result_lines.append(f"   📞 Phone: {phone}")
            if url and url.strip() and url.lower() not in ["unknown", "none", "null"]:
                result_lines.append(f"   🌐 Website: {url}")
            if hours and hours.strip() and hours.lower() not in ["unknown", "none", "null"]:
                result_lines.append(f"   🕒 Hours: {hours}")
            if price and price.strip() and price.lower() not in ["unknown", "none", "null"]:
                result_lines.append(f"   💰 Price: {price}")

            if content:
                # Use full content as description - no fragile regex parsing
                description = content.strip()
                
                # Basic cleanup only - preserve all content
                description = re.sub(r'\s+', ' ', description)  # Normalize whitespace
                description = re.sub(r'^\s*Landmark\s*[:.]?\s*', '', description, flags=re.IGNORECASE)  # Remove "Landmark:" prefix
                
                if description:
                    result_lines.append(f"   📝 Description: {description}")

            results.append("\n".join(result_lines))

        return f"Found {len(results)} landmarks matching '{query}':\n\n" + "\n\n".join(results)

    except Exception as e:
        logger.error(f"Error searching landmarks: {e}")
        return f"Error searching landmarks: {str(e)}"
