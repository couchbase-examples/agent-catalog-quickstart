"""
Landmark Search Tool - AgentC Integration for Travel Sample Landmarks

This tool demonstrates:
- Using AgentC for tool registration (@agentc.catalog.tool)
- Semantic search using the vector store for landmark data from travel-sample.inventory.landmark
- Integration with Couchbase travel-sample database
"""

import logging
import os
import sys
from datetime import timedelta

import dotenv

# Import shared modules using project root discovery
def find_project_root():
    """Find the project root by looking for the shared directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Look for the shared directory as the definitive marker
        shared_path = os.path.join(current, 'shared')
        if os.path.exists(shared_path) and os.path.isdir(shared_path):
            return current
        current = os.path.dirname(current)
    return None

# Add project root to Python path
project_root = find_project_root()
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import agentc after path is set
import agentc
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

        # Perform pure semantic search using the raw query
        retriever = index.as_retriever(similarity_top_k=limit)
        raw_query = (query or "").strip()
        nodes = retriever.retrieve(raw_query)

        # Log search info
        logger.info(f"Search query: '{raw_query}' found {len(nodes)} results")

        # Format results (minimal, straightforward)
        if not nodes:
            return f"No landmarks found matching your query: '{raw_query}'"

        import re

        # Precompile regex patterns for robust keyed-field extraction (non-greedy until next key or end)
        FIELD_KEYS = [
            "Address:", "Directions:", "Phone:", "Website:", "Hours:", "Price:",
            "Activity type:", "Type:", "State:", "Coordinates:", "ID:"
        ]
        NEXT_KEY_PATTERN = r"\s+(?=" + r"|".join([re.escape(k) for k in FIELD_KEYS]) + r"|$)"

        PAT_FIRSTLINE_1 = re.compile(
            r"^(?P<title>.+?)\s*\((?P<name>[^)]+)\)\s+in\s+(?P<city>[^,]+),\s*(?P<country>[^.]+)\.",
            re.IGNORECASE | re.DOTALL,
        )
        PAT_FIRSTLINE_2 = re.compile(
            r"^(?P<city1>[^/]+)/(?P<area>[^ ]+)\s*\((?P<name2>[^)]+)\)\s+in\s+(?P<city2>[^,]+),\s*(?P<country2>[^.]+)\.",
            re.IGNORECASE | re.DOTALL,
        )

        def extract_field(pattern_label: str, text: str) -> str:
            pat = re.compile(rf"{pattern_label}\s*(.*?){NEXT_KEY_PATTERN}", re.IGNORECASE | re.DOTALL)
            m = pat.search(text)
            return m.group(1).strip() if m else ""

        def parse_text_content(content: str) -> dict:
            data = {
                "name": "",
                "city": "",
                "country": "",
                "address": "",
                "directions": "",
                "phone": "",
                "url": "",
                "hours": "",
                "price": "",
                "activity": "",
                "state": "",
                "lat": "",
                "lon": "",
                "doc_id": "",
                "description": "",
            }
            if not content:
                return data

            # Normalize whitespace but keep full content
            txt = re.sub(r"\s+", " ", content.strip())

            # First line extraction
            m1 = PAT_FIRSTLINE_1.search(txt)
            if m1:
                data["name"] = m1.group("name").strip()
                data["city"] = m1.group("city").strip()
                data["country"] = m1.group("country").strip()
            else:
                m2 = PAT_FIRSTLINE_2.search(txt)
                if m2:
                    data["name"] = m2.group("name2").strip()
                    data["city"] = m2.group("city2").strip()
                    data["country"] = m2.group("country2").strip()
                else:
                    # Fallback: first line as name
                    first_line = content.split('\n')[0].strip()
                    if first_line and len(first_line) < 200:
                        data["name"] = first_line

            # Keyed fields
            data["description"] = extract_field("Description:", txt)
            data["address"] = extract_field("Address:", txt)
            data["directions"] = extract_field("Directions:", txt)
            data["phone"] = extract_field("Phone:", txt)
            data["url"] = extract_field("Website:", txt)
            data["hours"] = extract_field("Hours:", txt)
            data["price"] = extract_field("Price:", txt)
            data["activity"] = extract_field("Activity type:", txt)
            data["state"] = extract_field("State:", txt)

            # Coordinates
            mcoord = re.search(r"Coordinates:\s*(?P<lat>-?\d+(?:\.\d+)?),\s*(?P<lon>-?\d+(?:\.\d+)?)(?:\s*\(accuracy:[^)]+\))?",
                               txt, re.IGNORECASE)
            if mcoord:
                data["lat"] = mcoord.group("lat")
                data["lon"] = mcoord.group("lon")

            # ID
            mid = re.search(r"ID:\s*(?P<docid>\d+)", txt, re.IGNORECASE)
            if mid:
                data["doc_id"] = mid.group("docid")

            return data

        results = []
        seen = set()  # for deduplication by name+city
        MAX_DESC_LEN = 800  # soft cap for description length
        sources_payload = []
        for i, node in enumerate(nodes, 1):
            metadata = node.metadata or {}
            content = node.text or ""

            # Prefer metadata; then parse from content
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

            parsed = parse_text_content(content)
            # Fill blanks from parsed content
            name = name or parsed["name"]
            city = city or parsed["city"]
            country = country or parsed["country"]
            address = address or parsed["address"]
            phone = phone or parsed["phone"]
            url = url or parsed["url"]
            hours = hours or parsed["hours"]
            price = price or parsed["price"]
            activity = activity or parsed["activity"]
            state = state or parsed["state"]
            description = parsed["description"] or content.strip()

            # Use fallbacks for essential fields
            name = name or "Landmark"
            city = city or "Unknown City"
            country = country or "Unknown Country"
            activity = activity or "see"

            # Deduplicate by normalized name + city
            dedupe_key = f"{name.strip().lower()}|{city.strip().lower()}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            # Build result with robust formatting
            result_lines = [f"{i}. **{name}**"]
            result_lines.append(f"   📍 Location: {city}, {country}")
            if state and state.strip() and state.lower() != "unknown":
                result_lines.append(f"   🗺️ State: {state}")
            result_lines.append(f"   🎯 Activity: {activity.title()}")
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
            if description:
                # Keep full description, normalized
                description = re.sub(r"\s+", " ", description).strip()
                if len(description) > MAX_DESC_LEN:
                    description = description[:MAX_DESC_LEN].rstrip() + ".."
                result_lines.append(f"   📝 Description: {description}")

            results.append("\n".join(result_lines))

            # Append structured source for evaluator/observability
            doc_id = metadata.get("landmark_id") or metadata.get("doc_id") or getattr(node, "id_", None)
            sources_payload.append({
                "name": name,
                "city": city,
                "country": country,
                "state": state,
                "activity": activity,
                "address": address,
                "url": url,
                "hours": hours,
                "price": price,
                "doc_id": doc_id,
            })

        display_text = f"Found {len(results)} landmarks matching '{raw_query}':\n\n" + "\n\n".join(results)

        # Always return pretty-printed text for agent Observation/Final Answer
        return display_text

    except Exception as e:
        logger.error(f"Error searching landmarks: {e}")
        return f"Error searching landmarks: {str(e)}"
