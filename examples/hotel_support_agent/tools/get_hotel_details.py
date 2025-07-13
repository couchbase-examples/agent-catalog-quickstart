import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv
import os
from datetime import timedelta
from data.hotel_data import get_detailed_hotel_data

dotenv.load_dotenv()


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
        print(f"Could not connect to Couchbase cluster: {str(e)}")
        return None


@agentc.catalog.tool
def get_hotel_details(hotel_name: str) -> str:
    """Get comprehensive details for a specific hotel by name including amenities, pricing, and contact information."""
    try:
        # Get detailed hotel data from the data module first (primary source)
        detailed_hotels = get_detailed_hotel_data()
        
        # Try to find the hotel by normalizing the name
        normalized_name = hotel_name.lower().strip()
        
        # Check direct match or partial matches
        matched_hotel = None
        for key, hotel_info in detailed_hotels.items():
            if key == normalized_name or normalized_name in key or key in normalized_name:
                matched_hotel = hotel_info
                break
        
        if matched_hotel:
            return f"""**{matched_hotel['name']}**

**Location:** {matched_hotel['location']}
**Description:** {matched_hotel['description']}
**Price Range:** {matched_hotel['price_range']}
**Rating:** {matched_hotel['rating']}

**Amenities:**
{chr(10).join(f"• {amenity}" for amenity in matched_hotel['amenities'])}

**Contact Information:**
{matched_hotel['contact']}

**Address:** {matched_hotel['address']}

**Check-in Time:** {matched_hotel['check_in']}
**Check-out Time:** {matched_hotel['check_out']}
**Cancellation Policy:** {matched_hotel['cancellation']}"""
        
        # If not found in data module, try database search
        cluster = get_cluster_connection()
        if cluster:
            try:
                bucket_name = os.environ.get('CB_BUCKET', 'vector-search-testing')
                scope_name = os.environ.get('CB_SCOPE', 'agentc_data')
                collection_name = os.environ.get('CB_COLLECTION', 'hotel_data')
                
                query = f"""
                SELECT RAW content 
                FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` 
                WHERE LOWER(content) LIKE LOWER('%{hotel_name}%')
                LIMIT 1
                """
                
                result = cluster.query(query)
                rows = list(result.rows())
                
                if rows:
                    return rows[0]
            except Exception as db_error:
                print(f"Database search failed: {str(db_error)}")
        
        # If still not found, return helpful message
        return f"Sorry, I couldn't find detailed information for '{hotel_name}'. Please try searching for hotels first to see available options."
        
    except Exception as e:
        return f"I encountered an error while retrieving hotel details: {str(e)}. Please try again or search for hotels first."