import datetime
import logging
import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
_cluster = None


def _get_cluster():
    """Lazy connection to Couchbase cluster - only connects when needed."""
    global _cluster
    if _cluster is not None:
        return _cluster

    try:
        auth = couchbase.auth.PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
        )
        options = couchbase.options.ClusterOptions(auth)

        # Use WAN profile for better timeout handling with remote clusters
        options.apply_profile("wan_development")

        _cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            options
        )
        _cluster.wait_until_ready(timedelta(seconds=15))
        return _cluster
    except couchbase.exceptions.CouchbaseException as e:
        logger.error(f"Could not connect to Couchbase cluster: {e!s}")
        raise


def parse_booking_query(booking_query: str) -> dict:
    """Parse booking query input and return search parameters."""
    # More robust empty input detection
    if (not booking_query or
        booking_query.strip() == "" or
        booking_query.strip().lower() in ["all", "none", "show all", "get all"] or
        len(booking_query.strip()) <= 2):  # Handle single chars, colons, etc.
        return {"type": "all"}

    parts = booking_query.strip().split(",")
    if len(parts) != 3:
        raise ValueError("For specific booking search, use format 'source_airport,destination_airport,date'. Example: 'JFK,LAX,2024-12-25'. Or use empty string for all bookings.")
    
    source_airport, destination_airport, date = [part.strip().upper() for part in parts]
    
    # Validate airport codes
    if len(source_airport) != 3 or len(destination_airport) != 3:
        raise ValueError("Airport codes must be 3 letters. Example: 'JFK,LAX,2024-12-25'")
    
    if not source_airport.isalpha() or not destination_airport.isalpha():
        raise ValueError("Airport codes must be letters only. Example: 'JFK,LAX,2024-12-25'")
    
    # Validate date format
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format. Example: 'JFK,LAX,2024-12-25'")
    
    return {
        "type": "specific",
        "source_airport": source_airport,
        "destination_airport": destination_airport,
        "date": date
    }


def get_current_collection_name() -> str:
    """Get today's booking collection name."""
    return f"user_bookings_{datetime.date.today().strftime('%Y%m%d')}"


def format_booking_display(bookings: list) -> str:
    """Format bookings for display."""
    if not bookings:
        return "No bookings found."
    
    result = f"Your Current Bookings ({len(bookings)} found):\n\n"
    for i, booking in enumerate(bookings, 1):
        result += f"Booking {i}:\n"
        result += f"  Booking ID: {booking['booking_id']}\n"
        result += f"  Route: {booking['source_airport']} → {booking['destination_airport']}\n"
        result += f"  Date: {booking['departure_date']}\n"
        result += f"  Passengers: {booking['passengers']}\n"
        result += f"  Class: {booking['flight_class']}\n"
        result += f"  Total: ${booking['total_price']:.2f}\n"
        result += f"  Status: {booking.get('status', 'Confirmed')}\n"
        result += f"  Booked: {booking['booking_time'][:10]}\n\n"
    
    return result.strip()


@agentc.catalog.tool
def retrieve_flight_bookings(booking_query: str = "") -> str:
    """
    Retrieve flight bookings from Couchbase database.

    Input options:
    1. Empty string or "all" - retrieve all bookings
    2. "source_airport,destination_airport,date" - retrieve specific booking

    Examples:
    - "" or "all" - Show all bookings
    - "JFK,LAX,2024-12-25" - Show booking for specific flight
    """
    try:
        # Validate database connection
        if cluster is None:
            return "Database connection unavailable. Unable to retrieve bookings. Please try again later."
        
        # Parse and validate input
        search_params = parse_booking_query(booking_query)
        
        # Database configuration
        bucket_name = os.getenv("CB_BUCKET", "travel-sample")
        scope_name = "agentc_bookings"
        collection_name = get_current_collection_name()
        
        if search_params["type"] == "all":
            # Retrieve all bookings from current collection (simplified approach)
            query = f"""
            SELECT VALUE booking
            FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` booking
            WHERE booking.status = $status
            ORDER BY booking.booking_time DESC
            """

            result = _get_cluster().query(query, status="confirmed")

        else:
            # Retrieve specific booking using parameterized query (secure)
            query = f"""
            SELECT VALUE booking
            FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` booking
            WHERE booking.source_airport = $source_airport
            AND booking.destination_airport = $destination_airport
            AND booking.departure_date = $date
            AND booking.status = $status
            """

            result = _get_cluster().query(
                query,
                source_airport=search_params["source_airport"],
                destination_airport=search_params["destination_airport"],
                date=search_params["date"],
                status="confirmed"
            )
        
        bookings = list(result.rows())
        
        # Special message for specific booking search
        if search_params["type"] == "specific" and not bookings:
            return f"No bookings found for {search_params['source_airport']} → {search_params['destination_airport']} on {search_params['date']}."
        
        return format_booking_display(bookings)
        
    except ValueError as e:
        return f"Error: {str(e)}"
    except couchbase.exceptions.CouchbaseException as e:
        logger.exception(f"Database error: {e}")
        return "Database error: Unable to retrieve bookings. Please try again later."
    except Exception as e:
        logger.exception(f"Error retrieving bookings: {e}")
        return "Error retrieving bookings. Please try again."
