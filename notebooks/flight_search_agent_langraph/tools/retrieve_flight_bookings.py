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
try:
    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password"),
    )
    options = couchbase.options.ClusterOptions(auth)
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        options
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"


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
        bucket_name = os.getenv("CB_BUCKET", "travel-sample")
        scope_name = "agentc_bookings"

        # Parse input
        if not booking_query or booking_query.strip().lower() in ["", "all", "none"]:
            # Retrieve all bookings
            return _retrieve_all_bookings(bucket_name, scope_name)
        # Parse specific booking query
        parts = booking_query.strip().split(",")
        if len(parts) != 3:
            return "Error: For specific booking search, use format 'source_airport,destination_airport,date'. Example: 'JFK,LAX,2024-12-25'. Or use empty string for all bookings."

        source_airport, destination_airport, date = [part.strip().upper() for part in parts]

        # Validate inputs
        if len(source_airport) != 3 or len(destination_airport) != 3:
            return "Error: Airport codes must be 3 letters. Example: 'JFK,LAX,2024-12-25'"

        if not source_airport.isalpha() or not destination_airport.isalpha():
            return "Error: Airport codes must be letters only. Example: 'JFK,LAX,2024-12-25'"

        # Validate date format
        try:
            datetime.datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return "Error: Date must be in YYYY-MM-DD format. Example: 'JFK,LAX,2024-12-25'"

        return _retrieve_specific_booking(bucket_name, scope_name, source_airport, destination_airport, date)

    except Exception as e:
        logger.exception(f"Error retrieving bookings: {e}")
        return "Error retrieving bookings. Please try again."


def _retrieve_all_bookings(bucket_name: str, scope_name: str) -> str:
    """Retrieve all bookings from the last 30 days."""
    try:
        all_bookings = []
        today = datetime.date.today()

        # Search across the last 30 days of booking collections
        for days_back in range(30):
            check_date = today - datetime.timedelta(days=days_back)
            collection_name = f"user_bookings_{check_date.strftime('%Y%m%d')}"

            try:
                # Query this collection
                query = f"""
                SELECT VALUE booking
                FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` booking
                WHERE booking.status = 'confirmed'
                """

                result = cluster.query(query)
                rows = list(result.rows())
                all_bookings.extend(rows)

            except couchbase.exceptions.CouchbaseException:
                # Collection doesn't exist or is empty, continue
                continue

        if not all_bookings:
            return "No bookings found."

        # Sort bookings by booking time (most recent first)
        all_bookings.sort(key=lambda x: x.get("booking_time", ""), reverse=True)

        # Format bookings for display
        result = f"Your Current Bookings ({len(all_bookings)} found):\n\n"
        for i, booking in enumerate(all_bookings, 1):
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

    except Exception as e:
        logger.exception(f"Error retrieving all bookings: {e}")
        return "Unable to retrieve bookings. Please try again."


def _retrieve_specific_booking(bucket_name: str, scope_name: str, source_airport: str, destination_airport: str, date: str) -> str:
    """Retrieve bookings for a specific route and date."""
    try:
        # Try to find the booking in collections from the last 30 days
        today = datetime.date.today()
        found_bookings = []

        for days_back in range(30):
            check_date = today - datetime.timedelta(days=days_back)
            collection_name = f"user_bookings_{check_date.strftime('%Y%m%d')}"

            try:
                # Query for specific booking
                query = f"""
                SELECT VALUE booking
                FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` booking
                WHERE booking.source_airport = '{source_airport}'
                AND booking.destination_airport = '{destination_airport}'
                AND booking.departure_date = '{date}'
                AND booking.status = 'confirmed'
                """

                result = cluster.query(query)
                rows = list(result.rows())
                found_bookings.extend(rows)

            except couchbase.exceptions.CouchbaseException:
                # Collection doesn't exist, continue
                continue

        if not found_bookings:
            return f"No bookings found for {source_airport} → {destination_airport} on {date}."

        # Format found bookings
        result = f"Found {len(found_bookings)} booking(s) for {source_airport} → {destination_airport} on {date}:\n\n"
        for i, booking in enumerate(found_bookings, 1):
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

    except Exception as e:
        logger.exception(f"Error retrieving specific booking: {e}")
        return f"Error retrieving booking for {source_airport} → {destination_airport} on {date}. Please try again."
