import datetime
import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

dotenv.load_dotenv(override=True)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
try:
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        couchbase.options.ClusterOptions(
            authenticator=couchbase.auth.PasswordAuthenticator(
                username=os.getenv("CB_USERNAME", "Administrator"),
                password=os.getenv("CB_PASSWORD", "password"),
            )
        ),
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"
    print(error_msg)


@agentc.catalog.tool
def retrieve_flight_bookings() -> str:
    """
    Retrieve all flight bookings from Couchbase database.
    Single user system - retrieves all bookings for the user.
    """
    try:
        bucket_name = os.getenv("CB_BUCKET", "vector-search-testing")
        scope_name = "agentc_bookings"

        # Search across the last 30 days of booking collections
        all_bookings = []
        today = datetime.date.today()

        for days_back in range(30):
            check_date = today - datetime.timedelta(days=days_back)
            collection_name = f"user_bookings_{check_date.strftime('%Y%m%d')}"

            try:
                # Try to query this collection - single user system
                query = f"""
                SELECT VALUE booking
                FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` booking
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

    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Database error while retrieving bookings: {e!s}"
        return "Unable to retrieve bookings due to a database error. Please try again later."
    except Exception as e:
        error_msg = f"Error retrieving bookings: {e!s}"
        return "Error retrieving bookings. Please try again or contact customer service."
