import datetime
import logging
import os
import re
import uuid
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
        options,
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"


def _ensure_collection_exists(bucket_name: str, scope_name: str, collection_name: str):
    """Ensure the booking collection exists, create if it doesn't."""
    try:
        # Create scope if it doesn't exist
        bucket = cluster.bucket(bucket_name)
        bucket_manager = bucket.collections()

        try:
            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists:
                bucket_manager.create_scope(scope_name)
        except Exception:
            pass  # Scope might already exist

        # Create collection if it doesn't exist
        try:
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name
                and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if not collection_exists:
                bucket_manager.create_collection(scope_name, collection_name)
        except Exception:
            pass  # Collection might already exist

        # Create primary index if it doesn't exist
        try:
            cluster.query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            ).execute()
        except Exception:
            pass  # Index might already exist

    except Exception:
        pass


@agentc.catalog.tool
def save_flight_booking(booking_input: str) -> str:
    """
    Save a flight booking to Couchbase database.

    Input format: "source_airport,destination_airport,date"
    Example: "JFK,LAX,2024-12-25"

    - source_airport: 3-letter airport code (e.g. JFK)
    - destination_airport: 3-letter airport code (e.g. LAX)
    - date: YYYY-MM-DD format

    Checks for duplicate bookings before creating new ones.
    """
    try:
        # Parse input string
        if not booking_input or not isinstance(booking_input, str):
            return "Error: Input must be a string in format 'source_airport,destination_airport,date'"

        # Split and validate input
        parts = booking_input.strip().split(",")
        if len(parts) != 3:
            return "Error: Input must be in format 'source_airport,destination_airport,date'. Example: 'JFK,LAX,2024-12-25'"

        source_airport, destination_airport, departure_date = [part.strip() for part in parts]

        # Validate required fields
        if not source_airport or not destination_airport or not departure_date:
            return "Error: All fields are required: source_airport, destination_airport, date"

        # Validate and normalize airport codes
        source_airport = source_airport.upper()
        destination_airport = destination_airport.upper()

        if len(source_airport) != 3 or len(destination_airport) != 3:
            return f"Error: Airport codes must be 3 letters (e.g., JFK, LAX). Got: {source_airport}, {destination_airport}"

        if not source_airport.isalpha() or not destination_airport.isalpha():
            return f"Error: Airport codes must be letters only. Got: {source_airport}, {destination_airport}"

        # Validate and parse date
        try:
            # Handle relative dates
            if departure_date.lower() == "tomorrow":
                dep_date = datetime.date.today() + datetime.timedelta(days=1)
                departure_date = dep_date.strftime("%Y-%m-%d")
            elif departure_date.lower() == "today":
                dep_date = datetime.date.today()
                departure_date = dep_date.strftime("%Y-%m-%d")
            elif departure_date.lower() == "next week":
                dep_date = datetime.date.today() + datetime.timedelta(days=7)
                departure_date = dep_date.strftime("%Y-%m-%d")
            else:
                # Validate date format
                if not re.match(r"^\d{4}-\d{2}-\d{2}$", departure_date):
                    return "Error: Date must be in YYYY-MM-DD format. Example: 2024-12-25"
                dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()

            # Check if date is in the future (allow today for demo purposes)
            if dep_date < datetime.date.today():
                return f"Error: Departure date must be in the future. Today is {datetime.date.today().strftime('%Y-%m-%d')}. Please use a date like {(datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')}"
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format. Example: 2024-12-25"

        # Setup collection info
        bucket_name = os.getenv("CB_BUCKET", "travel-sample")
        scope_name = "agentc_bookings"
        collection_name = f"user_bookings_{datetime.date.today().strftime('%Y%m%d')}"

        # Ensure collection exists
        _ensure_collection_exists(bucket_name, scope_name, collection_name)

        # Check for duplicate bookings using Couchbase SDK
        duplicate_check_query = f"""
        SELECT booking_id, total_price
        FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`
        WHERE source_airport = $source_airport
        AND destination_airport = $destination_airport
        AND departure_date = $departure_date
        AND status = 'confirmed'
        """

        try:
            duplicate_result = cluster.query(
                duplicate_check_query,
                source_airport=source_airport, 
                destination_airport=destination_airport, 
                departure_date=departure_date
            )

            existing_bookings = list(duplicate_result.rows())

            if existing_bookings:
                existing_booking = existing_bookings[0]
                return f"""
Duplicate Booking Detected!

You already have a confirmed booking for this flight:
- Booking ID: {existing_booking['booking_id']}
- Route: {source_airport} → {destination_airport}
- Date: {departure_date}
- Total: ${existing_booking['total_price']:.2f}

No new booking was created. Use the existing booking ID for reference.
                """.strip()

        except Exception as e:
            # Continue with booking creation if duplicate check fails
            logger.warning(f"Duplicate check failed: {e}")

        # Generate booking ID
        booking_id = f"FL{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:8].upper()}"

        # Default booking values
        passengers = 1
        flight_class = "economy"
        base_price = 250  # Base price for economy
        total_price = base_price * passengers

        # Prepare booking data
        booking_data = {
            "booking_id": booking_id,
            "source_airport": source_airport,
            "destination_airport": destination_airport,
            "departure_date": departure_date,
            "passengers": passengers,
            "flight_class": flight_class,
            "total_price": total_price,
            "booking_time": datetime.datetime.now().isoformat(),
            "status": "confirmed",
        }

        # Insert booking into Couchbase
        insert_query = f"""
        INSERT INTO `{bucket_name}`.`{scope_name}`.`{collection_name}` (KEY, VALUE)
        VALUES ($booking_id, $booking_data)
        """

        cluster.query(insert_query, booking_id=booking_id, booking_data=booking_data).execute()

        # Create booking confirmation
        booking_summary = f"""
Flight Booking Confirmed!

Booking ID: {booking_id}
Route: {source_airport} → {destination_airport}
Departure Date: {departure_date}
Passengers: {passengers}
Class: {flight_class}
Total Price: ${total_price:.2f}

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!
        """

        return booking_summary.strip()

    except Exception as e:
        logger.exception(f"Booking processing error: {e}")
        return "Booking could not be processed. Please try again with format: 'source_airport,destination_airport,date' (e.g., 'JFK,LAX,2024-12-25')"
