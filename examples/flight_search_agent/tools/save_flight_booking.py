import datetime
import os
import uuid
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

    except Exception as e:
        print(f"Warning: Could not ensure collection exists: {e}")


@agentc.catalog.tool
def save_flight_booking(
    source_airport: str,
    destination_airport: str,
    departure_date: str,
    customer_id: str = "DEMO_USER",
    return_date: str = None,
    passengers: int = 1,
    flight_class: str = "economy",
) -> str:
    """
    Save a flight booking to Couchbase database.
    Handles flight reservations with validation, pricing, and booking confirmation.
    """
    try:
        # Validate required fields
        if not source_airport or not destination_airport or not departure_date:
            return "Error: Source airport, destination airport, and departure date are required for booking."

        # Validate and parse date
        try:
            # Handle relative dates
            if departure_date.lower() == "tomorrow":
                dep_date = datetime.date.today() + datetime.timedelta(days=1)
                departure_date = dep_date.strftime("%Y-%m-%d")
            elif departure_date.lower() == "today":
                dep_date = datetime.date.today()
                departure_date = dep_date.strftime("%Y-%m-%d")
            else:
                dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()

            # Check if date is in the future (allow today for demo purposes)
            if dep_date < datetime.date.today():
                return f"Error: Departure date must be in the future. Today is {datetime.date.today().strftime('%Y-%m-%d')}. Please use a date like {(datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')} or 'tomorrow'."
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format or 'tomorrow'."

        # Generate booking ID
        booking_id = (
            f"FL{customer_id[:4].upper()}{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:4].upper()}"
        )

        # Calculate pricing
        base_prices = {"economy": 250, "business": 750, "first": 1500, "premium": 400}
        base_price = base_prices.get(flight_class.lower(), 250)
        total_price = base_price * passengers

        # Prepare booking data
        booking_data = {
            "booking_id": booking_id,
            "customer_id": customer_id,
            "source_airport": source_airport.upper(),
            "destination_airport": destination_airport.upper(),
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": passengers,
            "flight_class": flight_class.title(),
            "total_price": total_price,
            "booking_time": datetime.datetime.now().isoformat(),
            "status": "confirmed",
        }

        # Setup collection name with timestamp
        bucket_name = os.getenv("CB_BUCKET", "vector-search-testing")
        scope_name = "agentc_bookings"
        collection_name = f"flight_bookings_{datetime.date.today().strftime('%Y%m%d')}"

        # Ensure collection exists
        _ensure_collection_exists(bucket_name, scope_name, collection_name)

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
Route: {source_airport.upper()} → {destination_airport.upper()}
Departure Date: {departure_date}
Return Date: {return_date if return_date else "One-way"}
Passengers: {passengers}
Class: {flight_class.title()}
Total Price: ${total_price:.2f}

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID
4. Booking confirmation sent to your email

Thank you for choosing our airline!
        """

        return booking_summary.strip()

    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Database error while saving booking: {e!s}"
        return f"Booking could not be saved due to a database error. Please try again later."
    except Exception as e:
        error_msg = f"Booking processing error: {e!s}"
        return f"Booking processing error. Please try again or contact customer service."
