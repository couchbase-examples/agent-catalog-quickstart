import datetime
import logging
import os
import uuid
from datetime import timedelta

import agentc
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import CouchbaseException
from couchbase.options import ClusterOptions
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
        auth = PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "Administrator"),
            password=os.getenv("CB_PASSWORD", "password"),
        )
        options = ClusterOptions(auth)

        # Use WAN profile for better timeout handling with remote clusters
        options.apply_profile("wan_development")

        _cluster = Cluster(
            os.getenv("CB_CONN_STRING", "couchbase://localhost"),
            options,
        )
        _cluster.wait_until_ready(timedelta(seconds=15))
        return _cluster
    except CouchbaseException as e:
        logger.error(f"Could not connect to Couchbase cluster: {e!s}")
        raise


def _ensure_collection_exists(bucket_name: str, scope_name: str, collection_name: str):
    """Ensure the booking collection exists, create if it doesn't."""
    try:
        # Create scope if it doesn't exist
        bucket = _get_cluster().bucket(bucket_name)
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
            _get_cluster().query(
                f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
            ).execute()
        except Exception:
            pass  # Index might already exist

    except Exception:
        pass


def validate_date(departure_date: str) -> tuple[datetime.date, str]:
    """Validate departure date is not in the past."""
    try:
        dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()

        if dep_date < datetime.date.today():
            today = datetime.date.today().strftime('%Y-%m-%d')
            raise ValueError(f"Departure date cannot be in the past. Today is {today}.")

        logger.info(f"🗓️ Date validated: {dep_date}")
        return dep_date, departure_date

    except ValueError as e:
        if "time data" in str(e) or "does not match format" in str(e):
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format (e.g., 2025-12-25)")
        raise


def calculate_price(flight_class: str, passengers: int) -> float:
    """Calculate total price based on class and passenger count."""
    base_prices = {"economy": 250, "business": 750, "first": 1200}
    base_price = base_prices.get(flight_class, 250)
    return base_price * passengers


def check_duplicate_booking(
    source_airport: str,
    destination_airport: str,
    departure_date: str,
    bucket_name: str,
    scope_name: str,
    collection_name: str
) -> str | None:
    """Check for existing duplicate bookings. Returns error message if found, None otherwise."""
    duplicate_check_query = f"""
    SELECT booking_id, total_price
    FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`
    WHERE source_airport = $source_airport
    AND destination_airport = $destination_airport
    AND departure_date = $departure_date
    AND status = 'confirmed'
    """

    try:
        duplicate_result = _get_cluster().query(
            duplicate_check_query,
            source_airport=source_airport,
            destination_airport=destination_airport,
            departure_date=departure_date
        )

        existing_bookings = list(duplicate_result.rows())

        if existing_bookings:
            existing_booking = existing_bookings[0]
            return f"""Duplicate booking found! You already have a confirmed booking:
- Booking ID: {existing_booking['booking_id']}
- Route: {source_airport} → {destination_airport}
- Date: {departure_date}
- Total: ${existing_booking['total_price']:.2f}

No new booking was created. Use the existing booking ID for reference."""

    except Exception as e:
        logger.warning(f"Duplicate check failed: {e}")

    return None


def create_booking_record(
    booking_id: str,
    source_airport: str,
    destination_airport: str,
    departure_date: str,
    passengers: int,
    flight_class: str,
    total_price: float
) -> dict:
    """Create booking data structure."""
    return {
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


def save_booking_to_db(booking_data: dict, bucket_name: str, scope_name: str, collection_name: str) -> None:
    """Save booking record to Couchbase database."""
    insert_query = f"""
    INSERT INTO `{bucket_name}`.`{scope_name}`.`{collection_name}` (KEY, VALUE)
    VALUES ($booking_id, $booking_data)
    """

    _get_cluster().query(insert_query,
                         booking_id=booking_data["booking_id"],
                         booking_data=booking_data).execute()


def format_booking_confirmation(booking_data: dict) -> str:
    """Format booking confirmation message."""
    return f"""Flight Booking Confirmed!

Booking ID: {booking_data['booking_id']}
Route: {booking_data['source_airport']} → {booking_data['destination_airport']}
Departure Date: {booking_data['departure_date']}
Passengers: {booking_data['passengers']}
Class: {booking_data['flight_class']}
Total Price: ${booking_data['total_price']:.2f}

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!"""


@agentc.catalog.tool
def save_flight_booking(
    source_airport: str,
    destination_airport: str,
    departure_date: str,
    passengers: int = 1,
    flight_class: str = "economy"
) -> str:
    """
    Save a flight booking to Couchbase database.

    Args:
        source_airport: 3-letter airport code (e.g., JFK)
        destination_airport: 3-letter airport code (e.g., LAX)
        departure_date: Date in YYYY-MM-DD format
        passengers: Number of passengers (1-20)
        flight_class: Flight class (economy, business, or first)

    Returns:
        Booking confirmation message
    """
    try:
        # Validate database connection
        if cluster is None:
            return "Database connection unavailable. Unable to save booking. Please try again later."

        # Normalize inputs
        source_airport = source_airport.upper()
        destination_airport = destination_airport.upper()
        flight_class = flight_class.lower()

        # Validate date
        dep_date, departure_date = validate_date(departure_date)

        # Setup database collection
        bucket_name = os.getenv("CB_BUCKET", "travel-sample")
        scope_name = "agentc_bookings"
        collection_name = f"user_bookings_{datetime.date.today().strftime('%Y%m%d')}"
        _ensure_collection_exists(bucket_name, scope_name, collection_name)

        # Check for duplicates
        duplicate_error = check_duplicate_booking(
            source_airport, destination_airport, departure_date,
            bucket_name, scope_name, collection_name
        )
        if duplicate_error:
            return duplicate_error

        # Calculate pricing
        total_price = calculate_price(flight_class, passengers)

        logger.info(
            f"🎯 Booking: {source_airport}→{destination_airport} "
            f"on {departure_date}, {passengers} pax, {flight_class} class"
        )

        # Create and save booking
        booking_id = f"FL{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:8].upper()}"
        booking_data = create_booking_record(
            booking_id, source_airport, destination_airport,
            departure_date, passengers, flight_class, total_price
        )
        save_booking_to_db(booking_data, bucket_name, scope_name, collection_name)

        return format_booking_confirmation(booking_data)

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.exception(f"Booking processing error: {e}")
        return f"Booking could not be processed: {str(e)}"
