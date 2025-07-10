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
    return_date: str = None,
    passengers: int = 1,
    flight_class: str = "economy",
) -> str:
    """
    Save a flight booking to Couchbase database.
    Handles flight reservations with validation, pricing, and booking confirmation.
    Single user system - no customer ID required.
    Checks for duplicate bookings before creating new ones.
    """
    try:
        # Validate required fields
        if not source_airport or not destination_airport or not departure_date:
            return "Error: Source airport, destination airport, and departure date are required for booking."

        # Validate and normalize airport codes
        source_airport = source_airport.upper().strip()
        destination_airport = destination_airport.upper().strip()
        
        if len(source_airport) != 3 or len(destination_airport) != 3:
            return f"Error: Airport codes must be 3 letters (e.g., JFK, LAX). Got: {source_airport}, {destination_airport}"
        
        # Validate passengers
        if not isinstance(passengers, int) or passengers < 1 or passengers > 9:
            return "Error: Number of passengers must be between 1 and 9."
        
        # Validate flight class (clean and normalize)
        original_flight_class = flight_class
        flight_class = str(flight_class).strip().strip('"\'').strip().lower()
        
        # Additional cleaning for edge cases
        flight_class = re.sub(r'["\'\s]+$', '', flight_class)  # Remove trailing quotes/spaces
        flight_class = re.sub(r'^["\'\s]+', '', flight_class)  # Remove leading quotes/spaces
        
        valid_classes = ["economy", "business", "first", "premium"]
        if flight_class not in valid_classes:
            logger.warning(f"Invalid flight class: original='{original_flight_class}', cleaned='{flight_class}'")
            return f"Error: Flight class must be one of: {', '.join(valid_classes)}. Got: '{flight_class}' (original: '{original_flight_class}')"

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
                dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()

            # Check if date is in the future (allow today for demo purposes)
            if dep_date < datetime.date.today():
                return f"Error: Departure date must be in the future. Today is {datetime.date.today().strftime('%Y-%m-%d')}. Please use a date like {(datetime.date.today() + datetime.timedelta(days=1)).strftime('%Y-%m-%d')} or 'tomorrow'."
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format or 'tomorrow'."

        # Setup collection info
        bucket_name = os.getenv("CB_BUCKET", "vector-search-testing")
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
        AND passengers = $passengers 
        AND flight_class = $flight_class
        AND status = 'confirmed'
        """
        
        try:
            duplicate_result = cluster.query(
                duplicate_check_query,
                source_airport=source_airport.upper(),
                destination_airport=destination_airport.upper(),
                departure_date=departure_date,
                passengers=passengers,
                flight_class=flight_class.title()
            )
            
            existing_bookings = list(duplicate_result.rows())
            
            if existing_bookings:
                existing_booking = existing_bookings[0]
                return f"""
Duplicate Booking Detected!

You already have a confirmed booking for this exact flight:
- Booking ID: {existing_booking['booking_id']}
- Route: {source_airport.upper()} → {destination_airport.upper()}
- Date: {departure_date}
- Passengers: {passengers}
- Class: {flight_class.title()}
- Total: ${existing_booking['total_price']:.2f}

No new booking was created. Use the existing booking ID for reference.
                """.strip()
                
        except Exception as e:
            # Continue with booking creation if duplicate check fails
            pass

        # Generate booking ID for single user
        booking_id = (
            f"FL{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:8].upper()}"
        )

        # Calculate pricing
        base_prices = {"economy": 250, "business": 750, "first": 1500, "premium": 400}
        base_price = base_prices.get(flight_class.lower(), 250)
        total_price = base_price * passengers

        # Prepare booking data
        booking_data = {
            "booking_id": booking_id,
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
