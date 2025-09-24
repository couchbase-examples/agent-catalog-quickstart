import datetime
import logging
import os
import re
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
cluster = None
try:
    auth = PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password"),
    )
    options = ClusterOptions(auth)
    
    # Use WAN profile for better timeout handling with remote clusters
    options.apply_profile("wan_development")
    
    cluster = Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        options,
    )
    cluster.wait_until_ready(timedelta(seconds=15))
except CouchbaseException as e:
    logger.error(f"Could not connect to Couchbase cluster: {e!s}")
    cluster = None


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


def parse_booking_input(booking_input: str) -> tuple[str, str, str, str]:
    """Parse and normalize booking input from natural language or structured format."""
    if not booking_input or not isinstance(booking_input, str):
        raise ValueError("Input must be a string in format 'source_airport,destination_airport,date'")
    
    original_input = booking_input.strip()
    
    # If already in correct format, use as-is
    if re.match(r"^[A-Z]{3},[A-Z]{3},\d{4}-\d{2}-\d{2}$", original_input):
        return original_input, original_input, "", ""
    
    # Extract airport codes from natural language
    airport_codes = re.findall(r'\b[A-Z]{3}\b', original_input.upper())
    
    # Extract or calculate date
    date_str = _parse_date_from_text(original_input)
    
    # Reconstruct input if we found airport codes
    if len(airport_codes) >= 2 and date_str:
        structured_input = f"{airport_codes[0]},{airport_codes[1]},{date_str}"
        return structured_input, original_input, airport_codes[0], airport_codes[1]
    
    # Try comma-separated format
    parts = original_input.split(",")
    if len(parts) >= 2:
        return original_input, original_input, "", ""
    
    raise ValueError(f"Could not parse booking request. Please use format 'JFK,LAX,2025-12-25' or specify clear airport codes and date. Input was: {original_input}")


def _parse_date_from_text(text: str) -> str:
    """Extract or calculate date from natural language text."""
    if re.search(r'\btomorrow\b', text, re.I):
        return (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if re.search(r'\bnext week\b', text, re.I):
        return (datetime.date.today() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Look for explicit date
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if date_match:
        return date_match.group(1)
    
    # Default to tomorrow if no date specified
    return (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")


def validate_booking_parts(booking_input: str) -> tuple[str, str, str]:
    """Validate and extract booking components from structured input."""
    parts = booking_input.strip().split(",")
    if len(parts) != 3:
        raise ValueError("Input must be in format 'source_airport,destination_airport,date'. Example: 'JFK,LAX,2024-12-25'")
    
    source_airport, destination_airport, departure_date = [part.strip() for part in parts]
    
    if not source_airport or not destination_airport or not departure_date:
        raise ValueError("All fields are required: source_airport, destination_airport, date")
    
    return source_airport, destination_airport, departure_date


def validate_airport_codes(source: str, destination: str) -> tuple[str, str]:
    """Validate and normalize airport codes."""
    source = source.upper()
    destination = destination.upper()
    
    if len(source) != 3 or len(destination) != 3:
        raise ValueError(f"Airport codes must be 3 letters (e.g., JFK, LAX). Got: {source}, {destination}")
    
    if not source.isalpha() or not destination.isalpha():
        raise ValueError(f"Airport codes must be letters only. Got: {source}, {destination}")
    
    return source, destination


def parse_and_validate_date(departure_date: str) -> tuple[datetime.date, str]:
    """Parse and validate departure date, handling relative dates."""
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
                raise ValueError("Date must be in YYYY-MM-DD format. Example: 2025-12-25")
            dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
        # Allow bookings for today and future dates
        if dep_date < datetime.date.today():
            today = datetime.date.today().strftime('%Y-%m-%d')
            raise ValueError(f"Departure date cannot be in the past. Today is {today}. Please use today's date or later.")

        # Add logging for debugging
        logger.info(f"🗓️ Date validation: dep_date={dep_date}, today={datetime.date.today()}, comparison={dep_date < datetime.date.today()}")
        return dep_date, departure_date

    except ValueError as e:
        if "time data" in str(e):
            raise ValueError("Invalid date format. Please use YYYY-MM-DD format. Example: 2025-12-25")
        raise


def parse_passenger_details(original_input: str) -> tuple[int, str]:
    """Extract passenger count and class from natural language input."""
    passengers = 1
    flight_class = "economy"
    
    # Parse passenger count - prefer explicit key=value when present
    # Pattern 0: key=value form like "passengers=2"
    kv_match = re.search(r'passengers\s*[:=]\s*(\d+)', original_input, re.I)
    if kv_match:
        passengers = int(kv_match.group(1))
    else:
        # Pattern 1: "2 passengers" or "2 passenger"
        passenger_match = re.search(r'(\d+)\s*passengers?', original_input, re.I)
        if passenger_match:
            passengers = int(passenger_match.group(1))
        else:
            # Pattern 2: Comma-separated format like "LAX,JFK,2025-08-06,2,business"
            parts = original_input.split(',')
            if len(parts) >= 4:  # source,dest,date,passengers,...
                # Attempt to find an integer in the 4th part or any part mentioning passengers
                parsed = False
                try:
                    passengers = int(parts[3].strip())
                    parsed = True
                except (ValueError, IndexError):
                    pass
                if not parsed:
                    for part in parts:
                        if 'passenger' in part.lower():
                            mnum = re.search(r'(\d+)', part)
                            if mnum:
                                passengers = int(mnum.group(1))
                                parsed = True
                                break
            else:
                # Pattern 3: Just a number anywhere (fallback)
                number_match = re.search(r'\b(\d+)\b', original_input)
                if number_match:
                    passengers = int(number_match.group(1))
    
    # Parse class - runs independently of passenger parsing
    # Enhanced patterns to catch "business class", "2 passengers, business class" etc.
    if (re.search(r'\bflight_class\s*[:=]\s*["\']?business["\']?', original_input, re.I) or
        re.search(r'\bbusiness\s*class\b', original_input, re.I) or
        re.search(r'\bbusiness\b', original_input, re.I)):
        flight_class = "business"
    elif (re.search(r'\bflight_class\s*[:=]\s*["\']?first["\']?', original_input, re.I) or
          re.search(r'\bfirst\s*class\b', original_input, re.I) or
          re.search(r'\bfirst\b', original_input, re.I)):
        flight_class = "first"
    elif (re.search(r'\bflight_class\s*[:=]\s*["\']?economy["\']?', original_input, re.I) or
          re.search(r'\beconomy\s*class\b', original_input, re.I) or
          re.search(r'\beconomy\b|\bbasic\b', original_input, re.I)):
        flight_class = "economy"
    
    return passengers, flight_class


def calculate_price(flight_class: str, passengers: int) -> float:
    """Calculate total price based on class and passenger count."""
    base_prices = {"economy": 250, "business": 750, "first": 1200}
    base_price = base_prices.get(flight_class, 250)
    return base_price * passengers


def check_duplicate_booking(source_airport: str, destination_airport: str, departure_date: str,
                          bucket_name: str, scope_name: str, collection_name: str) -> str | None:
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
        duplicate_result = cluster.query(
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


def create_booking_record(booking_id: str, source_airport: str, destination_airport: str,
                         departure_date: str, passengers: int, flight_class: str, total_price: float) -> dict:
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
    
    cluster.query(insert_query, 
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
def save_flight_booking(source_airport: str, destination_airport: str, departure_date: str,
                       passengers: int = 1, flight_class: str = "economy") -> str:
    """
    Save a flight booking to Couchbase database.

    Args:
        source_airport: 3-letter airport code (e.g. JFK)
        destination_airport: 3-letter airport code (e.g. LAX)
        departure_date: Date in YYYY-MM-DD format
        passengers: Number of passengers (1-10, default: 1)
        flight_class: Flight class - economy, business, or first (default: economy)

    Returns:
        Booking confirmation message with booking ID and details
    """
    try:
        # Log parameters to debug flight_class extraction
        logger.info(f"🎯 Booking parameters: source={source_airport}, dest={destination_airport}, date={departure_date}, passengers={passengers}, flight_class={flight_class}")

        # Validate database connection
        if cluster is None:
            return "Database connection unavailable. Unable to save booking. Please try again later."

        # Validate inputs with proper type checking
        source_airport, destination_airport = validate_airport_codes(source_airport, destination_airport)

        # Validate passenger count
        if not isinstance(passengers, int) or passengers < 1 or passengers > 10:
            return "Error: Number of passengers must be between 1 and 10"

        # Validate flight class
        valid_classes = ["economy", "business", "first"]
        if flight_class.lower() not in valid_classes:
            return f"Error: Flight class must be one of: {', '.join(valid_classes)}"
        flight_class = flight_class.lower()

        # Parse and validate date
        dep_date, departure_date = parse_and_validate_date(departure_date)

        # Setup database collection
        bucket_name = os.getenv("CB_BUCKET", "travel-sample")
        scope_name = "agentc_bookings"
        collection_name = f"user_bookings_{datetime.date.today().strftime('%Y%m%d')}"
        _ensure_collection_exists(bucket_name, scope_name, collection_name)

        # Check for duplicates
        duplicate_error = check_duplicate_booking(
            source_airport, destination_airport, departure_date,
            bucket_name, scope_name, collection_name)
        if duplicate_error:
            return duplicate_error

        # Calculate pricing
        total_price = calculate_price(flight_class, passengers)

        # Create and save booking
        booking_id = f"FL{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:8].upper()}"
        booking_data = create_booking_record(
            booking_id, source_airport, destination_airport,
            departure_date, passengers, flight_class, total_price)
        save_booking_to_db(booking_data, bucket_name, scope_name, collection_name)

        return format_booking_confirmation(booking_data)

    except ValueError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        logger.exception(f"Booking processing error: {e}")
        return f"Booking could not be processed: {str(e)}"
