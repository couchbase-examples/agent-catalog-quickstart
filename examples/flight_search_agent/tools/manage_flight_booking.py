import datetime
import json
import os
import uuid

import agentc


@agentc.catalog.tool
def manage_flight_booking(
    source_airport: str = "",
    destination_airport: str = "",
    departure_date: str = "",
    customer_id: str = "DEMO_USER",
    return_date: str = None,
    passengers: int = 1,
    flight_class: str = "economy",
    action: str = "book",
) -> str:
    """
    Manage flight booking requests including validation, pricing, and booking confirmation.
    Handles flight reservations, seat selection, and payment processing.
    Also supports retrieving existing bookings.
    """

    try:
        # Handle retrieval action
        if action.lower() == "retrieve":
            return _retrieve_bookings(customer_id)

        # Handle booking action (default)
        if not source_airport or not destination_airport or not departure_date:
            return "Error: Source airport, destination airport, and departure date are required for booking."

        # Validate date format and future dates
        try:
            # Handle "tomorrow" and other relative dates
            if departure_date.lower() == "tomorrow":
                dep_date = datetime.date.today() + datetime.timedelta(days=1)
            elif departure_date.lower() == "today":
                dep_date = datetime.date.today()
            else:
                dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()

            if dep_date < datetime.date.today():
                return "Error: Departure date must be in the future."
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format or 'tomorrow'."

        # Generate booking ID
        booking_id = (
            f"FL{customer_id[:4].upper()}{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:4].upper()}"
        )

        # Calculate base pricing
        base_prices = {"economy": 250, "business": 750, "first": 1500, "premium": 400}
        base_price = base_prices.get(flight_class.lower(), 250)
        total_price = base_price * passengers

        # Store booking in session file
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
        }
        _save_booking(booking_data)

        # Create booking summary
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

    except Exception as e:
        return f"Booking processing error: {e!s}. Please try again or contact customer service."


def _save_booking(booking_data):
    """Save booking data to session file."""
    session_file = "bookings_session.txt"
    try:
        # Read existing bookings
        bookings = []
        if os.path.exists(session_file):
            with open(session_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        bookings.append(json.loads(line))

        # Add new booking
        bookings.append(booking_data)

        # Write all bookings back
        with open(session_file, "w") as f:
            for booking in bookings:
                f.write(json.dumps(booking) + "\n")
    except Exception:
        # Fail silently for demo purposes
        pass


def _retrieve_bookings(customer_id):
    """Retrieve bookings for a customer from session file."""
    session_file = "bookings_session.txt"
    try:
        if not os.path.exists(session_file):
            return "No bookings found. You haven't made any bookings yet."

        bookings = []
        with open(session_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    booking = json.loads(line)
                    if booking.get("customer_id") == customer_id:
                        bookings.append(booking)

        if not bookings:
            return f"No bookings found for customer {customer_id}."

        # Format bookings for display
        result = f"Your Current Bookings ({len(bookings)} found):\n\n"
        for i, booking in enumerate(bookings, 1):
            result += f"Booking {i}:\n"
            result += f"  Booking ID: {booking['booking_id']}\n"
            result += f"  Route: {booking['source_airport']} → {booking['destination_airport']}\n"
            result += f"  Date: {booking['departure_date']}\n"
            result += f"  Passengers: {booking['passengers']}\n"
            result += f"  Class: {booking['flight_class']}\n"
            result += f"  Total: ${booking['total_price']:.2f}\n"
            result += f"  Booked: {booking['booking_time'][:10]}\n\n"

        return result.strip()

    except Exception as e:
        return f"Error retrieving bookings: {e!s}"
