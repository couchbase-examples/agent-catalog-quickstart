import datetime
import uuid

import agentc


@agentc.catalog.tool
def manage_flight_booking(
    customer_id: str,
    source_airport: str,
    destination_airport: str,
    departure_date: str,
    return_date: str = None,
    passengers: int = 1,
    flight_class: str = "economy",
) -> str:
    """
    Manage flight booking requests including validation, pricing, and booking confirmation.
    Handles flight reservations, seat selection, and payment processing.
    """

    try:
        # Validate booking request
        if not source_airport or not destination_airport:
            return "Error: Source and destination airports are required for booking."

        # Validate date format and future dates
        try:
            dep_date = datetime.datetime.strptime(departure_date, "%Y-%m-%d").date()
            if dep_date <= datetime.date.today():
                return "Error: Departure date must be in the future."
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format."

        # Generate booking ID
        booking_id = (
            f"FL{customer_id[:4].upper()}{dep_date.strftime('%m%d')}{str(uuid.uuid4())[:4].upper()}"
        )

        # Calculate base pricing
        base_prices = {"economy": 250, "business": 750, "first": 1500, "premium": 400}

        base_price = base_prices.get(flight_class.lower(), 250)
        total_price = base_price * passengers

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
        return f"Booking processing error: {str(e)}. Please try again or contact customer service."
