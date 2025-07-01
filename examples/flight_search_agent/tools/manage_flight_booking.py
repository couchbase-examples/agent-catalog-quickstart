from agentc.catalog import tool
from pydantic import BaseModel
from typing import Optional
import datetime


class FlightBookingRequest(BaseModel):
    """Flight booking request model."""
    customer_id: str
    source_airport: str
    destination_airport: str
    departure_date: str
    return_date: Optional[str] = None
    passengers: int = 1
    flight_class: str = "economy"


class BookingResponse(BaseModel):
    """Flight booking response model."""
    booking_id: str
    status: str
    message: str
    flight_details: dict
    total_price: float


@tool
def manage_flight_booking(booking_request: FlightBookingRequest) -> BookingResponse:
    """
    Manage flight booking requests including validation, pricing, and booking confirmation.
    Integrates with customer context and flight availability data.
    """
    
    try:
        # Validate booking request
        if not booking_request.source_airport or not booking_request.destination_airport:
            return BookingResponse(
                booking_id="",
                status="error",
                message="Source and destination airports are required",
                flight_details={},
                total_price=0.0
            )
        
        # Validate date format and future dates
        try:
            dep_date = datetime.datetime.strptime(booking_request.departure_date, "%Y-%m-%d").date()
            if dep_date <= datetime.date.today():
                return BookingResponse(
                    booking_id="",
                    status="error", 
                    message="Departure date must be in the future",
                    flight_details={},
                    total_price=0.0
                )
        except ValueError:
            return BookingResponse(
                booking_id="",
                status="error",
                message="Invalid date format. Use YYYY-MM-DD",
                flight_details={},
                total_price=0.0
            )
        
        # Generate booking ID
        booking_id = f"FL{booking_request.customer_id[:4]}{dep_date.strftime('%m%d')}{booking_request.source_airport}{booking_request.destination_airport}"
        
        # Calculate base pricing
        base_prices = {
            "economy": 250,
            "business": 750, 
            "first": 1500
        }
        
        base_price = base_prices.get(booking_request.flight_class.lower(), 250)
        total_price = base_price * booking_request.passengers
        
        # Create flight details
        flight_details = {
            "route": f"{booking_request.source_airport} → {booking_request.destination_airport}",
            "departure_date": booking_request.departure_date,
            "return_date": booking_request.return_date,
            "passengers": booking_request.passengers,
            "class": booking_request.flight_class,
            "estimated_duration": "2-8 hours depending on route"
        }
        
        return BookingResponse(
            booking_id=booking_id,
            status="confirmed",
            message=f"Flight booking confirmed for {booking_request.passengers} passenger(s)",
            flight_details=flight_details,
            total_price=total_price
        )
        
    except Exception as e:
        return BookingResponse(
            booking_id="",
            status="error",
            message=f"Booking processing error: {str(e)}",
            flight_details={},
            total_price=0.0
        )