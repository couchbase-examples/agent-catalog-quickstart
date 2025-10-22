"""
Pydantic schemas for structured tool inputs.

This module defines type-safe input schemas for all flight search tools,
enabling structured tool calling and automatic validation.
"""

from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class FlightSearchInput(BaseModel):
    """Input schema for flight search/lookup operations."""

    source_airport: str = Field(
        ...,
        description="3-letter IATA airport code for departure (e.g., JFK, LAX, ORD)",
        min_length=3,
        max_length=3
    )
    destination_airport: str = Field(
        ...,
        description="3-letter IATA airport code for arrival (e.g., JFK, LAX, ORD)",
        min_length=3,
        max_length=3
    )

    @field_validator('source_airport', 'destination_airport')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        """Validate and normalize airport codes."""
        if not v.isalpha():
            raise ValueError(f"Airport code must contain only letters: {v}")
        return v.upper()


class BookingInput(BaseModel):
    """Input schema for flight booking operations."""

    source_airport: str = Field(
        ...,
        description="3-letter IATA airport code for departure (e.g., JFK, LAX, ORD)",
        min_length=3,
        max_length=3
    )
    destination_airport: str = Field(
        ...,
        description="3-letter IATA airport code for arrival (e.g., JFK, LAX, ORD)",
        min_length=3,
        max_length=3
    )
    departure_date: str = Field(
        ...,
        description="Departure date in YYYY-MM-DD format (e.g., 2025-12-25)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    passengers: int = Field(
        default=1,
        description="Number of passengers (1-20)",
        ge=1,
        le=20
    )
    flight_class: str = Field(
        default="economy",
        description="Flight class: economy, business, or first"
    )

    @field_validator('source_airport', 'destination_airport')
    @classmethod
    def validate_airport_code(cls, v: str) -> str:
        """Validate and normalize airport codes."""
        if not v.isalpha():
            raise ValueError(f"Airport code must contain only letters: {v}")
        return v.upper()

    @field_validator('flight_class')
    @classmethod
    def validate_flight_class(cls, v: str) -> str:
        """Validate and normalize flight class."""
        valid_classes = ['economy', 'business', 'first']
        normalized = v.lower().strip()
        if normalized not in valid_classes:
            raise ValueError(f"Flight class must be one of: {', '.join(valid_classes)}")
        return normalized


class BookingQueryInput(BaseModel):
    """Input schema for querying existing bookings."""

    booking_id: Optional[str] = Field(
        default=None,
        description="Optional booking ID to search for specific booking. Leave empty to retrieve all bookings."
    )
    source_airport: Optional[str] = Field(
        default=None,
        description="Optional 3-letter airport code to filter by departure airport",
        min_length=3,
        max_length=3
    )
    destination_airport: Optional[str] = Field(
        default=None,
        description="Optional 3-letter airport code to filter by arrival airport",
        min_length=3,
        max_length=3
    )


class AirlineReviewInput(BaseModel):
    """Input schema for airline review search."""

    query: str = Field(
        ...,
        description="Search query for airline reviews (e.g., 'SpiceJet service quality', 'IndiGo food')",
        min_length=2
    )
