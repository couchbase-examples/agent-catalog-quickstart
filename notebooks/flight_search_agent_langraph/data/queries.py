"""
Shared flight search queries for both evaluation and testing.
"""

# Flight search queries (for evaluation and testing)
FLIGHT_SEARCH_QUERIES = [
    "Find flights from JFK to LAX",
    "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
    "Book an economy flight from JFK to MIA for next week, 1 passenger",
    "Show me my current flight bookings",
    "What do passengers say about SpiceJet's service quality?",
]

# Comprehensive reference answers based on actual system responses
FLIGHT_REFERENCE_ANSWERS = [
    # Query 1: Flight search JFK to LAX
    """Available flights from JFK to LAX:
- AS flight from JFK to LAX using 321 762
- B6 flight from JFK to LAX using 320  
- DL flight from JFK to LAX using 76W 752
- QF flight from JFK to LAX using 744
- AA flight from JFK to LAX using 32B 762
- UA flight from JFK to LAX using 757
- US flight from JFK to LAX using 32B 762
- VX flight from JFK to LAX using 320

These flights provide various airline options including American Airlines (AA), Delta Air Lines (DL), United Airlines (UA), JetBlue (B6), Virgin America (VX), Qantas (QF), US Airways (US), and Alaska Airlines (AS).""",

    # Query 2: Flight booking LAX to JFK for tomorrow
    """Flight Booking Confirmed!

Booking ID: FL0725ADB86B00
Route: LAX → JFK
Departure Date: 2025-07-25
Passengers: 1
Class: economy
Total Price: $250.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 3: Flight booking JFK to MIA for next week
    """Flight booking for JFK to MIA has been successfully processed.

Booking Details:
- Route: JFK → MIA (John F. Kennedy International Airport to Miami International Airport)  
- Date: Next week (7 days from today)
- Passengers: 1
- Class: Economy
- Status: Confirmed

The booking has been saved to your account and you can retrieve it using the booking management system.""",

    # Query 4: Show current flight bookings
    """Your current flight bookings:

Booking #1:
- Booking ID: FL0725ADB86B00
- Route: LAX → JFK
- Date: 2025-07-25
- Passengers: 1  
- Class: economy
- Total: $250.00
- Status: confirmed

Additional bookings may be available. Use the booking retrieval system to view all your confirmed reservations with complete details including booking dates and flight information.""",

    # Query 5: SpiceJet service quality reviews
    """Passenger reviews about SpiceJet's service quality show mixed experiences:

Positive Reviews:
- "A pleasant journey" - Air hostesses described as kind and helpful, with great hospitality for senior citizens (Rating: 10.0/10)
- Excellent customer care executives who helped resolve booking mistakes and provided great support
- Perfect onboard crew who provided care during medical issues, with Ms Rafat specifically mentioned for helping with medicines (Rating: 8.0/10)
- Professional, respectful flight attendants who behaved professionally on Hyderabad to Kishangarh route (Rating: 6.0/10)

Negative Reviews:
- Poor customer service with frequently changed flight timings and poor communication (Rating: 3.0/10)
- Flight delays up to 10 hours with no apology or refreshments offered
- Timeline management struggles noted as a primary concern

Overall, SpiceJet receives praise for crew professionalism and customer care, but faces criticism for punctuality and communication issues.""",
]

# Create dictionary for backward compatibility
QUERY_REFERENCE_ANSWERS = {
    query: answer for query, answer in zip(FLIGHT_SEARCH_QUERIES, FLIGHT_REFERENCE_ANSWERS)
}

def get_test_queries():
    """Return test queries for evaluation."""
    return FLIGHT_SEARCH_QUERIES

def get_evaluation_queries():
    """Get queries for evaluation"""
    return FLIGHT_SEARCH_QUERIES

def get_all_queries():
    """Get all available queries"""
    return FLIGHT_SEARCH_QUERIES

def get_simple_queries():
    """Get simple queries for basic testing"""
    return FLIGHT_SEARCH_QUERIES

def get_flight_policy_queries():
    """Return flight policy queries (for backward compatibility)."""
    return FLIGHT_SEARCH_QUERIES

def get_reference_answer(query: str) -> str:
    """Get the correct reference answer for a given query"""
    return QUERY_REFERENCE_ANSWERS.get(query, f"No reference answer available for: {query}")

def get_all_query_references():
    """Get all query-reference pairs"""
    return QUERY_REFERENCE_ANSWERS
