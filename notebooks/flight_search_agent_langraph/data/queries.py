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

1. AS flight from JFK to LAX using 321 762
2. B6 flight from JFK to LAX using 320
3. DL flight from JFK to LAX using 76W 752
4. QF flight from JFK to LAX using 744
5. AA flight from JFK to LAX using 32B 762
6. UA flight from JFK to LAX using 757
7. US flight from JFK to LAX using 32B 762
8. VX flight from JFK to LAX using 320""",

    # Query 2: Flight booking LAX to JFK for tomorrow, 2 passengers, business class
    # Note: Departure date and Booking ID are dynamically generated based on current date
    """Flight Booking Confirmed!

Booking ID: [Dynamically Generated]
Route: LAX → JFK
Departure Date: [Tomorrow's Date - Dynamically Calculated]
Passengers: 2
Class: business
Total Price: $1500.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 3: Flight booking JFK to MIA for next week
    # Note: Departure date and Booking ID are dynamically generated based on current date
    """Flight Booking Confirmed!

Booking ID: [Dynamically Generated]
Route: JFK → MIA
Departure Date: [Next Week's Date - Dynamically Calculated]
Passengers: 1
Class: economy
Total Price: $250.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 4: Show current flight bookings
    # Note: Booking IDs and dates are dynamically generated
    """Your Current Bookings (2 found):

Booking 1:
  Booking ID: [Dynamically Generated]
  Route: LAX → JFK
  Date: [Tomorrow's Date - Dynamically Calculated]
  Passengers: 2
  Class: business
  Total: $1500.00
  Status: Confirmed
  Booked: [Current Date]

Booking 2:
  Booking ID: [Dynamically Generated]
  Route: JFK → MIA
  Date: [Next Week's Date - Dynamically Calculated]
  Passengers: 1
  Class: economy
  Total: $250.00
  Status: Confirmed
  Booked: [Current Date]""",

    # Query 5: SpiceJet service quality reviews
    # Note: Vector search results are non-deterministic and may return different reviews each time
    # from the database of 2210+ airline reviews. All results are valid SpiceJet reviews.
    """Found 5 relevant airline reviews for 'SpiceJet service quality':

[Reviews include customer feedback about SpiceJet's service quality, covering aspects such as:
- Crew service and hospitality
- Flight delays and punctuality
- In-flight amenities and comfort
- Ground staff service
- Overall travel experience
- Ratings typically range from 2/10 to 10/10
- Mix of verified and unverified reviews
- Reviews from various dates and routes
- Both positive recommendations and criticisms]

Note: Specific reviews vary due to vector similarity search across 2210+ airline reviews.""",
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
