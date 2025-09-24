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
    """Flight Booking Confirmed!

Booking ID: FL09251563CACD
Route: LAX → JFK
Departure Date: 2025-09-25
Passengers: 2
Class: business
Total Price: $1500.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 3: Flight booking JFK to MIA for next week
    """Flight Booking Confirmed!

Booking ID: FL10014E7B9C2A
Route: JFK → MIA
Departure Date: 2025-10-01
Passengers: 1
Class: economy
Total Price: $250.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 4: Show current flight bookings
    """Your Current Bookings (2 found):

Booking 1:
  Booking ID: FL09251563CACD
  Route: LAX → JFK
  Date: 2025-09-25
  Passengers: 2
  Class: business
  Total: $1500.00
  Status: confirmed
  Booked: 2025-09-24

Booking 2:
  Booking ID: FL10014E7B9C2A
  Route: JFK → MIA
  Date: 2025-10-01
  Passengers: 1
  Class: economy
  Total: $250.00
  Status: confirmed
  Booked: 2025-09-24""",

    # Query 5: SpiceJet service quality reviews
    """Found 5 relevant airline reviews for 'SpiceJet service quality':

Review 1:
Airline: SpiceJet. Title: "Great travel experience". Review: ✅ Trip Verified |  Marvelous courteous crew who took good care of all passengers. They should be rewarded for the patience shown towards the unruly ones. Great travel experience.. Rating: 10.0/10. Reviewer: Ranjita Pandey. Date: 18th April 2024. Recommended: yes

Review 2:
Airline: SpiceJet. Title: "good service by the crew". Review: ✅ Trip Verified | I have had good service by the crew. It was amazing, the crew was very enthusiastic and warm welcome. It was one of the best services in my experience.. Rating: 10.0/10. Reviewer: K Mansour. Date: 10th August 2024. Recommended: yes

Review 3:
Airline: SpiceJet. Title: "delayed both ways by many hours". Review: Not Verified |  Flight was delayed both ways by many hours. Poor service for the same price as other airlines like IndiGo. No wifi or other amenities to compensate for terrible service.. Rating: 2.0/10. Reviewer: Somil Jain Jain. Date: 20th May 2022. Recommended: no

Review 4:
Airline: SpiceJet. Title: "Excellent service". Review: ✅ Trip Verified |  Excellent service by the ground staff courteous beyond expectations always willing to help in the real sense and not lipservice i will recommend to all whom I know. Rating: 10.0/10. Reviewer: Ramanathan Ramchandra. Date: 1st November 2023. Recommended: yes

Review 5:
Airline: SpiceJet. Title: "hospitality service given". Review: ✅ Trip Verified |  Seats are comparable if compare to other budget airlines. Cabin crews are friendly and hospitality service given with smiles. Very happy and enjoy experience.. Rating: 8.0/10. Reviewer: A Bameen. Date: 20th March 2022. Recommended: yes""",
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
