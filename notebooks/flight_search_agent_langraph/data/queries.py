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

Booking ID: FL08061563CACD
Route: LAX → JFK
Departure Date: 2025-08-06
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

Booking ID: FL08124E7B9C2A
Route: JFK → MIA
Departure Date: 2025-08-12
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
  Booking ID: FL08061563CACD
  Route: LAX → JFK
  Date: 2025-08-06
  Passengers: 2
  Class: business
  Total: $1500.00
  Status: confirmed
  Booked: 2025-08-05

Booking 2:
  Booking ID: FL08124E7B9C2A
  Route: JFK → MIA
  Date: 2025-08-12
  Passengers: 1
  Class: economy
  Total: $250.00
  Status: confirmed
  Booked: 2025-08-05""",

    # Query 5: SpiceJet service quality reviews
    """Found 5 relevant airline reviews for 'SpiceJet service':

Review 1:
Airline: SpiceJet. Title: "Service is impeccable". Review: ✅ Trip Verified | Much better than airbus models. Even the basic economy class has ambient lighting. Better personal air vents and better spotlights. Even overhead storage bins are good. Service is impeccable with proper care taken of guests...

Review 2:
Airline: SpiceJet. Title: "good service by the crew". Review: ✅ Trip Verified | I have had good service by the crew. It was amazing, the crew was very enthusiastic and warm welcome. It was one of the best services in my experience.. Rating: 10.0/10. Reviewer: K Mansour. Date: 10th August 2024. Recom...

Review 3:
Airline: SpiceJet. Title: "outstanding service I experienced". Review: Not Verified |  I wanted to take a moment to express my sincere thanks for the outstanding service I experienced on my recent flight from Pune to Delhi. SG-8937. From the moment I boarded, the warmth and friendliness of the air h...

Review 4:
Airline: SpiceJet. Title: "efficient and warm onboard service". Review: ✅ Trip Verified |  New Delhi to Kolkata. Delighted with the prompt, efficient and warm onboard service provided by the crew. Appreciate their efforts towards customer centricity.. Rating: 10.0/10. Reviewer: Debashis Roy. Date: 2...

Review 5:
Airline: SpiceJet. Title: "Service is very good". Review: Service is very good,  I am impressed with Miss Renu  who gave the best services ever. Thanks to Renu who is very sweet by her nature as well as her service. Rating: 9.0/10. Reviewer: Sanjay Patnaik. Date: 21st September 2023. Recommended: ye...""",
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
