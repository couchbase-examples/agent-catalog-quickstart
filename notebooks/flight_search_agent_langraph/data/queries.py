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
    """There are several flights available from JFK to LAX: AS flight using 321 762, B6 flight using 320, DL flight using 76W 752, QF flight using 744, AA flight using 32B 762, UA flight using 757, US flight using 32B 762, and VX flight using 320.""",

    # Query 2: Flight booking LAX to JFK
    """Flight Booking Confirmed!

Booking ID: FL0718575D3BEE
Route: LAX → JFK
Departure Date: 2025-07-18
Passengers: 1
Class: economy
Total Price: $250.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 3: Flight booking JFK to MIA
    """Flight Booking Confirmed!

Booking ID: FL07248898793D
Route: JFK → MIA
Departure Date: 2025-07-24
Passengers: 1
Class: economy
Total Price: $250.00

Next Steps:
1. Check-in opens 24 hours before departure
2. Arrive at airport 2 hours early for domestic flights  
3. Bring valid government-issued photo ID

Thank you for choosing our airline!""",

    # Query 4: Show current bookings
    """Your current flight bookings are as follows:

1. Booking ID: FL07248898793D
   - Route: JFK → MIA
   - Date: 2025-07-24
   - Passengers: 1
   - Class: economy
   - Total: $250.00
   - Status: confirmed
   - Booked: 2025-07-17

2. Booking ID: FL0718575D3BEE
   - Route: LAX → JFK
   - Date: 2025-07-18
   - Passengers: 1
   - Class: economy
   - Total: $250.00
   - Status: confirmed
   - Booked: 2025-07-17""",

    # Query 5: SpiceJet reviews
    """Found 5 relevant airline reviews for SpiceJet service quality:

Review 1:
Airline: SpiceJet. Title: "a pleasant journey". Review: ✅ Trip Verified |It was a pleasant journey on this SpiceJet flight. Air-hostess are so kind and helpful. Supported well for senior citizens with great hospitality. Thanks to SpiceJet team.. Rating: 10.0/10. Reviewer: Thyagaraju Palisetty. Date: 18th April 2024. Recommended: yes

Review 2:
Airline: SpiceJet. Title: "SpiceJet experience was good". Review: Not Verified | SpiceJet experience was good. The crew members service and behaviour was also good. I can rate my flight experience on SpiceJet airlines flight 10/10.. Rating: 10.0/10. Reviewer: Apurba Arun. Date: 21st September 2023. Recommended: yes

Review 3:
Airline: SpiceJet. Title: "a joy to have on board". Review: Not Verified | I would like to highly compliment you on one of your flight attendants, Devaiah Napanda, he was the economy class flight attendant on SpiceJet on 24th March 2024 from Ayodya to Bengaluru. He was attentive, competent, pleasant, helpful, humorous, cheerful, professional - just a joy to have on board. Thank you so much.. Rating: 10.0/10. Reviewer: Ramya Moolemajalu Vishwanatha. Date: 18th April 2024. Recommended: yes

Review 4:
Airline: SpiceJet. Title: "flight was delayed by 10 hours". Review: Not Verified |  I had booked five to and fro tickets to Srinagar from Delhi and back for 27 Aug 24 and 07 Sep 24. I am very disappointed to state that on both occasions the flights were more than 6 hours delayed. The return flight was delayed by 10 hours. There was no apology from the airline or even offer for some water and snack...

Review 5:
Airline: SpiceJet. Title: SpiceJet customer review. Review: Flight to Kolkata with Spicejet and return back to Delhi was the best. Comfortable and fast option. For my next flight to Kolkata I will for sure choose this Airline.. Rating: 6.0/10. Reviewer: R Martin. Date: 21st April 2019. Recommended: yes""",
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
