"""
Shared hotel search queries for both evaluation and testing.
"""

# Updated queries based on actual travel-sample data
HOTEL_SEARCH_QUERIES = [
    "Find me a hotel in San Francisco with free parking and breakfast",
    "I need a hotel in London with free internet access", 
    "Show me hotels in Paris with free breakfast",
]

# Reference answers based on actual search results from the database
QUERY_REFERENCE_ANSWERS = {
    "Find me a hotel in San Francisco with free parking and breakfast": """Found 1 hotel in San Francisco with partial matches:

Cow Hollow Motor Inn & Suites - San Francisco, United States
- Address: 2190 Lombard St, San Francisco, CA
- Free breakfast: Yes
- Free parking: No
- Price: $79-$145
- Phone: +1 415 921-5800
- Website: http://www.cowhollowmotorinn.com/

Note: This hotel offers free breakfast but does not offer free parking. No hotels were found in San Francisco that offer both free parking and free breakfast.""",

    "I need a hotel in London with free internet access": """Found 1 hotel in London with free internet access:

Chelsea House Hotel - London, United Kingdom
- Address: 96 Redcliffe Gdn, SW10 9HH, London
- Free internet: Yes
- Free breakfast: Yes
- Free parking: Yes
- Phone: +44 20 7835-1551
- Email: info@chelsea-house.co.uk
- Website: http://www.chelsea-house.co.uk/
- Location: South Kensington-Chelsea area, 5 min from Earl's Court Underground""",

    "Show me hotels in Paris with free breakfast": """Found 1 hotel in Paris with free breakfast:

Fraser Suites Harmonie Paris La Défense - Courbevoie, France
- Location: Courbevoie, Île-de-France, France (Paris/La Défense area)
- Free breakfast: Yes
- Free parking: Yes
- Free internet: No
- Website: http://paris.frasershospitality.com/
- Description: Higher-end apart-hotel close to the Esplanade de la Defense metro station
- Coordinates: 48.88819, 2.25211"""
}

def get_evaluation_queries():
    """Get queries for evaluation"""
    return HOTEL_SEARCH_QUERIES

def get_all_queries():
    """Get all available queries"""
    return HOTEL_SEARCH_QUERIES

def get_simple_queries():
    """Get simple queries for basic testing"""
    return HOTEL_SEARCH_QUERIES

def get_reference_answer(query: str) -> str:
    """Get the correct reference answer for a given query"""
    return QUERY_REFERENCE_ANSWERS.get(query, f"No reference answer available for: {query}")

def get_all_query_references():
    """Get all query-reference pairs"""
    return QUERY_REFERENCE_ANSWERS
