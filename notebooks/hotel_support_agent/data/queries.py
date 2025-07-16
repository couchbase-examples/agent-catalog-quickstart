"""
Shared hotel search queries for both evaluation and testing.
"""

# Updated queries based on actual travel-sample data
HOTEL_SEARCH_QUERIES = [
    "Find me a hotel in San Francisco with free parking and breakfast",
    "I need a hotel in London with free internet access", 
    "Show me hotels in Paris with free breakfast",
]

def get_evaluation_queries():
    """Get queries for evaluation"""
    return HOTEL_SEARCH_QUERIES

def get_all_queries():
    """Get all available queries"""
    return HOTEL_SEARCH_QUERIES

def get_simple_queries():
    """Get simple queries for basic testing"""
    return HOTEL_SEARCH_QUERIES
