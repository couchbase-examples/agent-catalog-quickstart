"""
Shared hotel search queries for both evaluation and testing.
"""

# Simple test queries based on actual cities in our travel-sample data
HOTEL_SEARCH_QUERIES = [
    "Find me a hotel in Los Angeles with free parking",
    "I need a budget hotel in San Diego with free breakfast", 
    "Show me luxury hotels in London with great reviews",
    "Find hotels in Paris with free Wi-Fi and breakfast",
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
