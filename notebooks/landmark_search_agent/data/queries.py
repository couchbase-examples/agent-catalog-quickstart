#!/usr/bin/env python3
"""
Landmark Search Queries for Evaluation

This module contains test queries and reference answers for evaluating
the landmark search agent using travel-sample.inventory.landmark data.

These queries are designed to be diverse in vector space to test different
types of landmark searches across various categories and locations.
"""

from typing import Dict, List

# 5 diverse test queries for landmark search evaluation based on REAL travel-sample data
# These queries are verified to return actual results from the database
LANDMARK_SEARCH_QUERIES = [
    "Find museums and galleries in Glasgow",  # Art & Culture, Scotland
    "Show me restaurants serving Asian cuisine",  # Food & Dining, Real Asian restaurants
    "What attractions can I see in Glasgow?",  # General sightseeing, Scotland
    "Tell me about Monet's House",  # Specific landmark, France
    "Find places to eat in Gillingham",  # Food, Real UK town
]

# Reference answers based on ACTUAL agent responses from travel-sample.inventory.landmark data
# These match what the agent ACTUALLY finds and returns from the real database
QUERY_REFERENCE_ANSWERS = {
    "Find museums and galleries in Glasgow": """Glasgow has several museums and galleries including the Gallery of Modern Art (Glasgow) located at Royal Exchange Square with a terrific collection of recent paintings and sculptures, the Kelvingrove Art Gallery and Museum on Argyle Street with one of the finest civic collections in Europe including works by Van Gogh, Monet and Rembrandt, the Hunterian Museum and Art Gallery at University of Glasgow with a world famous Whistler collection, and the Riverside Museum at 100 Pointhouse Place with an excellent collection of vehicles and transport history. All offer free admission except for special exhibitions.""",
    "Show me restaurants serving Asian cuisine": """There are several Asian restaurants available including Shangri-la Chinese Restaurant in Birmingham at 51 Station Street offering good quality Chinese food with spring rolls and sizzling steak, Taiwan Restaurant in San Francisco famous for their dumplings, Hong Kong Seafood Restaurant in San Francisco for sit-down dim sum, Cheung Hing Chinese Restaurant in San Francisco for Cantonese BBQ and roast duck, Vietnam Restaurant in San Francisco for Vietnamese dishes including crab soup and pork sandwich, and various other Chinese and Asian establishments across different locations.""",
    "What attractions can I see in Glasgow?": """Glasgow attractions include Glasgow Green (founded by Royal grant in 1450) with Nelson's Memorial and the Doulton Fountain, Glasgow University (founded 1451) with neo-Gothic architecture and commanding views, Glasgow Cathedral with fine Gothic architecture from medieval times, the City Chambers in George Square built in 1888 in Italian Renaissance style with guided tours available, Glasgow Central Station with its grand interior, and Kelvingrove Park which is popular with students and contains the Art Gallery and Museum.""",
    "Tell me about Monet's House": """Monet's House is located in Giverny, France at 84 rue Claude Monet. The house is quietly eccentric and highly interesting in an Orient-influenced style, featuring Monet's collection of Japanese prints. The main attraction is the gardens around the house, including the water garden with the Japanese bridge, weeping willows and waterlilies which are now iconic. It's open April-October, Monday-Sunday 9:30-18:00, with admission €9 for adults, €5 for students, €4 for disabled visitors, and free for under-7s. E-tickets can be purchased online and wheelchair access is available.""",
    "Find places to eat in Gillingham": """Gillingham has various dining options including Beijing Inn (Chinese restaurant at 3 King Street), Spice Court (Indian restaurant at 56-58 Balmoral Road opposite the railway station, award-winning with Sunday Buffet for £8.50), Hollywood Bowl (American-style restaurant at 4 High Street with burgers and ribs in a Hollywood-themed setting), Ossie's Fish and Chips (at 75 Richmond Road, known for the best fish and chips in the area), and Thai Won Mien (oriental restaurant at 59-61 High Street with noodles, duck and other oriental dishes).""",
}

# Category-based queries for testing specific search capabilities (based on real data)
CATEGORY_QUERIES = {
    "cultural": [
        "Find museums and galleries in Glasgow",
        "Show me historic buildings and architecture",
        "What art collections can I visit?",
    ],
    "culinary": [
        "Show me restaurants serving Asian cuisine",
        "Find places to eat in Gillingham",
        "What dining options are available?",
    ],
    "sightseeing": [
        "What attractions can I see in Glasgow?",
        "Show me historic landmarks and buildings",
        "Find interesting places to visit",
    ],
    "specific": [
        "Tell me about Monet's House",
        "Show me the Glasgow Cathedral",
        "What can you tell me about the Burrell Collection?",
    ],
}

# Location-based queries for geographic diversity testing (based on real data)
LOCATION_QUERIES = {
    "Scotland": [
        "Find museums and galleries in Glasgow",
        "What attractions can I see in Glasgow?",
        "Show me historic buildings in Glasgow",
    ],
    "England": [
        "Find places to eat in Gillingham",
        "Show me restaurants serving Asian cuisine",
        "What landmarks are in Gillingham?",
    ],
    "France": [
        "Tell me about Monet's House",
        "Show me attractions in Giverny",
        "What can I visit in France?",
    ],
    "UK_General": [
        "Find attractions in the United Kingdom",
        "Show me places to visit in the UK",
        "What can I see in Britain?",
    ],
}

# Activity-based queries for testing different search patterns
ACTIVITY_QUERIES = [
    "What can I see in Glasgow?",  # 'see' activity queries
    "Where can I eat in Gillingham?",  # 'eat' activity queries
    "Show me places to dine",  # Generic eating queries
    "Find things to visit and see",  # Generic sightseeing queries
    "What museums can I visit?",  # Specific venue type queries
]


def get_all_queries() -> List[str]:
    """Get all queries for comprehensive testing."""
    all_queries = LANDMARK_SEARCH_QUERIES.copy()

    # Add category queries
    for category_list in CATEGORY_QUERIES.values():
        all_queries.extend(category_list)

    # Add location queries
    for location_list in LOCATION_QUERIES.values():
        all_queries.extend(location_list)

    # Add activity queries
    all_queries.extend(ACTIVITY_QUERIES)

    return all_queries


def get_reference_answer(query: str) -> str:
    """Get reference answer for a specific query."""
    return QUERY_REFERENCE_ANSWERS.get(query, "No reference answer available for this query.")


def get_queries_by_category(category: str) -> List[str]:
    """Get queries filtered by category."""
    if category == "basic":
        return LANDMARK_SEARCH_QUERIES
    elif category == "category":
        return [q for queries in CATEGORY_QUERIES.values() for q in queries]
    elif category == "location":
        return [q for queries in LOCATION_QUERIES.values() for q in queries]
    elif category == "activity":
        return ACTIVITY_QUERIES
    else:
        return get_all_queries()


def get_queries_for_evaluation(limit: int = 5) -> List[str]:
    """Get a subset of queries for evaluation purposes."""
    return LANDMARK_SEARCH_QUERIES[:limit]


# Export commonly used items
__all__ = [
    "LANDMARK_SEARCH_QUERIES",
    "QUERY_REFERENCE_ANSWERS",
    "CATEGORY_QUERIES",
    "LOCATION_QUERIES",
    "ACTIVITY_QUERIES",
    "get_all_queries",
    "get_reference_answer",
    "get_queries_by_category",
    "get_queries_for_evaluation",
]
