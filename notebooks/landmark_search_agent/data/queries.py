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

# Reference answers based on ACTUAL travel-sample.inventory.landmark data
# These match what the agent will find in the real database
QUERY_REFERENCE_ANSWERS = {
    "Find museums and galleries in Glasgow": """Glasgow offers excellent museums including the Gallery of Modern Art with its terrific collection of recent paintings and sculptures, the Burrell Collection featuring over 9,000 artworks gifted by Sir William Burrell, the Hunterian Museum and Art Gallery with its world-famous Whistler collection, and the Glasgow Police Museum showcasing the world's first police force dating back to 1779. The Glasgow Science Centre provides interactive exhibits and an IMAX cinema.""",
    "Show me restaurants serving Asian cuisine": """You can find authentic Asian cuisine at Thai Won Mien, a really popular oriental restaurant with noodles, duck and other oriental specialties, Beijing Inn offering Chinese food just off the High Street, and Spice Court, an award-winning Indian restaurant opposite the railway station known for good value and quality food.""",
    "What attractions can I see in Glasgow?": """Glasgow offers numerous attractions including the magnificent Glasgow Cathedral with Gothic architecture from medieval times, the impressive City Chambers built in 1888 in Italian Renaissance style, Glasgow Cross marking the original medieval city center, and various bridges like the elegant Clyde Arc and the nicknamed 'Squiggly Bridge'. You can also visit Glasgow University (founded 1451), the distinctive Clyde Auditorium known as 'the Armadillo', and historic areas like Park Circus with Georgian townhouses.""",
    "Tell me about Monet's House": """Monet's House in Giverny, France is quietly eccentric and highly interesting in an Orient-influenced style. The house includes Monet's famous gardens and provides insight into the artist's life and work. It's a significant cultural landmark for art enthusiasts visiting France.""",
    "Find places to eat in Gillingham": """Gillingham offers diverse dining options including Hollywood Bowl, a newly extended lively American Hollywood-style restaurant in the high street, Thai Won Mien for oriental cuisine, Spice Court serving Indian food opposite the railway station, Beijing Inn for Chinese food, and Ossie's Fish and Chips for the best fish and chips in the area.""",
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
