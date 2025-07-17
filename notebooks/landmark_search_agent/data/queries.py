#!/usr/bin/env python3
"""
Landmark Search Queries for Evaluation

This module contains test queries and reference answers for evaluating
the landmark search agent using travel-sample.inventory.landmark data.

These queries are designed to be diverse in vector space to test different
types of landmark searches across various categories and locations.
"""

from typing import Dict, List

# 5 diverse test queries for landmark search evaluation
# These are designed to be far apart in vector space
LANDMARK_SEARCH_QUERIES = [
    "Find museums with art collections in London",           # Art & Culture, Europe
    "Show me outdoor hiking trails in national parks",       # Nature & Adventure, Outdoor
    "What are the best restaurants serving Asian cuisine",    # Food & Dining, Cuisine-specific  
    "Find historic churches and religious sites in Rome",    # Religious & Historic, Europe
    "Show me beaches and water activities in tropical locations"  # Beach & Water, Tropical
]

# Reference answers based on actual travel-sample.inventory.landmark data
# These should match what the agent would find in the database
QUERY_REFERENCE_ANSWERS = {
    "Find museums with art collections in London": """London offers several excellent museums with art collections including the British Museum with its extensive historical artifacts and sculptures, Tate Modern featuring contemporary art in a former power station, National Gallery housing masterpieces from European artists, and the Victoria and Albert Museum showcasing decorative arts and design. These museums provide comprehensive cultural experiences spanning different artistic periods and styles.""",
    
    "Show me outdoor hiking trails in national parks": """Popular hiking destinations include Yellowstone National Park with its geothermal features and wildlife viewing trails, Grand Canyon National Park offering rim and canyon hiking experiences, Yosemite National Park featuring waterfalls and granite cliff trails, and various state parks with marked hiking paths through forests, mountains, and scenic landscapes. These locations provide opportunities for different skill levels from easy walks to challenging multi-day treks.""",
    
    "What are the best restaurants serving Asian cuisine": """Top Asian dining experiences include authentic Chinese restaurants serving dim sum and regional specialties, Japanese sushi bars and ramen shops, Thai restaurants offering traditional curries and stir-fries, Korean BBQ establishments, Vietnamese pho restaurants, and Indian restaurants featuring diverse regional cuisines. Many cities also offer Asian fusion restaurants that blend traditional flavors with modern cooking techniques.""",
    
    "Find historic churches and religious sites in Rome": """Rome contains numerous significant religious landmarks including St. Peter's Basilica in Vatican City with its magnificent dome and art collections, the Pantheon as an ancient Roman temple converted to a church, Santa Maria Maggiore featuring beautiful mosaics, San Giovanni in Laterano as the Pope's cathedral church, and the Catacombs offering underground Christian burial sites with historical significance.""",
    
    "Show me beaches and water activities in tropical locations": """Tropical beach destinations offer activities such as snorkeling and diving in coral reefs, surfing on ocean waves, kayaking in clear lagoons, deep-sea fishing excursions, sailing and catamaran trips, parasailing for aerial views, jet skiing, and relaxing on white sand beaches. Popular locations include Caribbean islands, Hawaii, Southeast Asian coastal areas, and tropical Pacific islands with warm waters year-round."""
}

# Category-based queries for testing specific search capabilities
CATEGORY_QUERIES = {
    "cultural": [
        "Find museums with art collections in London",
        "Show me theaters and performance venues",
        "What cultural festivals happen annually?"
    ],
    "adventure": [
        "Show me outdoor hiking trails in national parks", 
        "Find mountain climbing and rock climbing spots",
        "What extreme sports activities are available?"
    ],
    "culinary": [
        "What are the best restaurants serving Asian cuisine",
        "Find local food markets and street food",
        "Show me wineries and vineyard tours"
    ],
    "religious": [
        "Find historic churches and religious sites in Rome",
        "Show me temples and spiritual retreat centers",
        "What pilgrimage routes are available?"
    ],
    "beach": [
        "Show me beaches and water activities in tropical locations",
        "Find coastal walks and seaside attractions",
        "What marine wildlife viewing opportunities exist?"
    ]
}

# Location-based queries for geographic diversity testing
LOCATION_QUERIES = {
    "Europe": [
        "Find museums with art collections in London",
        "Find historic churches and religious sites in Rome",
        "Show me castles and palaces in France"
    ],
    "Americas": [
        "Show me outdoor hiking trails in national parks",
        "Find historic sites in New York City",
        "What attractions are in San Francisco?"
    ],
    "Asia": [
        "What are the best restaurants serving Asian cuisine",
        "Find temples and gardens in Japan",
        "Show me markets and shopping in Bangkok"
    ],
    "Tropical": [
        "Show me beaches and water activities in tropical locations",
        "Find coral reefs for snorkeling and diving",
        "What island resort activities are available?"
    ]
}

# Complex multi-criteria queries for advanced testing
COMPLEX_QUERIES = [
    "Find family-friendly outdoor museums with educational programs",
    "Show me wheelchair accessible historic sites with guided tours", 
    "What are the best budget-friendly cultural attractions for students?",
    "Find romantic dining venues with scenic waterfront views",
    "Show me eco-friendly adventure activities in protected areas"
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

    # Add complex queries
    all_queries.extend(COMPLEX_QUERIES)

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
    elif category == "complex":
        return COMPLEX_QUERIES
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
    "COMPLEX_QUERIES",
    "get_all_queries",
    "get_reference_answer",
    "get_queries_by_category",
    "get_queries_for_evaluation",
]
