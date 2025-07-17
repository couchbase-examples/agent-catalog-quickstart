#!/usr/bin/env python3
"""
Landmark Search Queries for Evaluation

This module contains test queries and reference answers for evaluating
the landmark search agent using travel-sample.inventory.landmark data.
"""

from typing import Dict, List

# Test queries for landmark search evaluation
LANDMARK_SEARCH_QUERIES = [
    "Show me museums in Paris",
    "Find art galleries in New York",
    "What are the best landmarks in London?",
    "Show me parks and gardens in Tokyo",
    "Find historic sites in Rome",
    "What attractions are in Barcelona?",
    "Show me monuments in Washington DC",
    "Find outdoor activities in San Francisco",
    "What are the top sights in Amsterdam?",
    "Show me cultural sites in Berlin",
    "Find restaurants in San Francisco",
    "What are the best viewpoints in Edinburgh?",
    "Show me shopping areas in Tokyo",
    "Find beaches near Los Angeles",
    "What entertainment venues are in Las Vegas?",
    "Show me architectural landmarks in Chicago",
    "Find nature spots in Vancouver",
    "What are the must-see attractions in Sydney?",
    "Show me religious sites in Jerusalem",
    "Find family-friendly activities in Orlando",
]

# Reference answers based on actual travel-sample.inventory.landmark data
# These should match what the agent would find in the database
QUERY_REFERENCE_ANSWERS = {
    "Show me museums in Paris": """Paris offers world-class museums including the Louvre Museum with its extensive art collection, Musée d'Orsay featuring impressionist masterpieces, and Centre Pompidou for contemporary art. These museums provide comprehensive cultural experiences in the heart of Paris.""",
    "Find art galleries in New York": """New York City features premier art destinations such as the Metropolitan Museum of Art with its vast collections, Museum of Modern Art (MoMA) showcasing contemporary works, Guggenheim Museum with its distinctive architecture, and Whitney Museum focusing on American art.""",
    "What are the best landmarks in London?": """London's iconic landmarks include Big Ben and the Houses of Parliament, Tower Bridge spanning the Thames, the London Eye offering panoramic views, Buckingham Palace as the royal residence, Westminster Abbey with its historical significance, and the Tower of London housing the Crown Jewels.""",
    "Show me parks and gardens in Tokyo": """Tokyo provides beautiful green spaces including Ueno Park known for cherry blossoms, Shinjuku Gyoen with its diverse gardens, Imperial Palace East Gardens in the city center, and Yoyogi Park offering recreational activities near Shibuya.""",
    "Find historic sites in Rome": """Rome contains ancient landmarks such as the Colosseum showcasing gladiatorial history, Roman Forum as the center of ancient Rome, Pantheon with its impressive dome, Trevi Fountain for making wishes, and Vatican City with St. Peter's Basilica and the Sistine Chapel.""",
    "What attractions are in Barcelona?": """Barcelona features unique attractions including Sagrada Familia with Gaudí's distinctive architecture, Park Güell offering colorful mosaics and city views, Casa Batlló showcasing modernist design, Gothic Quarter with medieval streets, and Las Ramblas for vibrant street life.""",
    "Show me monuments in Washington DC": """Washington DC's monuments include the Lincoln Memorial honoring the 16th president, Washington Monument as the city's centerpiece, Jefferson Memorial beside the Tidal Basin, Vietnam Veterans Memorial with its moving wall, and the Capitol Building housing Congress.""",
    "Find outdoor activities in San Francisco": """San Francisco offers outdoor experiences at Golden Gate Park with its museums and gardens, Alcatraz Island for historical tours, Fisherman's Wharf with sea lions and dining, Golden Gate Bridge for walking and cycling, and nearby Muir Woods for hiking among redwoods.""",
    "What are the top sights in Amsterdam?": """Amsterdam's top attractions include the Anne Frank House preserving wartime history, Van Gogh Museum with the artist's masterpieces, Rijksmuseum showcasing Dutch art and history, Vondelpark for outdoor relaxation, and the historic canal ring offering scenic boat tours.""",
    "Show me cultural sites in Berlin": """Berlin's cultural landmarks include Brandenburg Gate symbolizing German reunification, Museum Island with world-class museums, East Side Gallery featuring Berlin Wall art, Pergamon Museum with ancient artifacts, and the Holocaust Memorial for historical remembrance.""",
    "Find restaurants in San Francisco": """San Francisco offers diverse dining experiences from Michelin-starred restaurants to local favorites, featuring fresh seafood at Fisherman's Wharf, innovative cuisine in the Mission District, authentic dim sum in Chinatown, and farm-to-table dining throughout the city.""",
    "What are the best viewpoints in Edinburgh?": """Edinburgh provides spectacular views from Edinburgh Castle overlooking the city, Arthur's Seat offering panoramic vistas, Calton Hill with its monuments, Scott Monument on Princes Street, and the Royal Mile connecting the castle to Holyrood Palace.""",
    "Show me shopping areas in Tokyo": """Tokyo features premier shopping districts including Shibuya for fashion and electronics, Ginza for luxury brands, Harajuku for youth culture and unique styles, Akihabara for electronics and anime merchandise, and Omotesando for high-end designer stores.""",
    "Find beaches near Los Angeles": """Los Angeles area beaches include Santa Monica Beach with its iconic pier, Venice Beach known for its boardwalk and street performers, Manhattan Beach offering upscale coastal living, Redondo Beach for family activities, and Malibu for celebrity spotting and scenic beauty.""",
    "What entertainment venues are in Las Vegas?": """Las Vegas entertainment includes world-class casinos along the Strip, spectacular shows featuring magic and music, themed hotels like Bellagio and Venetian, the High Roller observation wheel, and the Fremont Street Experience in downtown Las Vegas.""",
    "Show me architectural landmarks in Chicago": """Chicago's architectural highlights include Willis Tower (formerly Sears Tower) for city views, Frank Lloyd Wright's architecture in Oak Park, Millennium Park with Cloud Gate sculpture, Navy Pier for entertainment, and the Art Institute of Chicago housing impressive collections.""",
    "Find nature spots in Vancouver": """Vancouver offers natural attractions including Stanley Park with its seawall and forests, Queen Elizabeth Park featuring gardens and city views, VanDusen Botanical Garden with diverse plant collections, and nearby mountains for hiking and skiing activities.""",
    "What are the must-see attractions in Sydney?": """Sydney's iconic attractions include the Sydney Opera House with its distinctive architecture, Sydney Harbour Bridge for climbing and views, Bondi Beach for surfing and sunbathing, Royal Botanic Gardens for peaceful walks, and The Rocks area for historical exploration.""",
    "Show me religious sites in Jerusalem": """Jerusalem contains sacred sites including the Western Wall for Jewish prayer, Church of the Holy Sepulchre marking Jesus's crucifixion site, Al-Aqsa Mosque for Islamic worship, Via Dolorosa retracing Jesus's path, and the Garden of Gethsemane with ancient olive trees.""",
    "Find family-friendly activities in Orlando": """Orlando offers family attractions including Walt Disney World with its themed parks, Universal Studios for movie-based rides, SeaWorld for marine shows, ICON Park for dining and entertainment, and Gatorland for wildlife encounters and educational experiences.""",
}

# Activity-based queries for testing specific search capabilities
ACTIVITY_QUERIES = {
    "see": [
        "What can I see in London?",
        "Show me sights to see in Paris",
        "What viewing spots are available in Tokyo?",
    ],
    "do": [
        "What activities can I do in Barcelona?",
        "Show me things to do in New York",
        "What experiences are available in Rome?",
    ],
    "eat": [
        "Where can I eat in San Francisco?",
        "Show me dining options in Tokyo",
        "What restaurants are in Paris?",
    ],
    "buy": [
        "Where can I shop in New York?",
        "Show me shopping areas in Tokyo",
        "What markets are in Barcelona?",
    ],
}

# City-based queries for location-specific testing
CITY_QUERIES = {
    "Paris": [
        "Show me landmarks in Paris",
        "What attractions are in Paris?",
        "Find museums in Paris",
    ],
    "London": [
        "What can I visit in London?",
        "Show me London attractions",
        "Find historic sites in London",
    ],
    "Tokyo": [
        "What sights are in Tokyo?",
        "Show me Tokyo landmarks",
        "Find cultural sites in Tokyo",
    ],
    "New York": [
        "What attractions are in New York?",
        "Show me NYC landmarks",
        "Find art galleries in New York",
    ],
    "Rome": ["What can I see in Rome?", "Show me Roman landmarks", "Find ancient sites in Rome"],
}

# Complex queries for testing advanced search capabilities
COMPLEX_QUERIES = [
    "Find outdoor museums with historical significance",
    "Show me family-friendly attractions with educational value",
    "What are the best free attractions in major cities?",
    "Find landmarks that are accessible by public transportation",
    "Show me attractions that are open in the evening",
    "What are the most photographed landmarks in Europe?",
    "Find attractions suitable for rainy weather",
    "Show me landmarks with guided tour options",
    "What are the best attractions for art lovers?",
    "Find historic sites with interactive exhibits",
]


def get_all_queries() -> List[str]:
    """Get all queries for comprehensive testing."""
    all_queries = LANDMARK_SEARCH_QUERIES.copy()

    # Add activity queries
    for activity_list in ACTIVITY_QUERIES.values():
        all_queries.extend(activity_list)

    # Add city queries
    for city_list in CITY_QUERIES.values():
        all_queries.extend(city_list)

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
    elif category == "activity":
        return [q for queries in ACTIVITY_QUERIES.values() for q in queries]
    elif category == "city":
        return [q for queries in CITY_QUERIES.values() for q in queries]
    elif category == "complex":
        return COMPLEX_QUERIES
    else:
        return get_all_queries()


def get_queries_for_evaluation(limit: int = 10) -> List[str]:
    """Get a subset of queries for evaluation purposes."""
    return LANDMARK_SEARCH_QUERIES[:limit]


# Export commonly used items
__all__ = [
    "LANDMARK_SEARCH_QUERIES",
    "QUERY_REFERENCE_ANSWERS",
    "ACTIVITY_QUERIES",
    "CITY_QUERIES",
    "COMPLEX_QUERIES",
    "get_all_queries",
    "get_reference_answer",
    "get_queries_by_category",
    "get_queries_for_evaluation",
]
