"""
Points of Interest (POI) Search Tool for Agent Catalog

Finds restaurants, attractions, gas stations, and other POIs along routes.
"""

import agentc

# Sample POI data organized by region and category
POINTS_OF_INTEREST = {
    "california": {
        "restaurants": [
            {
                "name": "The French Laundry",
                "location": "Yountville, CA",
                "category": "Fine Dining",
                "description": "World-renowned Michelin 3-star restaurant in Napa Valley",
                "price_range": "$$$$",
                "specialties": ["French cuisine", "Tasting menu", "Wine pairing"]
            },
            {
                "name": "In-N-Out Burger",
                "location": "Multiple CA locations",
                "category": "Fast Food",
                "description": "Iconic California burger chain with fresh ingredients",
                "price_range": "$",
                "specialties": ["Burgers", "Animal style", "Fresh fries"]
            },
            {
                "name": "Fisherman's Wharf Restaurants",
                "location": "San Francisco, CA",
                "category": "Seafood",
                "description": "Collection of seafood restaurants with bay views",
                "price_range": "$$-$$$",
                "specialties": ["Dungeness crab", "Clam chowder", "Fresh fish"]
            }
        ],
        "attractions": [
            {
                "name": "Golden Gate Bridge",
                "location": "San Francisco, CA",
                "category": "Landmark",
                "description": "Iconic suspension bridge and symbol of San Francisco",
                "duration": "1-2 hours",
                "activities": ["Photography", "Walking", "Biking", "Views"]
            },
            {
                "name": "Hearst Castle",
                "location": "San Simeon, CA",
                "category": "Historic Site",
                "description": "Opulent mansion on Big Sur coast with guided tours",
                "duration": "3-4 hours",
                "activities": ["Tours", "Architecture", "Gardens", "History"]
            },
            {
                "name": "Monterey Bay Aquarium",
                "location": "Monterey, CA",
                "category": "Aquarium",
                "description": "World-class aquarium showcasing marine life",
                "duration": "3-4 hours",
                "activities": ["Marine exhibits", "Kelp forest", "Sea otters"]
            }
        ],
        "accommodations": [
            {
                "name": "Big Sur Lodge",
                "location": "Big Sur, CA",
                "category": "Nature Lodge",
                "description": "Rustic lodge in redwood forest setting",
                "price_range": "$$$",
                "amenities": ["Restaurant", "Hiking trails", "Pool", "Spa"]
            },
            {
                "name": "Hotel del Coronado",
                "location": "San Diego, CA",
                "category": "Historic Resort",
                "description": "Victorian beachfront resort with historic charm",
                "price_range": "$$$$",
                "amenities": ["Beach access", "Multiple restaurants", "Spa", "Golf"]
            }
        ]
    },

    "northeast": {
        "restaurants": [
            {
                "name": "Katz's Delicatessen",
                "location": "New York, NY",
                "category": "Deli",
                "description": "Historic Jewish deli famous for pastrami sandwiches",
                "price_range": "$$",
                "specialties": ["Pastrami", "Corned beef", "Pickles"]
            },
            {
                "name": "Legal Sea Foods",
                "location": "Boston, MA",
                "category": "Seafood",
                "description": "New England seafood chain known for fresh catch",
                "price_range": "$$$",
                "specialties": ["Lobster", "Clam chowder", "Fresh fish"]
            }
        ],
        "attractions": [
            {
                "name": "Statue of Liberty",
                "location": "New York, NY",
                "category": "Monument",
                "description": "Symbol of freedom and democracy in New York Harbor",
                "duration": "4-6 hours",
                "activities": ["Ferry ride", "Museum", "Crown access", "Ellis Island"]
            },
            {
                "name": "Freedom Trail",
                "location": "Boston, MA",
                "category": "Historic Walk",
                "description": "2.5-mile trail connecting historic Revolutionary War sites",
                "duration": "2-4 hours",
                "activities": ["Walking tour", "Historic sites", "Museums"]
            }
        ]
    },

    "midwest": {
        "restaurants": [
            {
                "name": "Portillo's",
                "location": "Chicago area",
                "category": "Chicago Style",
                "description": "Chicago-style hot dogs, Italian beef, and chocolate cake",
                "price_range": "$$",
                "specialties": ["Italian beef", "Hot dogs", "Chocolate cake"]
            }
        ],
        "attractions": [
            {
                "name": "Millennium Park",
                "location": "Chicago, IL",
                "category": "Urban Park",
                "description": "Downtown park featuring Cloud Gate sculpture and Crown Fountain",
                "duration": "2-3 hours",
                "activities": ["Cloud Gate", "Crown Fountain", "Concerts", "Walking"]
            }
        ]
    }
}

# Gas station chains by region
GAS_STATIONS = {
    "national": ["Shell", "Chevron", "BP", "Exxon", "Mobil", "Speedway"],
    "california": ["Arco", "76", "Valero"],
    "northeast": ["Sunoco", "Gulf", "Getty"],
    "midwest": ["Casey's", "Kwik Trip", "Marathon"],
    "south": ["RaceTrac", "QuikTrip", "Buc-ee's"],
    "west": ["Sinclair", "Conoco", "Phillips 66"]
}


@agentc.catalog.tool
def find_restaurants(
    location: str,
    cuisine_type: str = "",
    price_range: str = "",
    max_results: int = 5
) -> str:
    """
    Find restaurants along a route or in a specific location.

    Args:
        location: City, region, or area to search
        cuisine_type: Type of cuisine (optional)
        price_range: Price range ($, $$, $$$, $$$$) (optional)
        max_results: Maximum number of results to return

    Returns:
        Formatted list of restaurant recommendations
    """
    try:
        # Determine region from location
        location_lower = location.lower()
        region = None

        if any(ca_city in location_lower for ca_city in ["california", "ca", "san francisco", "los angeles", "napa", "monterey"]):
            region = "california"
        elif any(ne_city in location_lower for ne_city in ["new york", "boston", "philadelphia", "ny", "ma"]):
            region = "northeast"
        elif any(mw_city in location_lower for mw_city in ["chicago", "milwaukee", "detroit", "il", "wi"]):
            region = "midwest"

        if not region or region not in POINTS_OF_INTEREST:
            return f"🍽️ Restaurant search not available for '{location}'. Try major cities like San Francisco, New York, or Chicago."

        restaurants = POINTS_OF_INTEREST[region]["restaurants"]

        # Filter by criteria
        filtered_restaurants = []
        for restaurant in restaurants:
            # Filter by cuisine type
            if cuisine_type and cuisine_type.lower() not in restaurant["category"].lower():
                continue

            # Filter by price range
            if price_range and restaurant["price_range"] != price_range:
                continue

            filtered_restaurants.append(restaurant)

        if not filtered_restaurants:
            return f"🍽️ No restaurants found matching criteria in {location}."

        # Limit results
        filtered_restaurants = filtered_restaurants[:max_results]

        # Format results
        result = f"🍽️ **Restaurant Recommendations for {location.title()}**\n"
        result += "=" * 60 + "\n"

        if cuisine_type:
            result += f"Cuisine: {cuisine_type.title()}\n"
        if price_range:
            result += f"Price Range: {price_range}\n"

        result += f"\nFound {len(filtered_restaurants)} restaurants:\n\n"

        for i, restaurant in enumerate(filtered_restaurants, 1):
            result += f"**{i}. {restaurant['name']}**\n"
            result += f"📍 Location: {restaurant['location']}\n"
            result += f"🍽️ Category: {restaurant['category']}\n"
            result += f"💰 Price: {restaurant['price_range']}\n"
            result += f"📝 Description: {restaurant['description']}\n"

            if restaurant.get("specialties"):
                result += f"⭐ Specialties: {', '.join(restaurant['specialties'])}\n"

            result += "\n" + "─" * 50 + "\n\n"

        return result

    except Exception as e:
        return f"Error finding restaurants: {e!s}"


@agentc.catalog.tool
def find_attractions(
    location: str,
    category: str = "",
    duration: str = "",
    max_results: int = 5
) -> str:
    """
    Find tourist attractions and points of interest.

    Args:
        location: City, region, or area to search
        category: Type of attraction (landmark, museum, park, etc.)
        duration: Preferred visit duration (optional)
        max_results: Maximum number of results

    Returns:
        Formatted list of attraction recommendations
    """
    try:
        # Determine region
        location_lower = location.lower()
        region = None

        if any(ca_city in location_lower for ca_city in ["california", "ca", "san francisco", "los angeles", "napa", "monterey"]):
            region = "california"
        elif any(ne_city in location_lower for ne_city in ["new york", "boston", "philadelphia", "ny", "ma"]):
            region = "northeast"
        elif any(mw_city in location_lower for mw_city in ["chicago", "milwaukee", "detroit", "il", "wi"]):
            region = "midwest"

        if not region or region not in POINTS_OF_INTEREST:
            return f"🎯 Attraction search not available for '{location}'. Try major cities like San Francisco, New York, or Chicago."

        attractions = POINTS_OF_INTEREST[region]["attractions"]

        # Filter by criteria
        filtered_attractions = []
        for attraction in attractions:
            # Filter by category
            if category and category.lower() not in attraction["category"].lower():
                continue

            filtered_attractions.append(attraction)

        if not filtered_attractions:
            return f"🎯 No attractions found matching criteria in {location}."

        # Limit results
        filtered_attractions = filtered_attractions[:max_results]

        # Format results
        result = f"🎯 **Attractions in {location.title()}**\n"
        result += "=" * 50 + "\n"

        if category:
            result += f"Category: {category.title()}\n"

        result += f"\nFound {len(filtered_attractions)} attractions:\n\n"

        for i, attraction in enumerate(filtered_attractions, 1):
            result += f"**{i}. {attraction['name']}**\n"
            result += f"📍 Location: {attraction['location']}\n"
            result += f"🎯 Category: {attraction['category']}\n"
            result += f"⏱️ Duration: {attraction['duration']}\n"
            result += f"📝 Description: {attraction['description']}\n"

            if attraction.get("activities"):
                result += f"🎪 Activities: {', '.join(attraction['activities'])}\n"

            result += "\n" + "─" * 40 + "\n\n"

        return result

    except Exception as e:
        return f"Error finding attractions: {e!s}"


@agentc.catalog.tool
def find_gas_stations(region: str) -> str:
    """
    Find gas station chains available in a specific region.

    Args:
        region: Geographic region (california, northeast, midwest, south, west)

    Returns:
        List of gas station chains in the region
    """
    try:
        region_lower = region.lower()

        # Get national chains (available everywhere)
        gas_chains = GAS_STATIONS["national"].copy()

        # Add regional chains
        if region_lower in GAS_STATIONS:
            gas_chains.extend(GAS_STATIONS[region_lower])

        result = f"⛽ **Gas Stations in {region.title()}**\n"
        result += "=" * 40 + "\n"

        result += "**National Chains (everywhere):**\n"
        for chain in GAS_STATIONS["national"]:
            result += f"   • {chain}\n"

        if region_lower in GAS_STATIONS:
            result += f"\n**Regional Chains ({region.title()}):**\n"
            for chain in GAS_STATIONS[region_lower]:
                result += f"   • {chain}\n"

        result += "\n💡 **Tips:**\n"
        result += "   • Use apps like GasBuddy for current prices\n"
        result += "   • Many chains offer member discounts\n"
        result += "   • Highway rest stops often have multiple options\n"
        result += "   • Consider fuel efficiency when planning stops\n"

        return result

    except Exception as e:
        return f"Error finding gas stations: {e!s}"


@agentc.catalog.tool
def find_accommodations(
    location: str,
    accommodation_type: str = "",
    price_range: str = "",
    amenities: str = ""
) -> str:
    """
    Find hotels and accommodations along a route.

    Args:
        location: City or area to search
        accommodation_type: Type of lodging (hotel, resort, lodge, etc.)
        price_range: Price range preference
        amenities: Desired amenities

    Returns:
        Hotel and accommodation recommendations
    """
    try:
        # Determine region
        location_lower = location.lower()
        region = None

        if any(ca_city in location_lower for ca_city in ["california", "ca", "san francisco", "los angeles", "napa", "monterey"]):
            region = "california"
        elif any(ne_city in location_lower for ne_city in ["new york", "boston", "philadelphia", "ny", "ma"]):
            region = "northeast"
        elif any(mw_city in location_lower for mw_city in ["chicago", "milwaukee", "detroit", "il", "wi"]):
            region = "midwest"

        if not region or region not in POINTS_OF_INTEREST or "accommodations" not in POINTS_OF_INTEREST[region]:
            return f"🏨 Limited accommodation data for '{location}'. Consider checking booking sites like Booking.com, Hotels.com, or Airbnb."

        accommodations = POINTS_OF_INTEREST[region]["accommodations"]

        result = f"🏨 **Accommodations in {location.title()}**\n"
        result += "=" * 50 + "\n"

        for i, hotel in enumerate(accommodations, 1):
            result += f"**{i}. {hotel['name']}**\n"
            result += f"📍 Location: {hotel['location']}\n"
            result += f"🏨 Type: {hotel['category']}\n"
            result += f"💰 Price: {hotel['price_range']}\n"
            result += f"📝 Description: {hotel['description']}\n"

            if hotel.get("amenities"):
                result += f"🎯 Amenities: {', '.join(hotel['amenities'])}\n"

            result += "\n" + "─" * 40 + "\n\n"

        result += "💡 **Booking Tips:**\n"
        result += "   • Book in advance for better rates\n"
        result += "   • Check cancellation policies\n"
        result += "   • Compare prices across multiple sites\n"
        result += "   • Read recent guest reviews\n"
        result += "   • Consider location convenience to your route\n"

        return result

    except Exception as e:
        return f"Error finding accommodations: {e!s}"


@agentc.catalog.tool
def plan_stops_along_route(
    origin: str,
    destination: str,
    stop_types: str = "restaurants,gas,attractions"
) -> str:
    """
    Plan strategic stops along a route including restaurants, gas, and attractions.

    Args:
        origin: Starting point
        destination: Ending point
        stop_types: Comma-separated list of desired stop types

    Returns:
        Comprehensive stop planning recommendations
    """
    try:
        result = f"🛣️ **Route Stop Planning: {origin.title()} to {destination.title()}**\n"
        result += "=" * 70 + "\n"

        stop_list = [s.strip().lower() for s in stop_types.split(",")]

        result += "**Recommended Stop Strategy:**\n\n"

        if "gas" in stop_list:
            result += "⛽ **Fuel Stops:**\n"
            result += "   • Stop every 200-250 miles for fuel\n"
            result += "   • Check gas prices with apps before stopping\n"
            result += "   • Combine with meal/rest breaks for efficiency\n"
            result += "   • Keep tank above 1/4 full in remote areas\n\n"

        if "restaurants" in stop_list or "food" in stop_list:
            result += "🍽️ **Meal Breaks:**\n"
            result += "   • Plan lunch stop halfway through journey\n"
            result += "   • Research local specialties along route\n"
            result += "   • Allow 45-60 minutes for sit-down meals\n"
            result += "   • Consider drive-through for quick options\n\n"

        if "attractions" in stop_list:
            result += "🎯 **Attraction Stops:**\n"
            result += "   • Research must-see landmarks along route\n"
            result += "   • Allow extra time for photo opportunities\n"
            result += "   • Check opening hours and admission fees\n"
            result += "   • Consider seasonal accessibility\n\n"

        if "rest" in stop_list:
            result += "😴 **Rest Breaks:**\n"
            result += "   • Stop every 2 hours for safety\n"
            result += "   • Stretch legs and walk around\n"
            result += "   • Switch drivers if possible\n"
            result += "   • Use rest areas with facilities\n\n"

        result += "💡 **General Stop Tips:**\n"
        result += "   • Plan stops before getting hungry/tired\n"
        result += "   • Use apps to find highly-rated stops ahead\n"
        result += "   • Keep snacks and water in the car\n"
        result += "   • Allow flexibility for unexpected discoveries\n"
        result += "   • Check for road construction delays\n"

        return result

    except Exception as e:
        return f"Error planning route stops: {e!s}"
