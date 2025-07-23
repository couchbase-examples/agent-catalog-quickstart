"""
Distance Calculation Tool - Simplified for Tutorial

This tool demonstrates:
- Using AgentC for tool registration
- Simple distance and time calculations
- Basic cost estimation for different transport modes
"""

import math
import logging
import agentc

# Setup logging
logger = logging.getLogger(__name__)

# Common US cities with coordinates (expanded set)
CITY_COORDINATES = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "san francisco": (37.7749, -122.4194),
    "boston": (42.3601, -71.0589),
    "washington dc": (38.9072, -77.0369),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "las vegas": (36.1699, -115.1398),
    "phoenix": (33.4484, -112.0740),
    "dallas": (32.7767, -96.7970),
    "philadelphia": (39.9526, -75.1652),
    "detroit": (42.3314, -83.0458),
    "orlando": (28.5383, -81.3792),
    "portland": (45.5152, -122.6784),
    "austin": (30.2672, -97.7431),
    "nashville": (36.1627, -86.7816),
    "salt lake city": (40.7608, -111.8910),
    # Tourist destinations and mountain cities
    "aspen": (39.1911, -106.8175),
    "vail": (39.6403, -106.3742),
    "jackson": (43.4799, -110.7624),  # Jackson Hole
    "park city": (40.6461, -111.4980),
    "tahoe": (39.0968, -120.0324),  # Lake Tahoe
    "key west": (24.5551, -81.7800),
    "santa fe": (35.6870, -105.9378),
    "charleston": (32.7765, -79.9311),
    "savannah": (32.0835, -81.0998),
    "napa": (38.2975, -122.2869),
    "monterey": (36.6002, -121.8947),
    "santa barbara": (34.4208, -119.6982),
    "big sur": (36.2704, -121.8081),
    "sedona": (34.8697, -111.7610),
    "yellowstone": (44.4280, -110.5885),
    "grand canyon": (36.0544, -112.1401),
    "yosemite": (37.8651, -119.5383),
    "zion": (37.2982, -113.0263),
    "zion national park": (37.2982, -113.0263),  # Alias for zion
    "bryce": (37.5930, -112.1871),
    "bryce canyon": (37.5930, -112.1871),  # Alias for bryce
    "arches": (38.7331, -109.5925),  # Arches National Park
    "arches national park": (38.7331, -109.5925),  # Alias for arches
    "moab": (38.5733, -109.5498),  # Near Arches/Canyonlands
    "mammoth": (37.6489, -118.9720),  # Mammoth Lakes
    "steamboat": (40.4850, -106.8317),  # Steamboat Springs
    "telluride": (37.9375, -107.8123),
    "crested butte": (38.8697, -106.9878),
    "sun valley": (43.6966, -114.3558),
    "jackson hole": (43.4799, -110.7624),
    "whistler": (50.1163, -122.9574),  # Popular ski destination
    "banff": (51.1784, -115.5708),  # Popular mountain destination
}

# Transportation modes with average speeds (mph)
TRANSPORT_MODES = {
    "car": {"speed": 60, "cost_per_mile": 0.65},
    "train": {"speed": 70, "cost_per_mile": 0.25},
    "bus": {"speed": 50, "cost_per_mile": 0.10},
    "flight": {"speed": 500, "cost_per_mile": 0.15},
}


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in miles."""
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in miles
    return c * 3956


def find_city_coordinates(city_name: str) -> tuple:
    """Find coordinates for a city name with fuzzy matching."""
    city_key = city_name.lower().strip()

    # Try exact match first
    if city_key in CITY_COORDINATES:
        return CITY_COORDINATES[city_key]

    # Try partial match
    for city, coords in CITY_COORDINATES.items():
        if city_key in city or city in city_key:
            return coords

    return None


@agentc.catalog.tool
def calculate_distance(origin: str, destination: str, transport_mode: str = "car") -> str:
    """
    Calculate distance, travel time, and cost between two cities.

    Args:
        origin: Starting city name
        destination: Destination city name
        transport_mode: Transportation mode (car, train, bus, flight)

    Returns:
        Formatted string with distance, time, and cost calculations
    """
    try:
        logger.info(f"Calculating distance from {origin} to {destination} by {transport_mode}")

        # Find coordinates for both cities
        origin_coords = find_city_coordinates(origin)
        destination_coords = find_city_coordinates(destination)

        if not origin_coords:
            available_cities = ", ".join(sorted(CITY_COORDINATES.keys())[:10])  # Show first 10
            return f"❌ City '{origin}' not found. Sample available cities: {available_cities}..."

        if not destination_coords:
            available_cities = ", ".join(sorted(CITY_COORDINATES.keys())[:10])  # Show first 10
            return (
                f"❌ City '{destination}' not found. Sample available cities: {available_cities}..."
            )

        # Calculate distance
        distance = calculate_haversine_distance(
            origin_coords[0], origin_coords[1], destination_coords[0], destination_coords[1]
        )

        # Get transport mode details
        transport_mode_lower = transport_mode.lower()
        if transport_mode_lower not in TRANSPORT_MODES:
            return f"❌ Transport mode '{transport_mode}' not supported. Available modes: {', '.join(TRANSPORT_MODES.keys())}"

        mode_info = TRANSPORT_MODES[transport_mode_lower]
        speed = mode_info["speed"]
        cost_per_mile = mode_info["cost_per_mile"]

        # Calculate travel time
        travel_time_hours = distance / speed
        hours = int(travel_time_hours)
        minutes = int((travel_time_hours - hours) * 60)

        # Calculate cost
        total_cost = distance * cost_per_mile

        # Format results
        result = f"🧮 **Distance Calculator**\n"
        result += "=" * 40 + "\n"
        result += f"**Route:** {origin.title()} → {destination.title()}\n"
        result += f"**Distance:** {distance:.1f} miles ({distance * 1.6:.1f} km)\n"
        result += f"**Transport:** {transport_mode.title()}\n"
        result += f"**Travel Time:** {hours}h {minutes}m\n"
        result += f"**Estimated Cost:** ${total_cost:.2f}\n\n"

        # Add mode-specific notes
        if transport_mode_lower == "car":
            result += "🚗 **Note:** Includes gas, wear, and insurance costs. Add time for rest stops on long trips."
        elif transport_mode_lower == "flight":
            result += "✈️ **Note:** Flight time only. Add 2-3 hours for airport procedures."
        elif transport_mode_lower == "train":
            result += "🚂 **Note:** Comfortable travel without driving stress."
        elif transport_mode_lower == "bus":
            result += "🚌 **Note:** Most economical option. May include stops."

        logger.info(f"Distance calculation completed: {distance:.1f} miles")
        return result

    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return f"❌ Error calculating distance: {str(e)}"
