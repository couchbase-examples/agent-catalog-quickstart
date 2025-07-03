"""
Distance and Time Calculation Tool for Agent Catalog

Calculates distances, travel times, and costs between locations.
"""

import math

import agentc

# Major city coordinates for distance calculations
CITY_COORDINATES = {
    "new york": (40.7128, -74.0060),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
    "san francisco": (37.7749, -122.4194),
    "washington dc": (38.9072, -77.0369),
    "boston": (42.3601, -71.0589),
    "philadelphia": (39.9526, -75.1652),
    "seattle": (47.6062, -122.3321),
    "denver": (39.7392, -104.9903),
    "miami": (25.7617, -80.1918),
    "atlanta": (33.7490, -84.3880),
    "dallas": (32.7767, -96.7970),
    "houston": (29.7604, -95.3698),
    "phoenix": (33.4484, -112.0740),
    "las vegas": (36.1699, -115.1398),
    "portland": (45.5152, -122.6784),
    "milwaukee": (43.0389, -87.9065),
    "baltimore": (39.2904, -76.6122),
    "detroit": (42.3314, -83.0458),
    "minneapolis": (44.9778, -93.2650),
    "cleveland": (41.4993, -81.6944),
    "orlando": (28.5383, -81.3792),
    "tampa": (27.9506, -82.4572),
    "nashville": (36.1627, -86.7816),
    "charlotte": (35.2271, -80.8431),
    "salt lake city": (40.7608, -111.8910),
    "kansas city": (39.0997, -94.5786),
    "oklahoma city": (35.4676, -97.5164),
    "santa fe": (35.6870, -105.9378),
    "albuquerque": (35.0844, -106.6504),
    "tucson": (32.2226, -110.9747),
    "sacramento": (38.5816, -121.4944),
    "san diego": (32.7157, -117.1611),
    "monterey": (36.6002, -121.8947),
    "santa barbara": (34.4208, -119.6982),
    "napa": (38.2975, -122.2869),
    "aspen": (39.1911, -106.8175),
    "jackson": (43.4799, -110.7624),
    "bozeman": (45.6770, -111.0429),
    "missoula": (46.8721, -113.9940)
}

# Transportation speed estimates (mph)
TRANSPORT_SPEEDS = {
    "car": 55,          # Average including city/highway
    "highway": 70,      # Highway driving
    "city": 25,         # City driving
    "train": 60,        # Regular trains
    "high_speed_rail": 150,  # High-speed rail
    "bus": 45,          # Bus travel
    "walking": 3,       # Walking pace
    "cycling": 12,      # Bicycle
    "flight": 500       # Commercial flight
}

# Cost estimates per mile
TRANSPORT_COSTS = {
    "car": 0.65,        # IRS mileage rate (gas, wear, insurance)
    "flight": 0.15,     # Per mile flight cost
    "train": 0.25,      # Amtrak average
    "bus": 0.10,        # Bus travel
    "gas_only": 0.15    # Gas cost only
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth.

    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point

    Returns:
        Distance in miles
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in miles
    r = 3956
    return c * r


def get_city_coordinates(city_name: str) -> tuple:
    """Get coordinates for a city name."""
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
    Calculate distance, travel time, and estimated costs between two locations.

    Args:
        origin: Starting city or location
        destination: Ending city or location
        transport_mode: Mode of transportation (car, train, flight, bus, walking, cycling)

    Returns:
        Formatted string with distance, time, and cost calculations
    """
    try:
        # Get coordinates for both cities
        origin_coords = get_city_coordinates(origin)
        destination_coords = get_city_coordinates(destination)

        if not origin_coords:
            return f"❌ Could not find coordinates for '{origin}'. Please check the city name."

        if not destination_coords:
            return f"❌ Could not find coordinates for '{destination}'. Please check the city name."

        # Calculate distance
        distance = haversine_distance(
            origin_coords[0], origin_coords[1],
            destination_coords[0], destination_coords[1]
        )

        # Get speed for transport mode
        speed = TRANSPORT_SPEEDS.get(transport_mode.lower(), TRANSPORT_SPEEDS["car"])

        # Calculate travel time
        travel_time_hours = distance / speed
        hours = int(travel_time_hours)
        minutes = int((travel_time_hours - hours) * 60)

        # Calculate costs
        cost_per_mile = TRANSPORT_COSTS.get(transport_mode.lower(), TRANSPORT_COSTS["car"])
        estimated_cost = distance * cost_per_mile

        # Format result
        result = f"🗺️ **Route Calculation: {origin.title()} to {destination.title()}**\n"
        result += "=" * 60 + "\n"
        result += f"📏 **Distance:** {distance:.1f} miles ({distance * 1.6:.1f} km)\n"
        result += f"🚗 **Transport Mode:** {transport_mode.title()}\n"
        result += f"⏱️ **Estimated Travel Time:** {hours}h {minutes}m\n"
        result += f"💰 **Estimated Cost:** ${estimated_cost:.2f}\n\n"

        # Add transport-specific notes
        if transport_mode.lower() == "car":
            result += "🚗 **Driving Notes:**\n"
            result += f"   • Gas cost only: ~${distance * TRANSPORT_COSTS['gas_only']:.2f}\n"
            result += "   • Add time for rest stops on long trips\n"
            result += "   • Consider traffic in major cities\n"
            result += "   • Check for tolls on your route\n"

        elif transport_mode.lower() == "flight":
            result += "✈️ **Flight Notes:**\n"
            result += "   • Add 2-3 hours for airport procedures\n"
            result += "   • Consider ground transportation to/from airports\n"
            result += "   • Check baggage fees\n"

        elif transport_mode.lower() == "train":
            result += "🚂 **Train Notes:**\n"
            result += "   • Arrive 30 minutes before departure\n"
            result += "   • Consider booking in advance for better prices\n"
            result += "   • No driving/parking hassles\n"

        elif transport_mode.lower() == "bus":
            result += "🚌 **Bus Notes:**\n"
            result += "   • Budget-friendly option\n"
            result += "   • May have multiple stops\n"
            result += "   • Check schedule reliability\n"

        return result

    except Exception as e:
        return f"Error calculating distance: {e!s}"


@agentc.catalog.tool
def compare_transport_options(origin: str, destination: str) -> str:
    """
    Compare different transportation options for a route.

    Args:
        origin: Starting city or location
        destination: Ending city or location

    Returns:
        Comparison table of transportation options with time and cost
    """
    try:
        # Get coordinates
        origin_coords = get_city_coordinates(origin)
        destination_coords = get_city_coordinates(destination)

        if not origin_coords or not destination_coords:
            return "❌ Could not find coordinates for one or both cities."

        # Calculate base distance
        distance = haversine_distance(
            origin_coords[0], origin_coords[1],
            destination_coords[0], destination_coords[1]
        )

        # Compare different modes
        modes_to_compare = ["car", "flight", "train", "bus"]

        result = f"🚗✈️🚂🚌 **Transportation Comparison: {origin.title()} to {destination.title()}**\n"
        result += "=" * 70 + "\n"
        result += f"📏 Distance: {distance:.1f} miles\n\n"

        comparisons = []

        for mode in modes_to_compare:
            speed = TRANSPORT_SPEEDS.get(mode, TRANSPORT_SPEEDS["car"])
            cost_per_mile = TRANSPORT_COSTS.get(mode, TRANSPORT_COSTS["car"])

            # Calculate time and cost
            travel_time_hours = distance / speed
            hours = int(travel_time_hours)
            minutes = int((travel_time_hours - hours) * 60)
            cost = distance * cost_per_mile

            # Add airport/station time for relevant modes
            total_time_str = f"{hours}h {minutes}m"
            if mode == "flight":
                total_hours = hours + 3  # Add airport time
                total_time_str = f"{hours}h {minutes}m (+3h airports) = {total_hours}h {minutes}m total"
            elif mode == "train":
                total_time_str = f"{hours}h {minutes}m (+30min station)"

            comparisons.append({
                "mode": mode,
                "time": total_time_str,
                "cost": cost,
                "hours_only": travel_time_hours
            })

        # Sort by time for short distances, cost for long distances
        if distance < 300:
            comparisons.sort(key=lambda x: x["hours_only"])
            result += "**Sorted by travel time (best for shorter distances):**\n\n"
        else:
            comparisons.sort(key=lambda x: x["cost"])
            result += "**Sorted by cost (good for longer distances):**\n\n"

        for i, comp in enumerate(comparisons, 1):
            mode_icon = {"car": "🚗", "flight": "✈️", "train": "🚂", "bus": "🚌"}
            result += f"**{i}. {mode_icon[comp['mode']]} {comp['mode'].title()}**\n"
            result += f"   ⏱️ Time: {comp['time']}\n"
            result += f"   💰 Cost: ${comp['cost']:.2f}\n\n"

        # Add recommendations
        result += "💡 **Recommendations:**\n"
        if distance < 100:
            result += "   • Car or train recommended for short distances\n"
        elif distance < 500:
            result += "   • Consider train for comfort, flight for speed\n"
        else:
            result += "   • Flight usually best for long distances\n"

        result += "   • Check real-time prices and schedules\n"
        result += "   • Consider total door-to-door time including connections\n"

        return result

    except Exception as e:
        return f"Error comparing transport options: {e!s}"


@agentc.catalog.tool
def estimate_travel_costs(
    distance_miles: float,
    transport_mode: str,
    passengers: int = 1,
    include_extras: bool = True
) -> str:
    """
    Estimate detailed travel costs for a trip.

    Args:
        distance_miles: Distance of the trip in miles
        transport_mode: Mode of transportation
        passengers: Number of passengers/travelers
        include_extras: Whether to include additional costs like meals, parking

    Returns:
        Detailed cost breakdown
    """
    try:
        cost_per_mile = TRANSPORT_COSTS.get(transport_mode.lower(), TRANSPORT_COSTS["car"])
        base_cost = distance_miles * cost_per_mile * passengers

        result = "💰 **Travel Cost Estimate**\n"
        result += "=" * 40 + "\n"
        result += f"📏 Distance: {distance_miles:.1f} miles\n"
        result += f"🚗 Transport: {transport_mode.title()}\n"
        result += f"👥 Passengers: {passengers}\n\n"

        result += f"**Base Travel Cost:** ${base_cost:.2f}\n\n"

        if include_extras:
            result += "**Additional Estimated Costs:**\n"

            if transport_mode.lower() == "car":
                parking = min(distance_miles * 0.05, 50)  # Parking estimate
                tolls = distance_miles * 0.10  # Toll estimate
                result += f"   🅿️ Parking: ~${parking:.2f}\n"
                result += f"   🛣️ Tolls: ~${tolls:.2f}\n"

            elif transport_mode.lower() == "flight":
                baggage = 30 * passengers  # Baggage fees
                airport_transport = 50 * passengers  # Airport transportation
                result += f"   🧳 Baggage fees: ~${baggage:.2f}\n"
                result += f"   🚖 Airport transport: ~${airport_transport:.2f}\n"

            # Meals estimate for long trips
            if distance_miles > 200:
                meals = (distance_miles / 200) * 30 * passengers
                result += f"   🍽️ Meals: ~${meals:.2f}\n"

            # Lodging for very long trips
            if distance_miles > 600:
                nights = int(distance_miles / 600)
                lodging = nights * 100
                result += f"   🏨 Lodging ({nights} nights): ~${lodging:.2f}\n"

        return result

    except Exception as e:
        return f"Error estimating costs: {e!s}"
