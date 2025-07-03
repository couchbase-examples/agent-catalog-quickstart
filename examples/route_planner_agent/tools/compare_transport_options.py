"""
Transport Options Comparison Tool for Agent Catalog

Compares different transportation modes (car, train, bus, flight) with time and cost analysis.
"""


import agentc
import dotenv

dotenv.load_dotenv(override=True)


def _get_transport_data():
    """Get comprehensive transportation mode data."""
    return {
        "car": {
            "cost_per_mile": 0.56,  # IRS standard mileage rate
            "avg_speed": 60,  # mph on highways
            "pros": ["Door-to-door convenience", "Flexibility", "Luggage space", "Scenic route options"],
            "cons": ["Traffic delays", "Parking costs", "Driver fatigue", "Gas prices"],
            "suitable_distances": "50-500 miles"
        },
        "train": {
            "cost_per_mile": 0.15,  # Average Amtrak cost
            "avg_speed": 80,  # Including stops
            "pros": ["Comfortable seating", "No traffic", "City center to city center", "Environmental"],
            "cons": ["Limited routes", "Schedule constraints", "Possible delays", "Advance booking"],
            "suitable_distances": "100-1000 miles"
        },
        "bus": {
            "cost_per_mile": 0.08,  # Budget bus lines
            "avg_speed": 50,  # Including stops
            "pros": ["Very affordable", "Frequent departures", "No parking needed", "WiFi available"],
            "cons": ["Longer travel time", "Limited comfort", "Multiple stops", "Weather dependent"],
            "suitable_distances": "50-800 miles"
        },
        "flight": {
            "cost_per_mile": 0.20,  # Average domestic flight
            "avg_speed": 500,  # Including airport time
            "airport_time": 2,  # Hours for security and boarding
            "pros": ["Fastest for long distances", "Nationwide routes", "Professional service", "Time efficient"],
            "cons": ["Airport hassles", "Weather delays", "Luggage restrictions", "Higher cost"],
            "suitable_distances": "300+ miles"
        }
    }


@agentc.catalog.tool
def compare_transport_options(
    origin: str,
    destination: str,
    distance_miles: float,
    passengers: int = 1,
    budget_priority: str = "balanced"
) -> str:
    """
    Compare transportation options between two locations with detailed analysis.

    Args:
        origin: Starting location
        destination: Ending location
        distance_miles: Distance in miles between locations
        passengers: Number of passengers (default: 1)
        budget_priority: Priority level - "budget", "time", "comfort", or "balanced"

    Returns:
        Detailed comparison of transportation options with recommendations
    """
    try:
        transport_data = _get_transport_data()

        # Calculate options for each transport mode
        options = []

        for mode, data in transport_data.items():
            if mode == "flight" and distance_miles < 200:
                continue  # Skip flight for very short distances
            if mode == "train" and distance_miles < 50:
                continue  # Skip train for very short distances

            # Calculate travel time
            if mode == "flight":
                travel_time = (distance_miles / data["avg_speed"]) + data["airport_time"]
            else:
                travel_time = distance_miles / data["avg_speed"]

            # Calculate cost
            base_cost = distance_miles * data["cost_per_mile"]
            if mode == "car":
                total_cost = base_cost + (20 if distance_miles > 200 else 0)  # Add parking
            elif mode == "flight":
                total_cost = max(base_cost, 150)  # Minimum flight cost
            else:
                total_cost = base_cost

            # Adjust for multiple passengers
            if mode == "car":
                cost_per_person = total_cost / passengers  # Car cost is shared
            else:
                cost_per_person = total_cost * passengers  # Other modes charge per person

            option = {
                "mode": mode.title(),
                "travel_time": travel_time,
                "cost_per_person": cost_per_person,
                "total_cost": cost_per_person * passengers if mode != "car" else total_cost,
                "pros": data["pros"],
                "cons": data["cons"],
                "suitable_distances": data["suitable_distances"]
            }
            options.append(option)

        # Sort options based on priority
        if budget_priority == "budget":
            options.sort(key=lambda x: x["cost_per_person"])
        elif budget_priority == "time":
            options.sort(key=lambda x: x["travel_time"])
        elif budget_priority == "comfort":
            comfort_order = {"flight": 1, "train": 2, "car": 3, "bus": 4}
            options.sort(key=lambda x: comfort_order.get(x["mode"].lower(), 5))
        else:  # balanced
            # Score based on normalized time and cost
            max_time = max(opt["travel_time"] for opt in options)
            max_cost = max(opt["cost_per_person"] for opt in options)
            for opt in options:
                time_score = opt["travel_time"] / max_time
                cost_score = opt["cost_per_person"] / max_cost
                opt["balance_score"] = (time_score + cost_score) / 2
            options.sort(key=lambda x: x["balance_score"])

        # Format results
        result = f"🚗 **Transportation Options: {origin} to {destination}**\n"
        result += f"📏 Distance: {distance_miles:.0f} miles | 👥 Passengers: {passengers}\n"
        result += f"🎯 Priority: {budget_priority.title()}\n"
        result += "=" * 60 + "\n\n"

        for i, option in enumerate(options, 1):
            result += f"**{i}. {option['mode']} - "
            if budget_priority == "budget":
                result += "Most Affordable" if i == 1 else f"${option['cost_per_person']:.0f}/person"
            elif budget_priority == "time":
                result += "Fastest" if i == 1 else f"{option['travel_time']:.1f} hours"
            else:
                result += "Best Choice" if i == 1 else "Alternative"
            result += "**\n"

            result += f"⏱️ Travel Time: {option['travel_time']:.1f} hours\n"
            result += f"💰 Cost: ${option['cost_per_person']:.0f} per person"
            if passengers > 1:
                result += f" (${option['total_cost']:.0f} total)"
            result += "\n"

            result += f"✅ **Pros**: {', '.join(option['pros'][:2])}\n"
            result += f"⚠️ **Cons**: {', '.join(option['cons'][:2])}\n"
            result += f"📋 **Best for**: {option['suitable_distances']}\n"
            result += "─" * 40 + "\n\n"

        # Add recommendations
        result += "💡 **Recommendations:**\n"
        best_option = options[0]
        result += f"• **Primary choice**: {best_option['mode']} for optimal {budget_priority} balance\n"

        if len(options) > 1:
            alt_option = options[1]
            result += f"• **Alternative**: {alt_option['mode']} if {budget_priority} is less important\n"

        if distance_miles > 500:
            result += "• **Long distance tip**: Consider flying to save time\n"
        elif distance_miles < 100:
            result += "• **Short distance tip**: Car offers most flexibility\n"

        if passengers > 3:
            result += "• **Group travel**: Car becomes more economical with more passengers\n"

        return result

    except Exception as e:
        return f"Error comparing transport options: {e!s}. Please check your inputs and try again."


@agentc.catalog.tool
def estimate_travel_costs(
    distance_miles: float,
    transport_mode: str,
    passengers: int = 1,
    trip_duration_days: int = 1
) -> str:
    """
    Estimate detailed travel costs for a specific transportation mode.

    Args:
        distance_miles: Distance in miles
        transport_mode: Transportation mode (car, train, bus, flight)
        passengers: Number of passengers
        trip_duration_days: Duration of trip in days

    Returns:
        Detailed cost breakdown for the specified transport mode
    """
    try:
        transport_data = _get_transport_data()
        mode = transport_mode.lower()

        if mode not in transport_data:
            return f"Unknown transport mode: {transport_mode}. Available options: car, train, bus, flight"

        data = transport_data[mode]

        # Base transportation cost
        base_cost = distance_miles * data["cost_per_mile"]

        # Mode-specific adjustments
        if mode == "car":
            # Add gas, tolls, parking
            gas_cost = distance_miles * 0.12  # Average gas cost per mile
            toll_cost = 20 if distance_miles > 200 else 5
            parking_cost = 25 * trip_duration_days if trip_duration_days > 1 else 10
            total_transport = gas_cost + toll_cost + parking_cost
            cost_per_person = total_transport / passengers
        elif mode == "flight":
            total_transport = max(base_cost, 150)  # Minimum flight cost
            baggage_cost = 30 * passengers if distance_miles > 500 else 0
            total_transport += baggage_cost
            cost_per_person = total_transport
        else:  # train or bus
            total_transport = base_cost
            cost_per_person = total_transport

        # Calculate total for all passengers
        if mode == "car":
            total_cost = total_transport
        else:
            total_cost = cost_per_person * passengers

        # Additional costs for multi-day trips
        if trip_duration_days > 1:
            accommodation_cost = 120 * (trip_duration_days - 1) * passengers  # Per person per night
            meal_cost = 50 * trip_duration_days * passengers  # Per person per day
        else:
            accommodation_cost = 0
            meal_cost = 30 * passengers  # Just meals during travel

        total_trip_cost = total_cost + accommodation_cost + meal_cost

        # Format results
        result = f"💰 **Cost Estimate: {transport_mode.title()} Travel**\n"
        result += f"📏 Distance: {distance_miles:.0f} miles | 👥 Passengers: {passengers} | 📅 Duration: {trip_duration_days} days\n"
        result += "=" * 50 + "\n\n"

        result += "**Transportation Costs:**\n"
        if mode == "car":
            result += f"⛽ Fuel: ${distance_miles * 0.12:.0f}\n"
            result += f"🛣️ Tolls: ${20 if distance_miles > 200 else 5}\n"
            result += f"🅿️ Parking: ${25 * trip_duration_days if trip_duration_days > 1 else 10}\n"
            result += f"**Subtotal**: ${total_transport:.0f} (shared)\n"
        else:
            result += f"🎫 {transport_mode.title()} tickets: ${cost_per_person:.0f} × {passengers} = ${total_cost:.0f}\n"
            if mode == "flight" and distance_miles > 500:
                result += f"🧳 Baggage fees: ${30 * passengers}\n"

        if trip_duration_days > 1:
            result += "\n**Additional Costs:**\n"
            result += f"🏨 Accommodation: ${accommodation_cost:.0f} ({trip_duration_days-1} nights)\n"
            result += f"🍽️ Meals: ${meal_cost:.0f}\n"
        else:
            result += f"\n**Travel Day Meals**: ${meal_cost:.0f}\n"

        result += f"\n**Total Trip Cost**: ${total_trip_cost:.0f}\n"
        result += f"**Cost per person**: ${total_trip_cost/passengers:.0f}\n\n"

        # Add money-saving tips
        result += "💡 **Money-Saving Tips:**\n"
        if mode == "car":
            result += "• Share driving and costs with fellow travelers\n"
            result += "• Use gas price apps to find cheapest fuel stops\n"
            result += "• Consider free camping or budget motels\n"
        elif mode == "flight":
            result += "• Book flights 6-8 weeks in advance\n"
            result += "• Consider carry-on only to avoid baggage fees\n"
            result += "• Use flight comparison websites\n"
        elif mode == "train":
            result += "• Book early for better prices\n"
            result += "• Consider coach seats for shorter trips\n"
            result += "• Look for rail passes for multi-city travel\n"
        else:  # bus
            result += "• Book online for discounts\n"
            result += "• Travel on weekdays for lower prices\n"
            result += "• Bring your own snacks and entertainment\n"

        return result

    except Exception as e:
        return f"Error estimating travel costs: {e!s}. Please check your inputs and try again."
