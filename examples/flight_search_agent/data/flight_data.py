#!/usr/bin/env python3
"""
Comprehensive Flight Data for Vector Search Ingestion

Enhanced flight data covering policies, routes, schedules, and booking information.
This data will be used to populate the vector database for semantic search in the flight search agent.
"""

# Flight Policy Documents
FLIGHT_POLICIES = [
    {
        "policy_id": "CANCEL_001",
        "title": "Flight Cancellation Policy",
        "category": "cancellation",
        "content": """Flight cancellation policies vary by ticket type and timing. Economy tickets cancelled within 24 hours of booking receive full refunds. Cancellations made 7+ days before departure incur a $200 fee for domestic flights and $400 for international flights. Cancellations within 7 days of departure forfeit 50% of ticket value. Premium and business class tickets offer more flexible cancellation terms with reduced fees. No-show passengers forfeit entire ticket value unless extenuating circumstances apply.""",
    },
    {
        "policy_id": "BAGGAGE_001",
        "title": "Carry-on Baggage Restrictions",
        "category": "baggage",
        "content": """Carry-on bags must not exceed 22x14x9 inches and 15 pounds. Personal items like purses, laptops, and small backpacks are permitted in addition to carry-on bags. Liquids must be in containers of 3.4oz or less and fit in a quart-sized clear bag. Prohibited items include sharp objects, flammable materials, and large electronics. Medical equipment and baby formula are exempt from liquid restrictions with proper documentation.""",
    },
    {
        "policy_id": "BAGGAGE_002",
        "title": "Checked Baggage Fees and Limits",
        "category": "baggage",
        "content": """First checked bag costs $35 domestic, $60 international. Second bag costs $45 domestic, $100 international. Maximum weight is 50 pounds per bag. Overweight bags (51-70 lbs) incur $100 fee. Oversized bags exceeding 62 linear inches cost additional $200. Premium passengers receive one free checked bag. Military personnel with orders travel with free checked bags.""",
    },
    {
        "policy_id": "CHECKIN_001",
        "title": "Online and Mobile Check-in",
        "category": "check-in",
        "content": """Online check-in opens 24 hours before departure and closes 1 hour before domestic flights, 2 hours before international flights. Mobile boarding passes are accepted at all airports. Passengers with checked bags must visit counter or kiosk for bag drop. Seat selection is available during check-in for a fee on economy tickets. Check-in reminders are sent via email and SMS 24 hours prior to departure.""",
    },
    {
        "policy_id": "WEATHER_001",
        "title": "Weather Delay and Cancellation Policy",
        "category": "weather",
        "content": """Weather-related delays and cancellations are considered extraordinary circumstances beyond airline control. Passengers are rebooked on next available flight at no additional cost. Hotel accommodations not provided for weather delays, but airport vouchers may be issued for extended delays. Travel insurance recommended to cover weather-related expenses. Real-time flight status updates provided via app, email, and SMS.""",
    },
]

# Flight Route and Schedule Data
FLIGHT_ROUTES = [
    {
        "route_id": "NYC_LAX_001",
        "departure_airport": "JFK",
        "departure_city": "New York",
        "arrival_airport": "LAX",
        "arrival_city": "Los Angeles",
        "airline": "American Airlines",
        "flight_number": "AA123",
        "aircraft_type": "Boeing 777-200",
        "duration_hours": 6.5,
        "distance_miles": 2475,
        "departure_times": ["06:00", "12:30", "18:45"],
        "frequency": "Daily",
        "price_economy": 299,
        "price_business": 899,
        "price_first": 1499,
    },
    {
        "route_id": "LAX_NYC_001",
        "departure_airport": "LAX",
        "departure_city": "Los Angeles",
        "arrival_airport": "JFK",
        "arrival_city": "New York",
        "airline": "Delta Airlines",
        "flight_number": "DL456",
        "aircraft_type": "Airbus A330-300",
        "duration_hours": 5.5,
        "distance_miles": 2475,
        "departure_times": ["08:15", "14:20", "20:10"],
        "frequency": "Daily",
        "price_economy": 319,
        "price_business": 949,
        "price_first": 1599,
    },
    {
        "route_id": "SFO_CHI_001",
        "departure_airport": "SFO",
        "departure_city": "San Francisco",
        "arrival_airport": "ORD",
        "arrival_city": "Chicago",
        "airline": "United Airlines",
        "flight_number": "UA789",
        "aircraft_type": "Boeing 737-800",
        "duration_hours": 4.5,
        "distance_miles": 1846,
        "departure_times": ["07:30", "13:45", "19:00"],
        "frequency": "Daily",
        "price_economy": 249,
        "price_business": 649,
        "price_first": 1049,
    },
    {
        "route_id": "MIA_ATL_001",
        "departure_airport": "MIA",
        "departure_city": "Miami",
        "arrival_airport": "ATL",
        "arrival_city": "Atlanta",
        "airline": "Delta Airlines",
        "flight_number": "DL234",
        "aircraft_type": "Boeing 757-200",
        "duration_hours": 2.5,
        "distance_miles": 594,
        "departure_times": ["09:00", "15:30", "21:15"],
        "frequency": "Daily",
        "price_economy": 159,
        "price_business": 399,
        "price_first": 699,
    },
    {
        "route_id": "SEA_DEN_001",
        "departure_airport": "SEA",
        "departure_city": "Seattle",
        "arrival_airport": "DEN",
        "arrival_city": "Denver",
        "airline": "Southwest Airlines",
        "flight_number": "WN567",
        "aircraft_type": "Boeing 737-700",
        "duration_hours": 2.75,
        "distance_miles": 1024,
        "departure_times": ["10:45", "16:20"],
        "frequency": "Daily",
        "price_economy": 179,
        "price_business": 299,
        "price_first": 499,
    },
]

# Airline Information
AIRLINES = [
    {
        "airline_code": "AA",
        "airline_name": "American Airlines",
        "hub_airports": ["DFW", "CLT", "PHX", "MIA"],
        "alliance": "Oneworld",
        "baggage_policy": "First bag $35 domestic, $60 international",
        "cancellation_policy": "24-hour free cancellation, $200 domestic/$400 international fee after",
        "fleet_size": 950,
        "destinations": 350,
    },
    {
        "airline_code": "DL",
        "airline_name": "Delta Airlines",
        "hub_airports": ["ATL", "DTW", "MSP", "SEA"],
        "alliance": "SkyTeam",
        "baggage_policy": "First bag $35 domestic, $60 international",
        "cancellation_policy": "24-hour free cancellation, $200 domestic/$400 international fee after",
        "fleet_size": 900,
        "destinations": 325,
    },
    {
        "airline_code": "UA",
        "airline_name": "United Airlines",
        "hub_airports": ["ORD", "IAH", "EWR", "SFO"],
        "alliance": "Star Alliance",
        "baggage_policy": "First bag $35 domestic, $60 international",
        "cancellation_policy": "24-hour free cancellation, $200 domestic/$400 international fee after",
        "fleet_size": 850,
        "destinations": 340,
    },
    {
        "airline_code": "WN",
        "airline_name": "Southwest Airlines",
        "hub_airports": ["DAL", "BWI", "MDW", "PHX"],
        "alliance": "None",
        "baggage_policy": "First two bags free",
        "cancellation_policy": "No cancellation fees, credit valid for 12 months",
        "fleet_size": 800,
        "destinations": 120,
    },
]

# Airport Information
AIRPORTS = [
    {
        "airport_code": "JFK",
        "airport_name": "John F. Kennedy International Airport",
        "city": "New York",
        "state": "NY",
        "country": "USA",
        "timezone": "EST",
        "terminals": 6,
        "airlines": ["AA", "DL", "UA", "B6", "AF"],
        "amenities": ["WiFi", "Restaurants", "Shops", "Lounges", "Hotels"],
    },
    {
        "airport_code": "LAX",
        "airport_name": "Los Angeles International Airport",
        "city": "Los Angeles",
        "state": "CA",
        "country": "USA",
        "timezone": "PST",
        "terminals": 9,
        "airlines": ["AA", "DL", "UA", "WN", "AS"],
        "amenities": ["WiFi", "Restaurants", "Shops", "Lounges", "Hotels"],
    },
    {
        "airport_code": "ORD",
        "airport_name": "O'Hare International Airport",
        "city": "Chicago",
        "state": "IL",
        "country": "USA",
        "timezone": "CST",
        "terminals": 4,
        "airlines": ["UA", "AA", "DL", "WN"],
        "amenities": ["WiFi", "Restaurants", "Shops", "Lounges", "Hotels"],
    },
]

# Booking and Travel Information
BOOKING_CLASSES = [
    {
        "class_code": "Y",
        "class_name": "Economy",
        "description": "Standard seating with basic amenities",
        "baggage_allowance": "1 carry-on, 1 personal item",
        "meal_service": "Purchase required on most flights",
        "seat_selection": "Fee required for preferred seats",
        "changes_allowed": "Yes, with fees",
        "upgrades_available": "Yes, subject to availability",
    },
    {
        "class_code": "W",
        "class_name": "Premium Economy",
        "description": "Enhanced economy with extra legroom and priority boarding",
        "baggage_allowance": "1 carry-on, 1 personal item, priority handling",
        "meal_service": "Complimentary snacks and beverages",
        "seat_selection": "Included in fare",
        "changes_allowed": "Yes, reduced fees",
        "upgrades_available": "Yes, to business class",
    },
    {
        "class_code": "J",
        "class_name": "Business Class",
        "description": "Premium cabin with lie-flat seats and enhanced service",
        "baggage_allowance": "2 carry-on, 2 checked bags free",
        "meal_service": "Multi-course meals with wine service",
        "seat_selection": "Included, priority seating",
        "changes_allowed": "Yes, minimal or no fees",
        "upgrades_available": "Yes, to first class",
    },
    {
        "class_code": "F",
        "class_name": "First Class",
        "description": "Luxury cabin with private suites and concierge service",
        "baggage_allowance": "3 carry-on, 3 checked bags free",
        "meal_service": "Gourmet dining with premium beverages",
        "seat_selection": "Included, best available seats",
        "changes_allowed": "Yes, no fees",
        "upgrades_available": "None, highest class",
    },
]


def get_all_flight_data():
    """Return all flight-related data for vector search ingestion."""
    all_data = []

    # Add policies
    for policy in FLIGHT_POLICIES:
        all_data.append(
            {
                "type": "policy",
                "id": policy["policy_id"],
                "title": policy["title"],
                "category": policy["category"],
                "content": policy["content"],
            }
        )

    # Add route information
    for route in FLIGHT_ROUTES:
        content = f"Flight {route['flight_number']} operated by {route['airline']} from {route['departure_city']} ({route['departure_airport']}) to {route['arrival_city']} ({route['arrival_airport']}). Duration: {route['duration_hours']} hours, Distance: {route['distance_miles']} miles. Aircraft: {route['aircraft_type']}. Departure times: {', '.join(route['departure_times'])}. Economy: ${route['price_economy']}, Business: ${route['price_business']}, First: ${route['price_first']}."
        all_data.append(
            {
                "type": "route",
                "id": route["route_id"],
                "title": f"{route['departure_city']} to {route['arrival_city']} - {route['airline']}",
                "category": "flight_schedule",
                "content": content,
            }
        )

    # Add airline information
    for airline in AIRLINES:
        content = f"{airline['airline_name']} ({airline['airline_code']}) is a {airline.get('alliance', 'independent')} member airline with {airline['fleet_size']} aircraft serving {airline['destinations']} destinations. Hub airports: {', '.join(airline['hub_airports'])}. Baggage policy: {airline['baggage_policy']}. Cancellation policy: {airline['cancellation_policy']}."
        all_data.append(
            {
                "type": "airline",
                "id": airline["airline_code"],
                "title": airline["airline_name"],
                "category": "airline_info",
                "content": content,
            }
        )

    # Add airport information
    for airport in AIRPORTS:
        content = f"{airport['airport_name']} ({airport['airport_code']}) located in {airport['city']}, {airport['state']}, {airport['country']}. Timezone: {airport['timezone']}. Has {airport['terminals']} terminals. Airlines: {', '.join(airport['airlines'])}. Amenities: {', '.join(airport['amenities'])}."
        all_data.append(
            {
                "type": "airport",
                "id": airport["airport_code"],
                "title": airport["airport_name"],
                "category": "airport_info",
                "content": content,
            }
        )

    # Add booking class information
    for booking_class in BOOKING_CLASSES:
        content = f"{booking_class['class_name']} ({booking_class['class_code']}): {booking_class['description']}. Baggage: {booking_class['baggage_allowance']}. Meals: {booking_class['meal_service']}. Seat selection: {booking_class['seat_selection']}. Changes: {booking_class['changes_allowed']}. Upgrades: {booking_class['upgrades_available']}."
        all_data.append(
            {
                "type": "booking_class",
                "id": booking_class["class_code"],
                "title": f"{booking_class['class_name']} Class",
                "category": "booking_info",
                "content": content,
            }
        )

    return all_data


def get_data_by_category(category):
    """Return flight data filtered by category."""
    all_data = get_all_flight_data()
    return [item for item in all_data if item["category"] == category]


def get_data_by_type(data_type):
    """Return flight data filtered by type."""
    all_data = get_all_flight_data()
    return [item for item in all_data if item["type"] == data_type]


def search_routes(departure=None, arrival=None):
    """Search for flight routes by departure and/or arrival airport."""
    routes = []
    for route in FLIGHT_ROUTES:
        if departure and route["departure_airport"].upper() != departure.upper():
            continue
        if arrival and route["arrival_airport"].upper() != arrival.upper():
            continue
        routes.append(route)
    return routes


def get_airline_info(airline_code):
    """Get airline information by code."""
    for airline in AIRLINES:
        if airline["airline_code"].upper() == airline_code.upper():
            return airline
    return None


def get_airport_info(airport_code):
    """Get airport information by code."""
    for airport in AIRPORTS:
        if airport["airport_code"].upper() == airport_code.upper():
            return airport
    return None


if __name__ == "__main__":
    all_data = get_all_flight_data()

    print(f"Total flight data records: {len(all_data)}")
    print(f"Policies: {len([d for d in all_data if d['type'] == 'policy'])}")
    print(f"Routes: {len([d for d in all_data if d['type'] == 'route'])}")
    print(f"Airlines: {len([d for d in all_data if d['type'] == 'airline'])}")
    print(f"Airports: {len([d for d in all_data if d['type'] == 'airport'])}")
    print(f"Booking Classes: {len([d for d in all_data if d['type'] == 'booking_class'])}")

    # Test search functions
    print(f"\nRoutes from JFK: {len(search_routes(departure='JFK'))}")
    print(f"Routes to LAX: {len(search_routes(arrival='LAX'))}")

    # Show categories
    categories = set(d["category"] for d in all_data)
    print(f"\nCategories: {', '.join(categories)}")
