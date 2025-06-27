#!/usr/bin/env python3
"""
Direct test of Couchbase connection and SQL++ queries
"""

import os
import dotenv
import couchbase.cluster
import couchbase.auth
import couchbase.options

# Load environment
dotenv.load_dotenv()


def test_flight_lookup(source_airport: str, destination_airport: str):
    """Test the flight lookup SQL++ query directly"""

    try:
        # Connect to Couchbase
        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING"),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv("CB_USERNAME"), password=os.getenv("CB_PASSWORD")
                )
            ),
        )

        # Execute the same query as in lookup_flight_info.sqlpp
        query = """
        SELECT 
          r.airline,
          r.sourceairport,
          r.destinationairport,
          r.equipment,
          r.distance,
          a1.airportname AS source_name,
          a1.city AS source_city,
          a1.country AS source_country,
          a2.airportname AS dest_name, 
          a2.city AS dest_city,
          a2.country AS dest_country,
          CASE 
            WHEN r.distance < 500 THEN "Short-haul"
            WHEN r.distance < 1500 THEN "Medium-haul"
            ELSE "Long-haul"
          END AS flight_type,
          ROUND(r.distance / 500, 1) AS estimated_hours
        FROM 
          `travel-sample`.inventory.route r
        JOIN 
          `travel-sample`.inventory.airport a1 ON r.sourceairport = a1.faa
        JOIN 
          `travel-sample`.inventory.airport a2 ON r.destinationairport = a2.faa
        WHERE 
          r.sourceairport = $source_airport AND
          r.destinationairport = $destination_airport
        ORDER BY r.distance
        LIMIT 10
        """

        result = cluster.query(
            query,
            couchbase.options.QueryOptions(
                named_parameters={
                    "source_airport": source_airport,
                    "destination_airport": destination_airport,
                }
            ),
        )

        flights = []
        for row in result:
            flights.append(row)

        print(f"✅ Found {len(flights)} flights from {source_airport} to {destination_airport}")
        for i, flight in enumerate(flights, 1):
            print(
                f"{i}. {flight['airline']} - {flight['equipment']} - {flight['distance']} miles - {flight['flight_type']}"
            )
            print(
                f"   {flight['source_name']} ({flight['source_city']}) → {flight['dest_name']} ({flight['dest_city']})"
            )

        return flights

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def test_hotel_search(query: str):
    """Test hotel search as knowledge base proxy"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING"),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv("CB_USERNAME"), password=os.getenv("CB_PASSWORD")
                )
            ),
        )

        # Search hotels by description (simple text search)
        search_query = """
        SELECT h.name, h.description, h.city, h.country
        FROM `travel-sample`.inventory.hotel h
        WHERE LOWER(h.description) LIKE LOWER($search_term)
        LIMIT 5
        """

        result = cluster.query(
            search_query,
            couchbase.options.QueryOptions(named_parameters={"search_term": f"%{query}%"}),
        )

        hotels = []
        for row in result:
            hotels.append(row)

        print(f"✅ Found {len(hotels)} hotels matching '{query}'")
        for i, hotel in enumerate(hotels, 1):
            print(f"{i}. {hotel['name']} in {hotel['city']}, {hotel['country']}")
            print(f"   {hotel['description'][:100]}...")

        return hotels

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def explore_available_routes():
    """Explore what routes are actually available in the database"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING"),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv("CB_USERNAME"), password=os.getenv("CB_PASSWORD")
                )
            ),
        )

        # Find routes starting with ABE
        query = """
        SELECT DISTINCT r.sourceairport, r.destinationairport, a1.airportname AS source_name, a2.airportname AS dest_name
        FROM `travel-sample`.inventory.route r
        JOIN `travel-sample`.inventory.airport a1 ON r.sourceairport = a1.faa
        JOIN `travel-sample`.inventory.airport a2 ON r.destinationairport = a2.faa
        WHERE r.sourceairport = 'ABE' OR r.sourceairport = 'ABI'
        ORDER BY r.sourceairport, r.destinationairport
        """

        result = cluster.query(query)
        
        routes = []
        for row in result:
            routes.append(row)

        print(f"✅ Found {len(routes)} routes from ABE/ABI airports:")
        for route in routes:
            print(f"   {route['sourceairport']} → {route['destinationairport']}: {route['source_name']} to {route['dest_name']}")

        return routes

    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def test_simplified_query(source_airport: str, destination_airport: str):
    """Test the simplified SQL++ query structure"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv("CB_CONN_STRING"),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv("CB_USERNAME"), password=os.getenv("CB_PASSWORD")
                )
            ),
        )

        # Test the simplified query structure
        query = """
        SELECT 
          r.airline,
          r.sourceairport,
          r.destinationairport,
          r.equipment,
          r.distance,
          a1.airportname AS source_name,
          a1.city AS source_city,
          a1.country AS source_country,
          a2.airportname AS dest_name, 
          a2.city AS dest_city,
          a2.country AS dest_country,
          CASE 
            WHEN r.distance < 500 THEN "Short-haul"
            WHEN r.distance < 1500 THEN "Medium-haul"
            ELSE "Long-haul"
          END AS flight_type,
          ROUND(r.distance / 500, 1) AS estimated_hours,
          CASE 
            WHEN r.airline = 'DL' THEN "08:30"
            WHEN r.airline = 'AF' THEN "12:15"
            WHEN r.airline = 'KL' THEN "16:45"
            ELSE "20:00"
          END AS departure_time,
          r.airline || "123" AS flight_number,
          CASE 
            WHEN r.distance < 500 THEN 200 + 20
            WHEN r.distance < 1500 THEN 400 + 30  
            ELSE 800 + 50
          END AS estimated_price
        FROM 
          `travel-sample`.inventory.route r
        JOIN 
          `travel-sample`.inventory.airport a1 ON r.sourceairport = a1.faa
        JOIN 
          `travel-sample`.inventory.airport a2 ON r.destinationairport = a2.faa
        WHERE 
          r.sourceairport = $source_airport AND
          r.destinationairport = $destination_airport
        LIMIT 5
        """

        result = cluster.query(
            query,
            couchbase.options.QueryOptions(
                named_parameters={
                    "source_airport": source_airport,
                    "destination_airport": destination_airport,
                }
            ),
        )

        flights = []
        for row in result:
            flights.append(row)

        print(f"✅ Simplified query found {len(flights)} flights from {source_airport} to {destination_airport}")
        for i, flight in enumerate(flights, 1):
            print(
                f"{i}. {flight['airline']} - {flight['equipment']} - {flight['distance']} miles - {flight['flight_type']} - ${flight['estimated_price']} at {flight['departure_time']}"
            )

        return flights

    except Exception as e:
        print(f"❌ Simplified query error: {e}")
        return []


if __name__ == "__main__":
    print("🧪 Testing Direct Couchbase Connection")
    print("=" * 50)

    print("\n🔍 Exploring Available Routes:")
    explore_available_routes()

    print("\n1. Testing Flight Lookup (SFO → LAX):")
    test_flight_lookup("SFO", "LAX")

    print("\n2. Testing Flight Lookup (ABE → ATL) if available:")
    test_flight_lookup("ABE", "ATL")
    
    print("\n3. Testing Flight Lookup (ABI → DFW) if available:")
    test_flight_lookup("ABI", "DFW")

    print("\n4. Testing Hotel Search (knowledge base proxy):")
    test_hotel_search("luxury")

    print("\n5. Testing Simplified Query (ABE → ATL):")
    test_simplified_query("ABE", "ATL")

    print("\n6. Testing Hotel Search (policy proxy):")
    test_hotel_search("business")
