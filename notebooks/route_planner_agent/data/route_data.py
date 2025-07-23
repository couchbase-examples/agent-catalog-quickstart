"""
Route Data Knowledge Base

This module contains comprehensive travel and route information for the route planner agent.
Includes cities, landmarks, transportation options, and points of interest.
"""

def get_travel_knowledge_base():
    """
    Returns comprehensive travel and route knowledge base for vector search indexing.

    Returns:
        list: List of travel knowledge documents with metadata
    """

    travel_data = [
        # Major Cities and Routes
        {
            "title": "San Francisco to Los Angeles Route",
            "content": "The classic California coastal drive from San Francisco to Los Angeles covers approximately 380 miles via Highway 101 or 400 miles via scenic Highway 1 (Pacific Coast Highway). The PCH route takes 8-10 hours but offers breathtaking ocean views, passing through Monterey, Carmel, Big Sur, and Santa Barbara. Popular stops include Hearst Castle, McWay Falls, and Pismo Beach. Best time to travel is spring through fall. Highway 101 is faster at 6-7 hours but less scenic.",
            "metadata": {
                "route_type": "scenic_drive",
                "distance": "380-400 miles",
                "duration": "6-10 hours",
                "cities": ["San Francisco", "Los Angeles", "Monterey", "Santa Barbara"],
                "transport_mode": "car",
                "region": "California",
                "difficulty": "easy",
                "season": "year_round"
            }
        },

        {
            "title": "New York City to Washington DC Route",
            "content": "Multiple transportation options connect NYC and Washington DC. By car via I-95, the 225-mile journey takes 4-5 hours depending on traffic. Amtrak's Acela Express offers 3-hour high-speed rail service with frequent departures. Regular Amtrak trains take 3.5 hours. Bus services like Greyhound and Megabus provide budget options at 4-6 hours. Flying takes 1.5 hours but adds airport time. Popular stops include Philadelphia and Baltimore. Avoid driving during rush hours.",
            "metadata": {
                "route_type": "urban_corridor",
                "distance": "225 miles",
                "duration": "3-6 hours",
                "cities": ["New York", "Washington DC", "Philadelphia", "Baltimore"],
                "transport_mode": "multi_modal",
                "region": "Northeast",
                "difficulty": "moderate",
                "traffic": "heavy"
            }
        },

        {
            "title": "Chicago to Milwaukee Route",
            "content": "Short scenic route connecting two Great Lakes cities. Driving via I-94 covers 90 miles in 1.5-2 hours. Amtrak's Hiawatha Service offers comfortable 1.5-hour train rides with 7 daily departures. The route passes through pleasant Wisconsin countryside and Lake Michigan shoreline views. Milwaukee offers famous breweries, lakefront, and German heritage. No major stops between cities. Train is often preferable to avoid Chicago traffic and parking costs.",
            "metadata": {
                "route_type": "regional",
                "distance": "90 miles",
                "duration": "1.5-2 hours",
                "cities": ["Chicago", "Milwaukee"],
                "transport_mode": "train_car",
                "region": "Midwest",
                "difficulty": "easy",
                "feature": "great_lakes"
            }
        },

        # Points of Interest and Attractions
        {
            "title": "National Parks Route Planning",
            "content": "Planning multi-park road trips requires careful timing and routing. Popular circuits include: Utah's Big Five (Arches, Canyonlands, Capitol Reef, Bryce Canyon, Zion) taking 7-10 days; California's Sierra Circuit (Yosemite, Sequoia, Kings Canyon) requiring 5-7 days; Arizona's Desert Parks (Grand Canyon, Sedona, Antelope Canyon) needing 4-6 days. Spring and fall offer best weather. Book accommodations well in advance. Consider park passes for multiple visits.",
            "metadata": {
                "route_type": "nature_tour",
                "poi_category": "national_parks",
                "region": "Western US",
                "duration": "4-10 days",
                "transport_mode": "car",
                "difficulty": "moderate",
                "season": "spring_fall"
            }
        },

        {
            "title": "Food and Wine Route - Napa Valley",
            "content": "Napa Valley wine country offers world-class vineyards and restaurants within 30 miles. Plan 2-3 days minimum. Start in Napa city, visit Oxbow Public Market, then proceed north through St. Helena to Calistoga. Must-visit wineries include Robert Mondavi, Castello di Amorosa, and Schramsberg. Book tastings in advance. Consider wine train or guided tours to avoid driving. Pair with Michelin-starred restaurants like French Laundry (Yountville) or Auberge du Soleil (Rutherford).",
            "metadata": {
                "route_type": "culinary_tour",
                "poi_category": "wine_country",
                "region": "Northern California",
                "duration": "2-3 days",
                "cities": ["Napa", "St. Helena", "Calistoga", "Yountville"],
                "transport_mode": "car_tour",
                "difficulty": "easy",
                "feature": "wine_tasting"
            }
        },

        {
            "title": "Historic Route 66 Experience",
            "content": "The legendary Route 66 spans 2,448 miles from Chicago to Santa Monica, crossing 8 states. Complete journey takes 2-3 weeks. Key segments include: Chicago to St. Louis (classic Americana), Oklahoma City to Amarillo (Texas Panhandle), Albuquerque to Flagstaff (Southwest culture), and Barstow to Santa Monica (desert to ocean). Essential stops: Cadillac Ranch (Texas), Petrified Forest (Arizona), and various vintage motels and diners. Spring and fall are ideal for travel.",
            "metadata": {
                "route_type": "historic_scenic",
                "distance": "2,448 miles",
                "duration": "2-3 weeks",
                "cities": ["Chicago", "St. Louis", "Oklahoma City", "Albuquerque", "Flagstaff", "Santa Monica"],
                "transport_mode": "car",
                "region": "Cross Country",
                "difficulty": "challenging",
                "feature": "historic_route"
            }
        },

        # Transportation Modes and Options
        {
            "title": "European Train Travel - High-Speed Networks",
            "content": "Europe's integrated rail network enables efficient multi-country travel. High-speed trains include: TGV (France, 200+ mph), ICE (Germany, 200 mph), AVE (Spain, 192 mph), and Eurostar (UK-France, 186 mph). Eurail passes offer flexible travel across 33 countries. Popular routes: Paris-London (3.5 hours), Paris-Barcelona (6.5 hours), Munich-Vienna (4 hours). Book seats in advance for popular routes. Consider regional trains for scenic journeys through Alps or Rhine Valley.",
            "metadata": {
                "route_type": "international",
                "transport_mode": "high_speed_rail",
                "region": "Europe",
                "poi_category": "transportation",
                "duration": "varies",
                "difficulty": "easy",
                "feature": "rail_network"
            }
        },

        {
            "title": "Urban Public Transportation Planning",
            "content": "Major cities offer comprehensive public transit. New York: subway, bus, taxi apps (avoid driving Manhattan). London: Underground, buses, river boats (Oyster card recommended). Tokyo: JR trains, subway, buses (get JR Pass for tourists). San Francisco: BART, Muni, cable cars, ride-sharing. Download city transit apps, buy day/week passes, and learn basic navigation. Consider bike-sharing for short distances. Walking often fastest for nearby destinations.",
            "metadata": {
                "route_type": "urban",
                "transport_mode": "public_transit",
                "cities": ["New York", "London", "Tokyo", "San Francisco"],
                "poi_category": "transportation",
                "difficulty": "moderate",
                "feature": "city_navigation"
            }
        },

        # Outdoor and Adventure Routes
        {
            "title": "Hiking and Backpacking Routes",
            "content": "Epic hiking routes for outdoor enthusiasts: Appalachian Trail (2,190 miles, 5-7 months), Pacific Crest Trail (2,650 miles, 4-6 months), John Muir Trail (211 miles, 2-3 weeks), Tour du Mont Blanc (110 miles, 7-11 days). Day hikes: Angels Landing (Zion), Half Dome (Yosemite), Mount Washington (New Hampshire). Check permits, weather, and fitness requirements. Carry proper gear, maps, and emergency supplies. Consider guided trips for challenging routes.",
            "metadata": {
                "route_type": "outdoor_adventure",
                "poi_category": "hiking",
                "transport_mode": "hiking",
                "difficulty": "challenging",
                "duration": "1 day to 7 months",
                "feature": "wilderness",
                "region": "various"
            }
        },

        {
            "title": "Coastal and Island Routes",
            "content": "Scenic coastal drives and island hopping adventures: Big Sur Coastline (California), Oregon Coast Highway, Maine's Acadia Loop Road, Florida Keys Overseas Highway. Island routes: Hawaiian island hopping (inter-island flights), Greek Islands ferry routes, Caribbean island cruises. Consider weather, ferry schedules, and seasonal accessibility. Coastal routes offer lighthouse visits, seafood restaurants, and beach access. Plan for slower speeds on winding coastal roads.",
            "metadata": {
                "route_type": "coastal",
                "poi_category": "beaches_islands",
                "transport_mode": "car_ferry",
                "difficulty": "easy_moderate",
                "feature": "ocean_views",
                "season": "varies_by_location"
            }
        },

        # City-Specific Information
        {
            "title": "Los Angeles Area Route Planning",
            "content": "LA's sprawling geography requires strategic planning. Popular routes: Hollywood to Santa Monica (Sunset Boulevard), Downtown to Beverly Hills (Wilshire Corridor), LAX to Disneyland (405 to 5 freeway). Traffic peaks 7-10 AM and 4-7 PM weekdays. Use apps like Waze for real-time routing. Consider Metro Rail for airport-downtown connection. Beach cities (Santa Monica, Venice, Manhattan Beach) accessible via Pacific Coast Highway. Allow extra time for parking in popular areas.",
            "metadata": {
                "route_type": "urban",
                "city": "Los Angeles",
                "transport_mode": "car_transit",
                "region": "Southern California",
                "difficulty": "moderate",
                "traffic": "heavy",
                "feature": "sprawling_city"
            }
        },

        {
            "title": "Boston Freedom Trail and Historic Sites",
            "content": "Boston's compact size enables walking tours. Freedom Trail (2.5 miles) connects 16 historic sites including Boston Common, Faneuil Hall, and USS Constitution. Takes 2-4 hours walking. Combine with harbor tours, North End Italian dining, and Fenway Park visits. Use T (subway) for longer distances - Green Line to Fenway, Blue Line to airport. Walking most efficient in downtown area. Consider Boston CityPASS for attraction discounts.",
            "metadata": {
                "route_type": "historic_walking",
                "city": "Boston",
                "poi_category": "historic_sites",
                "transport_mode": "walking",
                "duration": "2-4 hours",
                "difficulty": "easy",
                "feature": "walkable_city",
                "region": "New England"
            }
        },

        # Seasonal and Weather Considerations
        {
            "title": "Winter Travel Route Considerations",
            "content": "Winter driving requires special preparation and route planning. Mountain passes may require chains or 4WD: Sierra Nevada (California), Rockies (Colorado), Cascades (Washington). Check road conditions and weather forecasts. Popular winter destinations: Aspen/Vail (skiing), Yellowstone (limited access), New England (fall foliage extends to early winter). Carry emergency supplies, extra warm clothing, and ensure vehicle winterization. Consider train or bus for mountain travel during storms.",
            "metadata": {
                "route_type": "seasonal",
                "season": "winter",
                "transport_mode": "car_train",
                "difficulty": "challenging",
                "feature": "mountain_weather",
                "poi_category": "winter_sports",
                "region": "mountainous_areas"
            }
        },

        {
            "title": "Summer Festival and Event Routes",
            "content": "Summer brings numerous festivals requiring advance planning. Music festivals: Coachella (California), Bonnaroo (Tennessee), Lollapalooza (Chicago). Art festivals: Burning Man (Nevada), Art Basel (Miami). Food festivals: Taste of Chicago, Maine Lobster Festival. Book accommodations early as nearby hotels fill quickly. Consider camping options at music festivals. Plan alternative routes as festivals cause local traffic congestion. Use ride-sharing or shuttles when available.",
            "metadata": {
                "route_type": "event_based",
                "season": "summer",
                "poi_category": "festivals",
                "transport_mode": "various",
                "difficulty": "moderate",
                "feature": "crowded_events",
                "planning": "advance_booking_required"
            }
        },

        {
            "title": "New York to Boston Route",
            "content": "The Northeast corridor from New York to Boston offers multiple transportation options. By car via I-95 North, the 215-mile journey takes 4-5 hours depending on traffic, passing through Connecticut. Amtrak's Acela Express provides 3.5-hour high-speed rail service with frequent departures. Regular Amtrak trains take 4-4.5 hours. Bus services like Greyhound and Peter Pan offer budget options at 4.5-5.5 hours. Flying takes 1.5 hours but adds airport time. Popular stops include New Haven, Hartford, and Providence. Avoid driving during rush hours and summer beach traffic.",
            "metadata": {
                "route_type": "urban_corridor",
                "distance": "215 miles",
                "duration": "3.5-5.5 hours",
                "cities": ["New York", "Boston", "New Haven", "Hartford", "Providence"],
                "transport_mode": "multi_modal",
                "region": "Northeast",
                "difficulty": "moderate",
                "traffic": "heavy"
            }
        },

        {
            "title": "Chicago to Detroit Route",
            "content": "The Great Lakes connection from Chicago to Detroit spans 280 miles via I-94 East through Indiana and Michigan. Driving takes 4.5-5 hours with minimal traffic, passing through Kalamazoo and Battle Creek. Amtrak's Wolverine service offers daily train service taking 5.5-6 hours with scenic views of Lake Michigan shoreline. Bus services provide budget options at 5-6 hours. The route passes through Indiana Dunes, Michigan wine country, and historic towns. Best driving times are mid-morning or early afternoon to avoid Chicago rush hour.",
            "metadata": {
                "route_type": "regional",
                "distance": "280 miles", 
                "duration": "4.5-6 hours",
                "cities": ["Chicago", "Detroit", "Kalamazoo", "Battle Creek"],
                "transport_mode": "car_train",
                "region": "Great Lakes",
                "difficulty": "easy",
                "feature": "lake_views"
            }
        },

        {
            "title": "Colorado Scenic Mountain Routes",
            "content": "Colorado offers spectacular mountain driving with numerous scenic routes. The Million Dollar Highway (US 550) from Durango to Silverton features dramatic cliff-side driving through the San Juan Mountains. Trail Ridge Road in Rocky Mountain National Park reaches 12,183 feet elevation with alpine tundra views (seasonal, typically May-October). The Peak to Peak Highway (CO 7) connects Estes Park to Central City through golden aspen forests. Independence Pass (CO 82) from Aspen to Leadville offers 12,095-foot summit views. Mount Evans Scenic Byway reaches 14,130 feet, the highest paved road in North America.",
            "metadata": {
                "route_type": "scenic_mountain",
                "region": "Colorado",
                "transport_mode": "car",
                "difficulty": "challenging",
                "season": "May-October",
                "feature": "mountain_views",
                "elevation": "high_altitude"
            }
        }
    ]

    return travel_data


def get_cities_data():
    """
    Returns detailed information about major cities for route planning.

    Returns:
        list: List of city information with transportation and attractions
    """

    cities_data = [
        {
            "name": "New York City",
            "region": "Northeast",
            "airports": ["JFK", "LGA", "EWR"],
            "train_stations": ["Penn Station", "Grand Central"],
            "public_transit": ["Subway", "Bus", "Taxi", "Ferry"],
            "attractions": ["Central Park", "Statue of Liberty", "Times Square", "Brooklyn Bridge"],
            "neighborhoods": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
            "best_months": ["April", "May", "September", "October"],
            "avoid_driving": True,
            "walking_friendly": True
        },

        {
            "name": "San Francisco",
            "region": "Northern California",
            "airports": ["SFO", "OAK", "SJC"],
            "train_stations": ["Caltrain", "BART"],
            "public_transit": ["BART", "Muni", "Cable Car", "Ferry"],
            "attractions": ["Golden Gate Bridge", "Alcatraz", "Fisherman's Wharf", "Lombard Street"],
            "neighborhoods": ["Mission", "Castro", "Chinatown", "North Beach"],
            "best_months": ["September", "October", "November"],
            "hills": True,
            "parking_expensive": True
        },

        {
            "name": "Chicago",
            "region": "Midwest",
            "airports": ["ORD", "MDW"],
            "train_stations": ["Union Station"],
            "public_transit": ["L Train", "Bus"],
            "attractions": ["Millennium Park", "Navy Pier", "Art Institute", "Wrigley Field"],
            "neighborhoods": ["Loop", "River North", "Lincoln Park", "Wicker Park"],
            "best_months": ["May", "June", "September", "October"],
            "winter_harsh": True,
            "lakefront": True
        }
    ]

    return cities_data


def get_transportation_modes():
    """
    Returns information about different transportation modes and their characteristics.

    Returns:
        list: List of transportation mode information
    """

    transport_data = [
        {
            "mode": "Car",
            "pros": ["Flexibility", "Door-to-door", "Luggage space", "Schedule control"],
            "cons": ["Traffic", "Parking costs", "Gas prices", "Driver fatigue"],
            "best_for": ["Rural areas", "Multiple stops", "Groups", "Scenic routes"],
            "costs": ["Gas", "Tolls", "Parking", "Rental fees"],
            "planning_factors": ["Traffic patterns", "Road conditions", "Weather", "Construction"]
        },

        {
            "mode": "Train",
            "pros": ["No traffic", "Relaxing", "City center to city center", "Productive time"],
            "cons": ["Limited routes", "Schedule dependent", "Luggage restrictions"],
            "best_for": ["Urban corridors", "Long distances", "Business travel"],
            "costs": ["Tickets", "Reservations", "Meals"],
            "planning_factors": ["Schedules", "Advance booking", "Seat selection"]
        },

        {
            "mode": "Flying",
            "pros": ["Speed", "Long distances", "Frequent schedules"],
            "cons": ["Airport time", "Security", "Weather delays", "Baggage fees"],
            "best_for": ["Long distances", "Time constraints", "International"],
            "costs": ["Airfare", "Baggage", "Airport parking", "Ground transport"],
            "planning_factors": ["Advance booking", "Airport location", "Connection time"]
        },

        {
            "mode": "Bus",
            "pros": ["Low cost", "No driving", "WiFi available", "Multiple stops"],
            "cons": ["Slower", "Limited comfort", "Schedule delays"],
            "best_for": ["Budget travel", "Students", "Short to medium distances"],
            "costs": ["Low ticket prices"],
            "planning_factors": ["Schedule reliability", "Boarding locations", "Comfort level"]
        }
    ]

    return transport_data


def load_route_data_to_couchbase(cluster, bucket_name, scope_name, collection_name, embeddings, index_name=None):
    """Load route data into Couchbase vector store using LlamaIndex.
    
    Args:
        cluster: Connected Couchbase cluster
        bucket_name: Name of the bucket
        scope_name: Name of the scope
        collection_name: Name of the collection
        embeddings: Configured embeddings model
        index_name: Optional index name (defaults to collection_name + '_index')
    """
    from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
    from llama_index.core import Document, VectorStoreIndex, StorageContext
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    
    if not index_name:
        index_name = f"{collection_name}_index"
    
    print(f"Loading route data to {bucket_name}.{scope_name}.{collection_name}")
    
    try:
        # Initialize LlamaIndex vector store with provided parameters
        vector_store = CouchbaseSearchVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            index_name=index_name,
        )
        print("✓ LlamaIndex vector store initialized")
        
        # Get route data
        travel_knowledge = get_travel_knowledge_base()
        
        print(f"Retrieved {len(travel_knowledge)} route knowledge entries")
        
        # Create LlamaIndex Document objects
        documents = []
        
        for i, entry in enumerate(travel_knowledge):
            # Create searchable text combining title and content
            text = f"{entry['title']}: {entry['content']}"
            
            # Create metadata
            metadata = {
                "route_id": f"route_{i+1}",
                "title": entry["title"],
                "type": "route",
                "source": "route_data"
            }
            
            # Add entry metadata if available
            if "metadata" in entry:
                # Convert list values to strings for Couchbase compatibility
                for key, value in entry["metadata"].items():
                    if isinstance(value, list):
                        metadata[key] = ", ".join(str(v) for v in value)
                    else:
                        metadata[key] = value
            
            # Create LlamaIndex Document
            doc = Document(
                text=text,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"Created {len(documents)} LlamaIndex documents")
        
        # Create ingestion pipeline with sentence splitter and embeddings
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                embeddings
            ],
            vector_store=vector_store
        )
        
        # Ingest documents using LlamaIndex pipeline
        print("⚡ Ingesting documents using LlamaIndex pipeline...")
        pipeline.run(documents=documents)
        
        print(f"✓ Successfully loaded {len(documents)} routes using LlamaIndex")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error loading route data: {e}")
        raise
