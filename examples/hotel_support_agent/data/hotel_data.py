"""
Hotel data module for the hotel support agent demo.
Contains mock hotel data used for vector search and detailed hotel information.
"""

def get_hotel_texts():
    """Returns formatted hotel texts for vector store embedding, extracted from detailed hotel data."""
    detailed_hotels = get_detailed_hotel_data()
    hotel_texts = []
    
    for hotel_info in detailed_hotels.values():
        # Extract basic amenities (first few) for the search text
        basic_amenities = hotel_info['amenities'][:6]  # Take first 6 amenities
        amenities_text = ', '.join(basic_amenities)
        
        text = f"{hotel_info['name']} in {hotel_info['location']}. {hotel_info['description']} Price range: {hotel_info['price_range']}. Rating: {hotel_info['rating']}. Amenities: {amenities_text}"
        hotel_texts.append(text)
    
    return hotel_texts

def get_detailed_hotel_data():
    """Returns detailed hotel information for the get_hotel_details tool."""
    detailed_hotels = {
        "grand palace hotel": {
            "name": "Grand Palace Hotel",
            "location": "Manhattan, New York City",
            "description": "Luxury 5-star hotel featuring elegant rooms with Manhattan skyline views",
            "price_range": "$300-$500 per night",
            "amenities": ["Rooftop Pool", "World-class Spa", "24/7 Fitness Center", "Michelin-starred Restaurant", "24/7 Room Service", "High-speed WiFi", "Concierge Service", "Valet Parking"],
            "rating": "4.8/5",
            "contact": "Phone: +1-212-555-0123, Email: reservations@grandpalacehotel.com",
            "address": "123 Fifth Avenue, Manhattan, NY 10001",
            "check_in": "3:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "seaside resort": {
            "name": "Seaside Resort",
            "location": "Miami Beach, Florida", 
            "description": "Oceanfront resort with private beach and stunning Atlantic Ocean views",
            "price_range": "$200-$400 per night",
            "amenities": ["Private Beach Access", "3 Swimming Pools", "Oceanview Restaurant", "Tiki Bar", "Water Sports Center", "Free WiFi", "Spa Services", "Kids Club"],
            "rating": "4.6/5",
            "contact": "Phone: +1-305-555-0456, Email: info@seasideresort.com",
            "address": "789 Ocean Drive, Miami Beach, FL 33139",
            "check_in": "4:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 48 hours before check-in"
        },
        "mountain lodge": {
            "name": "Mountain Lodge",
            "location": "Aspen, Colorado",
            "description": "Rustic mountain retreat perfect for skiing and outdoor adventures",
            "price_range": "$150-$300 per night",
            "amenities": ["Ski-in/Ski-out Access", "Stone Fireplace Lounge", "Mountain View Restaurant", "Outdoor Hot Tub", "Hiking Trail Access", "Free WiFi", "Game Room", "Ski Equipment Rental"],
            "rating": "4.5/5",
            "contact": "Phone: +1-970-555-0789, Email: bookings@mountainlodge.com",
            "address": "456 Alpine Way, Aspen, CO 81611",
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 72 hours before check-in"
        },
        "business center hotel": {
            "name": "Business Center Hotel",
            "location": "Downtown Chicago, Illinois",
            "description": "Modern business hotel with state-of-the-art conference facilities",
            "price_range": "$180-$280 per night",
            "amenities": ["24/7 Business Center", "Executive Meeting Rooms", "Fitness Center", "Business Lounge Restaurant", "Free WiFi", "Self-parking", "Express Check-in/out", "Printing Services"],
            "rating": "4.3/5",
            "contact": "Phone: +1-312-555-0321, Email: corporate@businesscenterhotel.com",
            "address": "321 Michigan Avenue, Chicago, IL 60601",
            "check_in": "3:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "boutique inn": {
            "name": "Boutique Inn",
            "location": "San Francisco, California",
            "description": "Charming boutique hotel with unique artistic decor and personalized service",
            "price_range": "$220-$350 per night",
            "amenities": ["Personal Concierge", "Artisan Restaurant", "Craft Cocktail Bar", "Free WiFi", "Pet-friendly Policies", "Valet Parking", "Custom Room Design", "Local Art Gallery"],
            "rating": "4.7/5",
            "contact": "Phone: +1-415-555-0654, Email: stay@boutiqueinn.com",
            "address": "987 Union Square, San Francisco, CA 94108",
            "check_in": "4:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "budget inn sf": {
            "name": "Budget Inn SF",
            "location": "San Francisco Airport Area, California",
            "description": "Affordable hotel near San Francisco International Airport with shuttle service",
            "price_range": "$80-$120 per night",
            "amenities": ["Free Airport Shuttle", "24/7 Front Desk", "Free WiFi", "Continental Breakfast", "Self-parking", "Business Center", "Fitness Room", "Laundry Facilities"],
            "rating": "4.0/5",
            "contact": "Phone: +1-650-555-0987, Email: info@budgetinnsf.com",
            "address": "456 Airport Blvd, San Francisco, CA 94128",
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "los angeles downtown hotel": {
            "name": "Los Angeles Downtown Hotel",
            "location": "Downtown Los Angeles, California",
            "description": "Modern urban hotel in the heart of LA with rooftop dining and city views",
            "price_range": "$160-$240 per night",
            "amenities": ["Free Continental Breakfast", "Self-parking Available", "Rooftop Restaurant", "24/7 Fitness Center", "Free WiFi", "Business Center", "Concierge Service", "Room Service"],
            "rating": "4.4/5",
            "contact": "Phone: +1-213-555-0432, Email: reservations@ladowntownhotel.com",
            "address": "789 Spring Street, Los Angeles, CA 90014",
            "check_in": "3:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "la luxury resort": {
            "name": "LA Luxury Resort",
            "location": "Beverly Hills, Los Angeles, California",
            "description": "Luxury resort in Beverly Hills with world-class amenities and spa services",
            "price_range": "$400-$650 per night",
            "amenities": ["Complimentary Breakfast", "Valet Parking", "Full-Service Spa", "Multiple Pools", "Fine Dining Restaurant", "Free WiFi", "Concierge Service", "Golf Course Access"],
            "rating": "4.9/5",
            "contact": "Phone: +1-310-555-0876, Email: reservations@laluxuryresort.com",
            "address": "321 Rodeo Drive, Beverly Hills, CA 90210",
            "check_in": "4:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 48 hours before check-in"
        },
        "ocean breeze resort": {
            "name": "Ocean Breeze Resort",
            "location": "Malibu, California",
            "description": "Luxury oceanfront resort with private beach, world-class spa, and championship golf course",
            "price_range": "$400-$600 per night",
            "amenities": ["Private Beach Access", "Full-Service Spa", "Championship Golf Course", "Infinity Pool", "Multiple Fine Dining Restaurants", "Free WiFi", "Tennis Court", "Yacht Charter"],
            "rating": "4.9/5",
            "contact": "Phone: +1-310-555-0987, Email: reservations@oceanbreezeresort.com",
            "address": "12345 Pacific Coast Highway, Malibu, CA 90265",
            "check_in": "4:00 PM",
            "check_out": "12:00 PM",
            "cancellation": "Free cancellation up to 72 hours before check-in"
        },
        "city loft hotel": {
            "name": "City Loft Hotel",
            "location": "Seattle, Washington",
            "description": "Modern urban hotel with industrial design, rooftop bar, and downtown location",
            "price_range": "$160-$240 per night",
            "amenities": ["Rooftop Bar", "24/7 Fitness Center", "Business Center", "Contemporary Restaurant", "Free WiFi", "Pet-friendly Policies", "Valet Parking", "Electric Car Charging"],
            "rating": "4.4/5",
            "contact": "Phone: +1-206-555-0432, Email: info@citylofthotel.com",
            "address": "888 Pike Street, Seattle, WA 98101",
            "check_in": "3:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 24 hours before check-in"
        },
        "wellness retreat": {
            "name": "Wellness Retreat",
            "location": "Sedona, Arizona",
            "description": "Spiritual wellness resort with meditation gardens, full spa services, and red rock views",
            "price_range": "$250-$450 per night",
            "amenities": ["Full-Service Spa", "Meditation Gardens", "Yoga Studio", "Infinity Pool", "Farm-to-table Restaurant", "Free WiFi", "Hiking Trails", "Spiritual Wellness Programs"],
            "rating": "4.8/5",
            "contact": "Phone: +1-928-555-0876, Email: reservations@wellnesssedona.com",
            "address": "567 Red Rock Drive, Sedona, AZ 86336",
            "check_in": "4:00 PM",
            "check_out": "11:00 AM",
            "cancellation": "Free cancellation up to 48 hours before check-in"
        }
    }
    
    return detailed_hotels


def load_hotel_data_to_couchbase(cluster, bucket_name, scope_name, collection_name, embeddings, index_name=None):
    """Load hotel data into Couchbase vector store.
    
    Args:
        cluster: Connected Couchbase cluster
        bucket_name: Name of the bucket
        scope_name: Name of the scope
        collection_name: Name of the collection
        embeddings: Configured embeddings model
        index_name: Optional index name (defaults to collection_name + '_index')
    """
    from langchain_couchbase.vectorstores import CouchbaseVectorStore
    
    if not index_name:
        index_name = f"{collection_name}_index"
    
    print(f"Loading hotel data to {bucket_name}.{scope_name}.{collection_name}")
    
    print(f"Cluster: {cluster}")
    print(f"Bucket name: {bucket_name}")
    print(f"Scope name: {scope_name}")
    print(f"Collection name: {collection_name}")
    print(f"Embeddings: {embeddings}")
    print(f"Index name: {index_name}")
    
    try:
        # Initialize vector store with provided parameters
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=bucket_name,
            scope_name=scope_name,
            collection_name=collection_name,
            embedding=embeddings,
            index_name=index_name,
        )
        print("✓ Vector store initialized")
        
        # Get hotel data
        hotel_texts = get_hotel_texts()
        detailed_hotels = get_detailed_hotel_data()
        
        print(f"Retrieved {len(hotel_texts)} hotel descriptions")
        
        # Prepare documents for ingestion
        texts = []
        metadatas = []
        
        for i, text in enumerate(hotel_texts):
            texts.append(text)
            
            # Extract hotel name from text (first part before " in ")
            hotel_name = text.split(" in ")[0].lower()
            
            # Find corresponding detailed data
            detailed_info = detailed_hotels.get(hotel_name, {})
            
            # Create metadata
            metadata = {
                "hotel_id": f"hotel_{i+1}",
                "name": detailed_info.get("name", hotel_name),
                "location": detailed_info.get("location", ""),
                "rating": detailed_info.get("rating", ""),
                "price_range": detailed_info.get("price_range", ""),
                "type": "hotel",
                "source": "hotel_data"
            }
            metadatas.append(metadata)
        
        print(f"Prepared {len(texts)} hotel documents for ingestion")
        
        # Add documents to vector store
        print(f"Adding {len(texts)} documents to vector store...")
        try:
            vector_store.add_texts(texts=texts, metadatas=metadatas)
            print(f"✓ Successfully added {len(texts)} documents")
            successful_count = len(texts)
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
            raise
        
        if successful_count == len(texts):
            print(f"✅ Successfully loaded all {successful_count} hotels into vector store")
        else:
            print(f"⚠️ Loaded {successful_count}/{len(texts)} hotels (some failed due to timeouts)")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error loading hotel data: {e}")
        raise
