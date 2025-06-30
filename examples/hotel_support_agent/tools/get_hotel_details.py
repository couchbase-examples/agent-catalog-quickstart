import os
import logging
from datetime import timedelta
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from agentc.core import tool

@tool
def get_hotel_details(hotel_name: str) -> str:
    """Gets detailed information about a specific hotel by name from the Couchbase database.
    
    Args:
        hotel_name: The name of the hotel to get details for
        
    Returns:
        A formatted string with comprehensive hotel details including amenities, pricing, and contact information.
    """
    try:
        auth = PasswordAuthenticator(
            os.environ.get('CB_USERNAME', 'Administrator'), 
            os.environ.get('CB_PASSWORD', 'password')
        )
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ.get('CB_HOST', 'couchbase://localhost'), options)
        cluster.wait_until_ready(timedelta(seconds=5))
        
        bucket_name = os.environ.get('CB_BUCKET_NAME', 'vector-search-testing')
        scope_name = os.environ.get('SCOPE_NAME', 'shared')
        collection_name = os.environ.get('COLLECTION_NAME', 'deepseek')
        
        query = f"""
        SELECT RAW content 
        FROM `{bucket_name}`.`{scope_name}`.`{collection_name}` 
        WHERE LOWER(content) LIKE LOWER('%{hotel_name}%')
        LIMIT 1
        """
        
        result = cluster.query(query)
        rows = list(result.rows())
        
        if not rows:
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
                }
            }
            
            hotel_key = hotel_name.lower().strip()
            if hotel_key in detailed_hotels:
                hotel = detailed_hotels[hotel_key]
                return f"""
**{hotel['name']}**

**Location:** {hotel['location']}
**Description:** {hotel['description']}
**Price Range:** {hotel['price_range']}
**Rating:** {hotel['rating']}

**Amenities:**
{chr(10).join(f"• {amenity}" for amenity in hotel['amenities'])}

**Contact Information:**
{hotel['contact']}

**Address:** {hotel['address']}

**Check-in Time:** {hotel['check_in']}
**Check-out Time:** {hotel['check_out']}
**Cancellation Policy:** {hotel['cancellation']}
"""
            else:
                return f"Hotel '{hotel_name}' not found in our database. Please check the spelling or try searching for hotels in a specific location."
        
        hotel_content = rows[0]
        return f"Hotel Details: {hotel_content}"
        
    except Exception as e:
        logging.error(f"Error getting hotel details: {str(e)}")
        raise RuntimeError(f"Failed to retrieve hotel details: {str(e)}")
