import logging
import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

dotenv.load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Agent Catalog imports this file once. To share Couchbase connections, use a global variable.
cluster = None
try:
    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password")
    )
    options = couchbase.options.ClusterOptions(auth)
    
    # Use WAN profile for better timeout handling with remote clusters
    options.apply_profile("wan_development")
    
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        options
    )
    cluster.wait_until_ready(timedelta(seconds=15))
except couchbase.exceptions.CouchbaseException as e:
    logger.error(f"Could not connect to Couchbase cluster: {e!s}")
    cluster = None


@agentc.catalog.tool  
def lookup_flight_info(source_airport: str, destination_airport: str) -> str:
    """Find flight routes between two airports with airline and aircraft information.
    
    Args:
        source_airport: 3-letter source airport code (e.g., JFK)
        destination_airport: 3-letter destination airport code (e.g., LAX)
    
    Returns:
        Formatted string with available flights
    """
    try:
        # Validate database connection
        if cluster is None:
            return "Database connection unavailable. Please try again later."
        
        # Validate input parameters
        if not source_airport or not destination_airport:
            return "Error: Both source and destination airports are required."

        # Clean and validate airport codes
        source_airport = source_airport.upper().strip()
        destination_airport = destination_airport.upper().strip()

        if len(source_airport) != 3 or len(destination_airport) != 3:
            return f"Error: Airport codes must be 3 letters (e.g., JFK, LAX). Got: {source_airport}, {destination_airport}"

        if not source_airport.isalpha() or not destination_airport.isalpha():
            return f"Error: Airport codes must be letters only. Got: {source_airport}, {destination_airport}"

        # Clean, simple query
        query = """
        SELECT VALUE r.airline || " flight from " || r.sourceairport || " to " ||
                     r.destinationairport || " using " || r.equipment
        FROM `travel-sample`.inventory.route r
        WHERE r.sourceairport = $source_airport 
        AND r.destinationairport = $destination_airport
        AND r.airline IS NOT NULL 
        AND r.equipment IS NOT NULL
        LIMIT 10
        """

        result = cluster.query(query, source_airport=source_airport, destination_airport=destination_airport)
        flights = list(result.rows())

        if not flights:
            return f"No flights found from {source_airport} to {destination_airport}. Please check airport codes."

        # Format results nicely
        response = f"Available flights from {source_airport} to {destination_airport}:\n\n"
        for i, flight in enumerate(flights, 1):
            response += f"{i}. {flight}\n"
        
        return response.strip()

    except couchbase.exceptions.CouchbaseException as e:
        logger.exception(f"Database error: {e}")
        return "Database error: Unable to search flights. Please try again later."
    except Exception as e:
        logger.exception(f"Error looking up flights: {e}")
        return f"Error: Could not process flight lookup. Please check your input format."
