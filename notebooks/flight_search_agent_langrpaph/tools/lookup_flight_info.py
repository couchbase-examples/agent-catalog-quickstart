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
try:
    cluster = couchbase.cluster.Cluster(
        os.getenv("CB_CONN_STRING", "couchbase://localhost"),
        couchbase.options.ClusterOptions(
            authenticator=couchbase.auth.PasswordAuthenticator(
                username=os.getenv("CB_USERNAME", "Administrator"),
                password=os.getenv("CB_PASSWORD", "password")
            )
        ),
    )
    cluster.wait_until_ready(timedelta(seconds=5))
except couchbase.exceptions.CouchbaseException as e:
    error_msg = f"Could not connect to Couchbase cluster: {e!s}"
    print(error_msg)


@agentc.catalog.tool
def lookup_flight_info(source_airport: str, destination_airport: str) -> list[str]:
    """Find flight routes between two airports with airline and aircraft information."""
    
    # Validate input parameters
    if not source_airport or not destination_airport:
        return ["Error: Both source and destination airports are required."]
    
    # Ensure airport codes are uppercase and 3 characters
    source_airport = source_airport.upper().strip()
    destination_airport = destination_airport.upper().strip()
    
    if len(source_airport) != 3 or len(destination_airport) != 3:
        return [f"Error: Airport codes must be 3 letters (e.g., JFK, LAX). Got: {source_airport}, {destination_airport}"]
    
    # Updated query with null handling
    query = """
    FROM
        `travel-sample`.inventory.route r
    WHERE
        r.sourceairport = $source_airport AND
        r.destinationairport = $destination_airport AND
        r.airline IS NOT NULL AND
        r.equipment IS NOT NULL
    SELECT VALUE
        r.airline || " flight from " || r.sourceairport || " to " ||
        r.destinationairport || " using " || r.equipment
    LIMIT
        10
    """

    try:
        result = cluster.query(
            query, source_airport=source_airport, destination_airport=destination_airport
        )
        rows = list(result.rows())
    except couchbase.exceptions.CouchbaseException as e:
        error_msg = f"Failed to lookup flight information: {e!s}"
        logger.error(error_msg)
        return [f"Database error: Unable to search flights. Please try again later."]

    if not rows:
        return [f"No flights found from {source_airport} to {destination_airport}. Please check airport codes or try different dates."]

    return rows