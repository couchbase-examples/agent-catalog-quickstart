import os
from datetime import timedelta

import agentc
import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

dotenv.load_dotenv(override=True)

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
    query = """
    FROM
        `travel-sample`.inventory.route r
    WHERE
        r.sourceairport = $source_airport AND
        r.destinationairport = $destination_airport
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
        raise RuntimeError(error_msg) from e

    if not rows:
        return [f"No flights found from {source_airport} to {destination_airport}"]

    return rows