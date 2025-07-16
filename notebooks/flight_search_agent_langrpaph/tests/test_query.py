#!/usr/bin/env python3
"""Test script to check Couchbase connection and sample data from travel-sample."""

import os
from datetime import timedelta

import couchbase.auth
import couchbase.cluster
import couchbase.exceptions
import couchbase.options
import dotenv

dotenv.load_dotenv(override=True)


def test_connection_and_data():
    """Test connection to Couchbase and sample data from travel-sample."""
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
        cluster.wait_until_ready(timedelta(seconds=10))
        print("✅ Connected to Couchbase successfully")

        # Test airline data
        print("\n📊 Sample airline data:")
        airline_query = """
        SELECT *
        FROM `travel-sample`.`inventory`.`airline` 
        LIMIT 10
        """

        result = cluster.query(airline_query)
        rows = list(result.rows())
        for i, row in enumerate(rows, 1):
            print(f"{i}. {row}")

        # Test route data
        print("\n🛫 Sample route data:")
        route_query = """
        SELECT *
        FROM `travel-sample`.inventory.route 
        LIMIT 10
        """

        result = cluster.query(route_query)
        rows = list(result.rows())
        for i, row in enumerate(rows, 1):
            print(f"{i}. {row}")

        # Test specific route lookup (SFO to LAX)
        print("\n🔍 Sample flight lookup (SFO to LAX):")
        flight_query = """
        FROM `travel-sample`.inventory.route r
        WHERE r.sourceairport = "SFO" AND r.destinationairport = "LAX"
        SELECT VALUE r.airline || " flight from " || r.sourceairport || " to " || 
                     r.destinationairport || " using " || r.equipment
        LIMIT 10
        """

        result = cluster.query(flight_query)
        rows = list(result.rows())
        if rows:
            for i, row in enumerate(rows, 1):
                print(f"{i}. {row}")
        else:
            print("No flights found from SFO to LAX")

    except couchbase.exceptions.CouchbaseException as e:
        print(f"❌ Couchbase error: {e}")
    except Exception as e:
        print(f"❌ General error: {e}")


if __name__ == "__main__":
    test_connection_and_data()
