"""
Quick script to find available routes in travel-sample database
"""

from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to Couchbase
cluster = Cluster(
    os.getenv("CB_CONN_STRING", "couchbase://127.0.0.1"),
    ClusterOptions(
        authenticator=PasswordAuthenticator(
            username=os.getenv("CB_USERNAME", "kaustav"),
            password=os.getenv("CB_PASSWORD", "password"),
        )
    ),
)

bucket = cluster.bucket("travel-sample")
scope = bucket.scope("inventory")

# Query for available routes
query = """
SELECT DISTINCT r.sourceairport, r.destinationairport, a1.airportname AS source_name, a2.airportname AS dest_name
FROM `travel-sample`.inventory.route r
JOIN `travel-sample`.inventory.airport a1 ON r.sourceairport = a1.faa
JOIN `travel-sample`.inventory.airport a2 ON r.destinationairport = a2.faa
WHERE r.sourceairport IS NOT NULL AND r.destinationairport IS NOT NULL
LIMIT 10
"""

try:
    result = cluster.query(query)
    print("Available routes in travel-sample:")
    print("=" * 50)
    for row in result:
        print(
            f"{row['sourceairport']} → {row['destinationairport']}: {row['source_name']} to {row['dest_name']}"
        )
except Exception as e:
    print(f"Error querying routes: {e}")
