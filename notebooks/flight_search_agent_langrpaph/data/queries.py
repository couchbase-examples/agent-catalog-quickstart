"""
Airline Review Queries for Flight Search Agent

This module contains the test queries for the flight search agent,
replacing the original flight policy queries with airline review queries.
"""

# Test queries for flight search agent evaluation
TEST_QUERIES = [
    "Find flights from JFK to LAX",
    "What do passengers say about JAL's service quality?",
    "Book a flight from SFO to ORD tomorrow for 2 passengers in business class", 
    "Show me my current flight bookings",
    "What do airline reviews say about JAL's baggage handling?",
]

def get_test_queries():
    """Return test queries for evaluation."""
    return TEST_QUERIES

def get_evaluation_queries():
    """Return evaluation queries (same as test queries)."""
    return TEST_QUERIES

def get_flight_policy_queries():
    """Return flight policy queries (for backward compatibility)."""
    return TEST_QUERIES 