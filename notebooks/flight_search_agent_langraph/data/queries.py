"""
Airline Review Queries for Flight Search Agent

This module contains the test queries for the flight search agent,
updated to use Indian Airlines dataset context.
"""

# Test queries for flight search agent evaluation
# Updated to match the Indian Airlines dataset context
TEST_QUERIES = [
    "Find flights from JFK to LAX",
    "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
    "Book an economy flight from JFK to MIA for next week, 1 passenger",
    "Show me my current flight bookings",
    "What do passengers say about IndiGo's service quality?",
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
