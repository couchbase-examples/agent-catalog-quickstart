#!/usr/bin/env python3
"""
Simple Flight Search Agent Evaluation

A basic evaluation script that focuses on agent performance without complex Phoenix evaluations.
This provides a clean baseline for testing the flight search agent.
"""

import json
import os
import sys
import logging
import time
from typing import Dict, List
from datetime import datetime

import agentc
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path to import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import Phoenix for basic session
try:
    import phoenix as px

    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

# Import flight search components
try:
    from main import setup_environment, FlightSearchGraph

    FLIGHT_AGENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Flight search components not available: {e}")
    FLIGHT_AGENT_AVAILABLE = False


class SimpleFlightEvaluator:
    """Simple flight search agent evaluator without complex Phoenix dependencies."""

    def __init__(self):
        self.catalog = None
        self.span = None
        self.agent = None
        self.phoenix_session = None

    def start_phoenix_session(self):
        """Start Phoenix session for observability."""
        if PHOENIX_AVAILABLE:
            try:
                logger.info("🔧 Starting Phoenix session...")
                self.phoenix_session = px.launch_app()
                logger.info(f"🌐 Phoenix UI: {self.phoenix_session.url}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Phoenix session failed: {e}")
                return False
        return False

    def setup_agent(self):
        """Setup agent catalog and flight search graph."""
        try:
            logger.info("🔧 Setting up environment...")
            setup_environment()
            logger.info("✅ Environment setup completed")

            logger.info("🔧 Setting up Agent Catalog...")
            self.catalog = agentc.Catalog()
            self.span = self.catalog.Span(name="FlightSearchEvaluation")
            logger.info("✅ Agent Catalog initialized successfully")

            # Verify tools are available
            tool_names = [
                "lookup_flight_info",
                "save_flight_booking",
                "retrieve_flight_bookings",
                "search_flight_policies",
            ]

            found_tools = []
            for tool_name in tool_names:
                try:
                    tool = self.catalog.find("tool", name=tool_name)
                    if tool:
                        found_tools.append(tool_name)
                except:
                    pass

            logger.info(f"✅ Found {len(found_tools)} tools: {found_tools}")

            # Create flight search graph
            logger.info("🔧 Creating flight search agent...")
            flight_graph = FlightSearchGraph(catalog=self.catalog, span=self.span)
            self.agent = flight_graph.compile()
            logger.info("✅ Flight search agent created")

            return True

        except Exception as e:
            logger.error(f"❌ Error setting up agent: {e}")
            return False

    def run_agent_query(self, query: str, timeout: int = 60) -> Dict:
        """Run a single query through the agent with timeout."""
        try:
            logger.info(f"🔄 Processing query: {query}")

            # Create initial state
            initial_state = FlightSearchGraph.build_starting_state(query=query)

            # Run the agent with timeout
            start_time = time.time()
            result = self.agent.invoke(initial_state)
            elapsed_time = time.time() - start_time

            # Extract response
            response = self._extract_response(result)

            logger.info(f"✅ Response received in {elapsed_time:.2f}s")

            return {
                "query": query,
                "response": response,
                "success": True,
                "error": None,
                "elapsed_time": elapsed_time,
                "response_length": len(response),
            }

        except Exception as e:
            logger.error(f"❌ Error processing query '{query}': {e}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "elapsed_time": 0,
                "response_length": 0,
            }

    def _extract_response(self, result: dict) -> str:
        """Extract response text from agent result."""
        try:
            # Try to get messages
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                elif isinstance(last_message, dict):
                    return last_message.get("content", str(last_message))

            # Try search results
            search_results = result.get("search_results", [])
            if search_results:
                return str(search_results)

            # Fallback
            return str(result)

        except Exception as e:
            return f"Error extracting response: {e}"

    def evaluate_response(self, query: str, response: str) -> Dict:
        """Evaluate response quality with simple metrics."""
        evaluation = {
            "query_type": self._classify_query(query),
            "has_flight_info": self._has_flight_info(response),
            "has_booking_info": self._has_booking_info(response),
            "has_policy_info": self._has_policy_info(response),
            "has_errors": self._has_errors(response),
            "response_length": len(response),
            "completeness_score": 0.0,
        }

        # Calculate simple completeness score
        score = 0.0
        if evaluation["has_flight_info"]:
            score += 30
        if evaluation["has_booking_info"]:
            score += 25
        if evaluation["has_policy_info"]:
            score += 20
        if not evaluation["has_errors"]:
            score += 15
        if 50 <= evaluation["response_length"] <= 1000:
            score += 10

        evaluation["completeness_score"] = min(score, 100.0)
        return evaluation

    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["book", "reserve", "purchase"]):
            return "booking_request"
        elif any(word in query_lower for word in ["policy", "baggage", "cancel", "refund"]):
            return "policy_inquiry"
        elif any(word in query_lower for word in ["find", "search", "show", "options"]):
            return "flight_search"
        elif any(word in query_lower for word in ["my", "current", "existing"]):
            return "booking_retrieval"
        else:
            return "general_inquiry"

    def _has_flight_info(self, response: str) -> bool:
        """Check if response contains flight information."""
        indicators = ["flight", "airline", "departure", "arrival", "aircraft", "gate"]
        return any(indicator in response.lower() for indicator in indicators)

    def _has_booking_info(self, response: str) -> bool:
        """Check if response contains booking information."""
        indicators = ["booking", "reservation", "ticket", "passenger", "confirmation"]
        return any(indicator in response.lower() for indicator in indicators)

    def _has_policy_info(self, response: str) -> bool:
        """Check if response contains policy information."""
        indicators = ["policy", "baggage", "cancellation", "refund", "terms"]
        return any(indicator in response.lower() for indicator in indicators)

    def _has_errors(self, response: str) -> bool:
        """Check if response contains error indicators."""
        error_indicators = ["error", "failed", "could not", "unable to", "exception"]
        return any(indicator in response.lower() for indicator in error_indicators)

    def run_evaluation(self, test_queries: List[str]) -> pd.DataFrame:
        """Run complete evaluation pipeline."""
        logger.info("🚀 Starting Simple Flight Search Agent Evaluation")
        logger.info("=" * 60)

        # Start Phoenix session
        phoenix_started = self.start_phoenix_session()

        # Setup agent
        if not self.setup_agent():
            logger.error("❌ Failed to setup agent")
            return pd.DataFrame()

        # Run evaluation
        results = []
        total_queries = len(test_queries)

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n📝 Query {i}/{total_queries}: {query}")
            logger.info("-" * 40)

            # Run agent query
            result = self.run_agent_query(query)

            # Evaluate response
            evaluation = self.evaluate_response(result["query"], result["response"])

            # Combine results
            combined_result = {**result, **evaluation}
            results.append(combined_result)

            # Log summary
            if result["success"]:
                logger.info(
                    f"✅ Score: {evaluation['completeness_score']:.1f}% | Type: {evaluation['query_type']}"
                )
                logger.info(
                    f"📊 Features: Flight({evaluation['has_flight_info']}) Book({evaluation['has_booking_info']}) Policy({evaluation['has_policy_info']})"
                )
            else:
                logger.info(f"❌ Failed: {result['error']}")

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Print summary
        self._print_summary(results_df, phoenix_started)

        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_agent_simple_evaluation_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"\n💾 Results exported to: {filename}")

        return results_df

    def _print_summary(self, results_df: pd.DataFrame, phoenix_started: bool):
        """Print evaluation summary."""
        logger.info("\n📊 Evaluation Summary")
        logger.info("=" * 60)

        total = len(results_df)
        successful = results_df["success"].sum()
        avg_score = results_df["completeness_score"].mean()
        avg_time = results_df["elapsed_time"].mean()

        logger.info(f"✅ Success Rate: {successful / total * 100:.1f}% ({successful}/{total})")
        logger.info(f"📈 Average Score: {avg_score:.1f}%")
        logger.info(f"⏱️ Average Response Time: {avg_time:.2f}s")

        # Query type breakdown
        query_types = results_df["query_type"].value_counts()
        logger.info(f"\n📋 Query Types:")
        for query_type, count in query_types.items():
            avg_score = results_df[results_df["query_type"] == query_type][
                "completeness_score"
            ].mean()
            logger.info(f"  {query_type}: {count} queries (avg: {avg_score:.1f}%)")

        # Feature detection
        features = ["has_flight_info", "has_booking_info", "has_policy_info"]
        logger.info(f"\n📊 Feature Detection:")
        for feature in features:
            count = results_df[feature].sum()
            logger.info(
                f"  {feature.replace('has_', '').replace('_', ' ').title()}: {count}/{total}"
            )

        # Phoenix info
        if phoenix_started:
            logger.info(f"\n🌐 Phoenix UI: {self.phoenix_session.url}")
        else:
            logger.info(f"\n⚠️ Phoenix not available - basic evaluation only")


def main():
    """Main evaluation function."""
    # Test queries covering different flight search scenarios
    test_queries = [
        "Find flights from JFK to LAX",
        "What is the baggage policy for carry-on items?",
        "Book a flight from SFO to ORD tomorrow for 2 passengers",
        "Show me my current flight bookings",
        "What are the cancellation fees for domestic flights?",
        "Find the cheapest flight from Miami to Atlanta tomorrow",
    ]

    # Run evaluation
    evaluator = SimpleFlightEvaluator()
    results = evaluator.run_evaluation(test_queries)

    logger.info("\n✅ Simple evaluation complete!")
    return results


if __name__ == "__main__":
    main()
