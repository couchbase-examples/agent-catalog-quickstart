#!/usr/bin/env python3
"""
Arize AI Integration for Flight Search Agent Evaluation

This module provides comprehensive evaluation capabilities using Arize AI observability
platform for the flight search agent. It demonstrates how to:

1. Set up Arize observability for LangGraph-based agents
2. Create and manage evaluation datasets for flight search scenarios
3. Run automated evaluations with LLM-as-a-judge for response quality
4. Track performance metrics and traces for flight booking systems
5. Monitor tool usage and booking effectiveness

The implementation integrates with the existing Agent Catalog infrastructure
while extending it with Arize AI capabilities for production monitoring.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import agentc
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose logs
for module in ["httpx", "opentelemetry", "phoenix", "openai", "langchain", "agentc_core"]:
    logging.getLogger(module).setLevel(logging.WARNING)

# Configuration constants
SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
PROJECT_NAME = "flight-search-agent-evaluation"

# Add parent directory to path to import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import Arize dependencies with fallback
try:
    import phoenix as px
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.evals import (
        QA_PROMPT_RAILS_MAP,
        QA_PROMPT_TEMPLATE,
        RAG_RELEVANCY_PROMPT_RAILS_MAP,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        OpenAIModel,
        llm_classify,
    )
    from phoenix.otel import register

    ARIZE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Arize dependencies not available: {e}")
    logger.warning("Running in local evaluation mode only...")
    ARIZE_AVAILABLE = False

# Import flight search components
try:
    from main import FlightSearchGraph, setup_environment

    FLIGHT_AGENT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Flight search components not available: {e}")
    FLIGHT_AGENT_AVAILABLE = False


class ArizeFlightSearchEvaluator:
    """
    Comprehensive evaluation system for flight search agents using Arize AI.

    This class provides:
    - Flight search performance evaluation with multiple metrics
    - Tool effectiveness monitoring (lookup, booking, retrieval, policies)
    - Response quality assessment and booking accuracy tracking
    - Comparative analysis of different flight search strategies
    - Arize AI platform integration for production monitoring
    """

    def __init__(self):
        """Initialize the Arize evaluator with Agent Catalog integration."""
        self.catalog = None
        self.span = None
        self.agent = None
        self.arize_client = None
        self.dataset_id = None
        self.tracer_provider = None
        self.phoenix_session = None

        # Initialize Arize observability if available
        if ARIZE_AVAILABLE:
            self._setup_arize_observability()

            # Initialize evaluation models
            self.evaluator_llm = OpenAIModel(model="gpt-4o")

            # Define evaluation rails
            self.relevance_rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
            self.qa_rails = list(QA_PROMPT_RAILS_MAP.values())
        else:
            logger.warning("⚠️ Arize not available - running basic evaluation only")

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            logger.info("🔧 Setting up Arize observability...")

            # Start Phoenix session
            self.phoenix_session = px.launch_app()
            logger.info(f"🌐 Phoenix UI: {self.phoenix_session.url}")

            # Register Phoenix OTEL and get tracer provider
            self.tracer_provider = register(
                project_name=PROJECT_NAME,
                endpoint="http://localhost:6006/v1/traces",
            )
            logger.info("✅ Phoenix OTEL registered successfully")

            # Instrument LangChain and OpenAI with new approach
            instrumentors = [
                ("LangChain", LangChainInstrumentor),
                ("OpenAI", OpenAIInstrumentor),
            ]

            for name, instrumentor_class in instrumentors:
                try:
                    instrumentor = instrumentor_class()
                    if not instrumentor.is_instrumented_by_opentelemetry:
                        instrumentor.instrument(tracer_provider=self.tracer_provider)
                        logger.info(f"✅ {name} instrumented successfully")
                    else:
                        logger.info(f"ℹ️ {name} already instrumented, skipping")
                except Exception as e:
                    logger.warning(f"⚠️ {name} instrumentation failed: {e}")
                    continue

            # Initialize Arize datasets client if credentials available
            if API_KEY != "your-api-key" and SPACE_ID != "your-space-id":
                try:
                    self.arize_client = ArizeDatasetsClient(
                        developer_key=API_KEY, api_key=API_KEY, space_id=SPACE_ID
                    )
                    logger.info("✅ Arize datasets client initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize Arize datasets client: {e}")
                    self.arize_client = None
            else:
                logger.info("ℹ️ Arize credentials not configured - using local evaluation only")

            logger.info("✅ Arize observability configured successfully")

        except Exception as e:
            logger.error(f"⚠️ Error setting up Arize observability: {e}")

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

            # Count available tools by trying to find the known tools
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

    def run_agent_query(self, query: str) -> Dict:
        """Run a single query through the agent."""
        try:
            logger.info(f"🔄 Processing query: {query}")

            # Create initial state
            initial_state = FlightSearchGraph.build_starting_state(query=query)

            # Run the agent
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
            }

        except Exception as e:
            logger.error(f"❌ Error processing query '{query}': {e}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
                "elapsed_time": 0,
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

    def analyze_response(self, query: str, response: str) -> Dict:
        """Analyze response quality with basic metrics."""
        analysis = {
            "query_type": self._classify_query_type(query),
            "has_flight_info": self._has_flight_info(response),
            "has_booking_info": self._has_booking_info(response),
            "has_policy_info": self._has_policy_info(response),
            "response_length": len(response),
            "quality_score": 0.0,
        }

        # Calculate quality score
        score = 0.0
        if analysis["has_flight_info"]:
            score += 3.0
        if analysis["has_booking_info"]:
            score += 2.0
        if analysis["has_policy_info"]:
            score += 1.0
        if 50 <= analysis["response_length"] <= 1000:
            score += 2.0
        if not self._has_error_indicators(response):
            score += 2.0

        analysis["quality_score"] = min(score, 10.0)
        return analysis

    def _classify_query_type(self, query: str) -> str:
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

    def _has_error_indicators(self, response: str) -> bool:
        """Check if response has error indicators."""
        error_indicators = ["error", "failed", "could not", "unable to", "exception"]
        return any(indicator in response.lower() for indicator in error_indicators)

    def run_arize_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Arize-based LLM evaluations on the results.

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            DataFrame with additional Arize evaluation columns
        """
        if not ARIZE_AVAILABLE:
            logger.warning("⚠️ Arize not available - skipping LLM evaluations")
            return results_df

        logger.info(f"🧠 Running Arize LLM evaluations on {len(results_df)} responses...")
        logger.info("   📋 Evaluation criteria:")
        logger.info("      🔍 Relevance: Does the response address the flight search query?")
        logger.info("      🎯 Correctness: Is the flight information accurate and helpful?")

        # Sample evaluation data preview
        if len(results_df) > 0:
            sample_row = results_df.iloc[0]
            logger.info(f"\n   🔍 Sample evaluation data:")
            logger.info(f"      Query: {sample_row['query']}")
            logger.info(f"      Response: {sample_row['response'][:100]}...")

        try:
            # Prepare data for evaluation with correct column names
            evaluation_data = []
            for _, row in results_df.iterrows():
                evaluation_data.append(
                    {
                        "input": row["query"],
                        "output": row["response"],
                        "reference": f"A helpful response about {row['query_type'].replace('_', ' ')}",
                        "context": row["response"],  # Use response as context for relevance
                    }
                )

            eval_df = pd.DataFrame(evaluation_data)

            # Run relevance evaluation
            logger.info(f"\n   🔍 Evaluating relevance...")
            try:
                relevance_results = llm_classify(
                    dataframe=eval_df[["input", "context"]],
                    model=self.evaluator_llm,
                    template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                    rails=self.relevance_rails,
                    provide_explanation=True,
                )

                # Add relevance results
                if hasattr(relevance_results, "columns") and "label" in relevance_results.columns:
                    results_df["arize_relevance"] = relevance_results["label"]
                    results_df["arize_relevance_explanation"] = relevance_results.get(
                        "explanation", ""
                    )
                else:
                    results_df["arize_relevance"] = "unknown"
                    results_df["arize_relevance_explanation"] = "Evaluation failed"

            except Exception as e:
                logger.warning(f"   ⚠️ Relevance evaluation failed: {e}")
                results_df["arize_relevance"] = "unknown"
                results_df["arize_relevance_explanation"] = f"Error: {e}"

            # Run correctness evaluation
            logger.info(f"   🎯 Evaluating correctness...")
            try:
                correctness_results = llm_classify(
                    dataframe=eval_df[["input", "output", "reference"]],
                    model=self.evaluator_llm,
                    template=QA_PROMPT_TEMPLATE,
                    rails=self.qa_rails,
                    provide_explanation=True,
                )

                # Add correctness results
                if (
                    hasattr(correctness_results, "columns")
                    and "label" in correctness_results.columns
                ):
                    results_df["arize_correctness"] = correctness_results["label"]
                    results_df["arize_correctness_explanation"] = correctness_results.get(
                        "explanation", ""
                    )
                else:
                    results_df["arize_correctness"] = "unknown"
                    results_df["arize_correctness_explanation"] = "Evaluation failed"

            except Exception as e:
                logger.warning(f"   ⚠️ Correctness evaluation failed: {e}")
                results_df["arize_correctness"] = "unknown"
                results_df["arize_correctness_explanation"] = f"Error: {e}"

            # Display sample explanations
            if len(results_df) > 0:
                logger.info(f"\n   📝 Sample evaluation explanations:")
                for i in range(min(2, len(results_df))):
                    row = results_df.iloc[i]
                    logger.info(f"      Query: {row['query']}")
                    if "arize_relevance_explanation" in row:
                        logger.info(
                            f"      Relevance ({row['arize_relevance']}): {str(row['arize_relevance_explanation'])[:100]}..."
                        )
                    if "arize_correctness_explanation" in row:
                        logger.info(
                            f"      Correctness ({row['arize_correctness']}): {str(row['arize_correctness_explanation'])[:100]}..."
                        )
                    logger.info("")

            logger.info(f"   ✅ Arize LLM evaluations completed")

        except Exception as e:
            logger.error(f"   ❌ Error running Arize LLM evaluations: {e}")
            # Add default values if evaluation fails
            results_df["arize_relevance"] = "unknown"
            results_df["arize_relevance_explanation"] = f"Error: {e}"
            results_df["arize_correctness"] = "unknown"
            results_df["arize_correctness_explanation"] = f"Error: {e}"

        return results_df

    def create_arize_dataset(self, results_df: pd.DataFrame) -> str:
        """Create an Arize dataset from evaluation results."""
        if not self.arize_client:
            logger.warning("⚠️ Arize client not available - skipping dataset creation")
            return None

        try:
            logger.info("📊 Creating Arize dataset...")

            # Prepare dataset
            dataset_name = f"flight-search-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Convert results to Arize format
            dataset_data = []
            for _, row in results_df.iterrows():
                dataset_data.append(
                    {
                        "input": row["query"],
                        "output": row["response"],
                        "query_type": row["query_type"],
                        "success": row["success"],
                        "quality_score": row["quality_score"],
                        "arize_relevance": row.get("arize_relevance", "unknown"),
                        "arize_correctness": row.get("arize_correctness", "unknown"),
                    }
                )

            # Create dataset
            dataset = self.arize_client.create_dataset(
                dataset_name=dataset_name, dataset_type=GENERATIVE, data=pd.DataFrame(dataset_data)
            )

            logger.info(f"✅ Arize dataset created: {dataset_name}")
            return dataset.id

        except Exception as e:
            logger.error(f"❌ Error creating Arize dataset: {e}")
            return None

    def run_evaluation(self, test_queries: List[str]) -> pd.DataFrame:
        """Run complete evaluation pipeline with Arize AI integration."""
        logger.info("🚀 Starting Arize Flight Search Agent Evaluation...")
        logger.info("=" * 70)

        # Setup agent
        if not self.setup_agent():
            logger.error("❌ Failed to setup agent")
            return pd.DataFrame()

        # Run queries
        results = []
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n📝 Query {i}/{len(test_queries)}: {query}")
            logger.info("-" * 50)

            # Run agent query
            result = self.run_agent_query(query)

            # Analyze response
            analysis = self.analyze_response(result["query"], result["response"])

            # Combine results
            combined_result = {**result, **analysis}
            results.append(combined_result)

            # Log results
            if result["success"]:
                logger.info(
                    f"✅ Score: {analysis['quality_score']:.1f}/10.0 | Type: {analysis['query_type']}"
                )
            else:
                logger.info(f"❌ Error: {result['error']}")

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Run Arize evaluations if available
        if ARIZE_AVAILABLE:
            results_df = self.run_arize_evaluations(results_df)

        # Create Arize dataset
        dataset_id = self.create_arize_dataset(results_df)

        # Print summary
        self._print_summary(results_df, dataset_id)

        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flight_agent_arize_evaluation_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"\n💾 Results exported to '{filename}'")

        if self.phoenix_session:
            logger.info(f"\n🌐 Phoenix UI: {self.phoenix_session.url}")

        return results_df

    def _print_summary(self, results_df: pd.DataFrame, dataset_id: str):
        """Print evaluation summary."""
        logger.info("\n📊 Arize Evaluation Summary")
        logger.info("=" * 70)

        total = len(results_df)
        successful = results_df["success"].sum()
        avg_quality = results_df["quality_score"].mean()

        logger.info(
            f"✅ Success Rate: {successful / total * 100:.1f}% ({successful}/{total} queries)"
        )
        logger.info(f"📈 Average Quality Score: {avg_quality:.1f}/10.0")

        # Arize results if available
        if "arize_relevance" in results_df.columns:
            relevance_counts = results_df["arize_relevance"].value_counts()
            correctness_counts = results_df["arize_correctness"].value_counts()

            logger.info(f"\n🔍 Arize LLM Evaluation Results:")
            logger.info(f"   📋 Relevance: {dict(relevance_counts)}")
            logger.info(f"   ✅ Correctness: {dict(correctness_counts)}")

        # Dataset info
        if dataset_id:
            logger.info(f"\n📊 Arize Dataset: {dataset_id}")

        # Query type breakdown
        query_types = results_df["query_type"].value_counts()
        logger.info(f"\n📋 Query Type Breakdown:")
        for query_type, count in query_types.items():
            avg_score = results_df[results_df["query_type"] == query_type]["quality_score"].mean()
            logger.info(f"  {query_type}: {count} queries (avg: {avg_score:.1f}/10.0)")

    def cleanup(self):
        """Clean up Arize session."""
        try:
            logger.info("🧹 Cleaning up Arize session...")
            # Phoenix sessions cleanup automatically
            logger.info("✅ Cleanup complete")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")


def main():
    """Main evaluation function with Arize AI integration."""
    # Test queries covering different flight search scenarios
    test_queries = [
        "Find flights from JFK to LAX",
        "What is the baggage policy for carry-on items?",
        "Book a flight from SFO to ORD tomorrow for 2 passengers in business class",
        "Show me my current flight bookings",
        "What are the cancellation fees for domestic flights?",
        "Find flights from Miami to Atlanta and book the cheapest option for tomorrow",
    ]

    # Run evaluation
    evaluator = ArizeFlightSearchEvaluator()
    results = evaluator.run_evaluation(test_queries)

    logger.info("\n✅ Arize evaluation complete!")

    # Cleanup
    evaluator.cleanup()

    return results


if __name__ == "__main__":
    main()
