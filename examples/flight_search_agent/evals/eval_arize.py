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

import json
import os
import pathlib
import sys
import unittest.mock
import logging
import time
from typing import Dict, List
from uuid import uuid4

import agentc
import pandas as pd

# Configure logging to reduce verbosity
logging.basicConfig(level=logging.INFO)
# Suppress verbose HTTP logs and embeddings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("opentelemetry").setLevel(logging.WARNING)
logging.getLogger("phoenix").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("agentc_core").setLevel(logging.WARNING)

# Configuration constants
SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
PROJECT_NAME = "flight-search-agent-evaluation"

# Import the flight search agent components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import (
    setup_environment,
    CouchbaseClient,
    FlightSearchAgent,
    FlightSearchGraph,
    FlightSearchState,
)

# Import necessary dependencies for CouchbaseSetup
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from couchbase.exceptions import CouchbaseException
from datetime import timedelta
import time

# Try to import Arize dependencies with fallback
try:
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.experiments.types import (
        ExperimentTaskResultColumnNames,
        EvaluationResultColumnNames,
    )
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    from phoenix.evals import (
        QA_PROMPT_RAILS_MAP,
        QA_PROMPT_TEMPLATE,
        RAG_RELEVANCY_PROMPT_RAILS_MAP,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        OpenAIModel,
        llm_classify,
    )

    ARIZE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Arize dependencies not available: {e}")
    print("   Running in local evaluation mode only...")
    ARIZE_AVAILABLE = False


class CouchbaseSetup:
    """Handle Couchbase cluster setup and configuration for flight search evaluation."""

    def __init__(self):
        self.cluster = None
        self.collection = None
        self.logger = logging.getLogger(__name__)

    def setup_environment(self):
        """Setup required environment variables."""
        required_vars = [
            "OPENAI_API_KEY",
            "CB_CONN_STRING",
            "CB_USERNAME",
            "CB_PASSWORD",
            "CB_BUCKET",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            # Set defaults for missing variables
            defaults = {
                "CB_CONN_STRING": "couchbase://localhost",
                "CB_USERNAME": "Administrator",
                "CB_PASSWORD": "password",
                "CB_BUCKET": "vector-search-testing",
            }

            for var in missing_vars:
                if var in defaults:
                    os.environ[var] = defaults[var]
                    self.logger.info(f"Set default for {var}: {defaults[var]}")

        # Set defaults for optional variables
        if not os.getenv("INDEX_NAME"):
            os.environ["INDEX_NAME"] = "vector_search_agentcatalog"
        if not os.getenv("SCOPE_NAME"):
            os.environ["SCOPE_NAME"] = "shared"
        if not os.getenv("COLLECTION_NAME"):
            os.environ["COLLECTION_NAME"] = "agentcatalog"

        self.logger.info(f"Using Couchbase connection: {os.getenv('CB_CONN_STRING')}")
        self.logger.info(f"Using bucket: {os.getenv('CB_BUCKET')}")
        self.logger.info(f"Using scope: {os.getenv('SCOPE_NAME')}")
        self.logger.info(f"Using collection: {os.getenv('COLLECTION_NAME')}")
        self.logger.info(f"Using index: {os.getenv('INDEX_NAME')}")

    def setup_couchbase_connection(self):
        """Setup Couchbase cluster connection."""
        try:
            # Extract host from connection string
            conn_string = os.environ["CB_CONN_STRING"]
            if "://" in conn_string:
                host = conn_string
            else:
                host = f"couchbase://{conn_string}"

            auth = PasswordAuthenticator(os.environ["CB_USERNAME"], os.environ["CB_PASSWORD"])
            options = ClusterOptions(auth)
            self.cluster = Cluster(host, options)
            self.cluster.wait_until_ready(timedelta(seconds=10))
            self.logger.info("Successfully connected to Couchbase cluster")
            return self.cluster
        except CouchbaseException as e:
            raise ConnectionError(f"Failed to connect to Couchbase: {e}")

    def setup_bucket_scope_collection(self):
        """Setup bucket, scope, and collection."""
        try:
            bucket_name = os.environ["CB_BUCKET"]
            scope_name = os.environ["SCOPE_NAME"]
            collection_name = os.environ["COLLECTION_NAME"]

            # Check bucket
            try:
                bucket = self.cluster.bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' exists")
            except Exception:
                self.logger.info(f"Creating bucket '{bucket_name}'...")
                bucket_settings = CreateBucketSettings(
                    name=bucket_name,
                    bucket_type="couchbase",
                    ram_quota_mb=1024,
                    flush_enabled=True,
                    num_replicas=0,
                )
                self.cluster.buckets().create_bucket(bucket_settings)
                time.sleep(5)
                bucket = self.cluster.bucket(bucket_name)
                self.logger.info(f"Bucket '{bucket_name}' created successfully")

            # Setup scope and collection
            bucket_manager = bucket.collections()

            scopes = bucket_manager.get_all_scopes()
            scope_exists = any(scope.name == scope_name for scope in scopes)

            if not scope_exists and scope_name != "_default":
                self.logger.info(f"Creating scope '{scope_name}'...")
                bucket_manager.create_scope(scope_name)
                self.logger.info(f"Scope '{scope_name}' created successfully")

            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name
                and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )

            if not collection_exists:
                self.logger.info(f"Creating collection '{collection_name}'...")
                bucket_manager.create_collection(scope_name, collection_name)
                self.logger.info(f"Collection '{collection_name}' created successfully")

            self.collection = bucket.scope(scope_name).collection(collection_name)
            time.sleep(3)

            # Create primary index
            try:
                self.cluster.query(
                    f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`"
                ).execute()
                self.logger.info("Primary index created successfully")
            except Exception as e:
                self.logger.warning(f"Error creating primary index: {e}")

            return self.collection

        except Exception as e:
            raise RuntimeError(f"Error setting up bucket/scope/collection: {e}")

    def setup_vector_search_index(self):
        """Setup vector search index."""
        try:
            # Load index definition from agentcatalog_index.json
            index_file = "agentcatalog_index.json"

            # Look for the index file in the current directory
            if not os.path.exists(index_file):
                # Try looking in the parent directory
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                index_file = os.path.join(parent_dir, "agentcatalog_index.json")

            if os.path.exists(index_file):
                with open(index_file, "r") as f:
                    index_definition = json.load(f)

                scope_index_manager = (
                    self.cluster.bucket(os.environ["CB_BUCKET"])
                    .scope(os.environ["SCOPE_NAME"])
                    .search_indexes()
                )

                existing_indexes = scope_index_manager.get_all_indexes()
                index_name = index_definition["name"]

                if index_name not in [index.name for index in existing_indexes]:
                    self.logger.info(f"Creating vector search index '{index_name}'...")
                    search_index = SearchIndex.from_json(index_definition)
                    scope_index_manager.upsert_index(search_index)
                    self.logger.info(f"Vector search index '{index_name}' created successfully")
                else:
                    self.logger.info(f"Vector search index '{index_name}' already exists")
            else:
                self.logger.warning(f"Index definition file {index_file} not found")

        except Exception as e:
            self.logger.error(f"Error setting up vector search index: {e}")


class ArizeFlightSearchEvaluator:
    """
    Comprehensive evaluation system for flight search agents using Arize AI.

    This class provides:
    - Flight search performance evaluation with multiple metrics
    - Tool effectiveness monitoring (lookup, booking, retrieval, policies)
    - Response quality assessment and booking accuracy tracking
    - Comparative analysis of different flight search strategies
    """

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the Arize evaluator with Agent Catalog integration."""
        self.catalog = catalog
        self.span = span
        self.arize_client = None
        self.dataset_id = None
        self.tracer_provider = None

        # Initialize Arize observability if available
        if ARIZE_AVAILABLE:
            self._setup_arize_observability()

            # Initialize evaluation models
            self.evaluator_llm = OpenAIModel(model="gpt-4o")

            # Define evaluation rails
            self.relevance_rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
            self.qa_rails = list(QA_PROMPT_RAILS_MAP.values())
        else:
            print("⚠️ Arize not available - running basic evaluation only")

    def setup_couchbase_infrastructure(self):
        """Setup Couchbase infrastructure for evaluation."""
        logger = logging.getLogger(__name__)

        try:
            couchbase_setup = CouchbaseSetup()
            couchbase_setup.setup_environment()
            cluster = couchbase_setup.setup_couchbase_connection()
            collection = couchbase_setup.setup_bucket_scope_collection()
            couchbase_setup.setup_vector_search_index()

            # Create Couchbase client for compatibility with existing code
            couchbase_client = CouchbaseClient(
                conn_string=os.environ["CB_CONN_STRING"],
                username=os.environ["CB_USERNAME"],
                password=os.environ["CB_PASSWORD"],
                bucket_name=os.environ["CB_BUCKET"],
            )

            logger.info("Couchbase infrastructure setup complete for evaluation")
            return couchbase_client
        except Exception as e:
            logger.error(f"Error setting up Couchbase infrastructure: {e}")
            raise

    def create_agent(self, catalog: agentc.Catalog, span: agentc.Span):
        """Create a flight search agent for evaluation."""
        logger = logging.getLogger(__name__)

        try:
            # Create the flight search graph
            flight_graph = FlightSearchGraph(catalog=catalog, span=span)

            # Compile the graph
            compiled_graph = flight_graph.compile()

            logger.info("Flight search agent created for evaluation")
            return compiled_graph
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            # Setup tracer provider for Phoenix (local only)
            self.tracer_provider = TracerProvider()
            trace.set_tracer_provider(self.tracer_provider)

            print("✅ Local tracing configured successfully")

            # Instrument LangChain and OpenAI
            instrumentors = [
                ("LangChain", LangChainInstrumentor),
                ("OpenAI", OpenAIInstrumentor),
            ]

            for name, instrumentor_class in instrumentors:
                try:
                    instrumentor = instrumentor_class()
                    if not instrumentor.is_instrumented_by_opentelemetry:
                        instrumentor.instrument(tracer_provider=self.tracer_provider)
                        print(f"✅ {name} instrumented successfully")
                    else:
                        print(f"ℹ️ {name} already instrumented, skipping")
                except Exception as e:
                    print(f"⚠️ {name} instrumentation failed: {e}")
                    continue

            # Initialize Arize datasets client
            if API_KEY != "your-api-key":
                try:
                    self.arize_client = ArizeDatasetsClient(developer_key=API_KEY, api_key=API_KEY)
                    print("✅ Arize datasets client initialized")
                except Exception as e:
                    print(f"⚠️ Could not initialize Arize datasets client: {e}")
                    self.arize_client = None

            print("✅ Arize observability configured successfully")

        except Exception as e:
            print(f"⚠️ Error setting up Arize observability: {e}")

    def run_flight_search_evaluation(self, test_inputs: List[str]) -> pd.DataFrame:
        """
        Run flight search evaluation on a set of test inputs.

        Args:
            test_inputs: List of flight search queries to evaluate

        Returns:
            DataFrame with evaluation results
        """
        print(f"🚀 Running evaluation on {len(test_inputs)} queries...")

        # Setup infrastructure
        print("🔧 Setting up Couchbase infrastructure...")
        couchbase_client = self.setup_couchbase_infrastructure()
        print("✅ Couchbase infrastructure setup complete")

        # Create agent
        print("🤖 Creating flight search agent...")
        agent_graph = self.create_agent(self.catalog, self.span)
        print("✅ Flight search agent created")

        results = []

        for i, query in enumerate(test_inputs, 1):
            print(f"  📝 Query {i}/{len(test_inputs)}: {query}")

            try:
                print(f"     🔄 Agent processing query...")

                # Create initial state using the proper method
                initial_state = FlightSearchGraph.build_starting_state(query=query)

                # Run the agent
                result = agent_graph.invoke(initial_state)

                # Extract the response from the final state
                # The result should have messages or similar structure
                response = self._extract_response(result)

                print(f"     ✅ Response received")

                # Analyze response
                has_flight_info = self._check_flight_info(response)
                has_booking_info = self._check_booking_info(response)
                has_policy_info = self._check_policy_info(response)
                appropriate_response = self._check_appropriate_response(query, response)

                # Create response preview
                response_preview = (response[:150] + "...") if len(response) > 150 else response
                print(f"     💬 Response preview: {response_preview}")

                results.append(
                    {
                        "example_id": f"flight_query_{i}",
                        "query": query,
                        "output": response,
                        "has_flight_info": has_flight_info,
                        "has_booking_info": has_booking_info,
                        "has_policy_info": has_policy_info,
                        "appropriate_response": appropriate_response,
                        "response_length": len(response),
                    }
                )

            except Exception as e:
                print(f"     ❌ Error processing query: {e}")
                results.append(
                    {
                        "example_id": f"flight_query_{i}",
                        "query": query,
                        "output": f"Error: {str(e)}",
                        "has_flight_info": False,
                        "has_booking_info": False,
                        "has_policy_info": False,
                        "appropriate_response": False,
                        "response_length": 0,
                    }
                )

        return pd.DataFrame(results)

    def _extract_response(self, result: dict) -> str:
        """Extract response text from LangGraph result."""
        try:
            # Check if there are messages in the result
            messages = result.get("messages", [])
            if messages:
                # Get the last message (which should be the AI response)
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                elif isinstance(last_message, dict):
                    return last_message.get("content", str(last_message))

            # Fallback to search results if available
            search_results = result.get("search_results", [])
            if search_results:
                return f"Found {len(search_results)} flight options. " + str(search_results)

            # Final fallback
            return str(result)
        except Exception as e:
            return f"Error extracting response: {e}"

    def _check_flight_info(self, response_text: str) -> bool:
        """Check if response contains flight information."""
        flight_indicators = [
            "flight",
            "airline",
            "departure",
            "arrival",
            "aircraft",
            "gate",
            "terminal",
            "seat",
        ]
        return any(indicator in response_text.lower() for indicator in flight_indicators)

    def _check_booking_info(self, response_text: str) -> bool:
        """Check if response contains booking information."""
        booking_indicators = [
            "booking",
            "reservation",
            "ticket",
            "passenger",
            "confirmation",
            "booked",
            "reserved",
        ]
        return any(indicator in response_text.lower() for indicator in booking_indicators)

    def _check_policy_info(self, response_text: str) -> bool:
        """Check if response mentions policies."""
        policy_indicators = [
            "policy",
            "baggage",
            "cancellation",
            "refund",
            "terms",
            "conditions",
            "rules",
        ]
        return any(indicator in response_text.lower() for indicator in policy_indicators)

    def _check_appropriate_response(self, query: str, response: str) -> bool:
        """Check if response is appropriate for the query."""
        # Check for hallucination indicators
        hallucination_indicators = [
            "mars",
            "pluto",
            "atlantis",
            "fictional",
            "impossible",
            "cannot travel",
        ]
        has_hallucination = any(
            indicator in response.lower() for indicator in hallucination_indicators
        )

        # Check for reasonable response length
        reasonable_length = 50 < len(response) < 2000

        # Check for flight-related content
        has_flight_content = self._check_flight_info(response)

        return not has_hallucination and reasonable_length and has_flight_content

    def run_arize_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Arize-based LLM evaluations on the results.

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            DataFrame with additional Arize evaluation columns
        """
        if not ARIZE_AVAILABLE:
            print("⚠️ Arize not available - skipping LLM evaluations")
            return results_df

        print(f"🧠 Running LLM-based evaluations on {len(results_df)} responses...")
        print("   📋 Evaluation criteria:")
        print("      🔍 Relevance: Does the response address the flight search query?")
        print("      🎯 Correctness: Is the flight information accurate and helpful?")

        # Sample evaluation data preview
        if len(results_df) > 0:
            sample_row = results_df.iloc[0]
            print(f"\n   🔍 Sample evaluation data:")
            print(f"      Query: {sample_row['query']}")
            print(f"      Response: {sample_row['output'][:100]}...")

        try:
            # Prepare data for evaluation
            evaluation_data = []
            for _, row in results_df.iterrows():
                evaluation_data.append(
                    {
                        "reference": row["output"],  # Use the agent's output as reference
                        "input": row["query"],
                        "output": row["output"],
                    }
                )

            eval_df = pd.DataFrame(evaluation_data)

            # Run relevance evaluation
            print(f"\n   🔍 Evaluating relevance...")
            relevance_results = llm_classify(
                dataframe=eval_df,
                model=self.evaluator_llm,
                template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                rails=self.relevance_rails,
                provide_explanation=True,
            )

            # Run correctness evaluation
            print(f"   🎯 Evaluating correctness...")
            correctness_results = llm_classify(
                dataframe=eval_df,
                model=self.evaluator_llm,
                template=QA_PROMPT_TEMPLATE,
                rails=self.qa_rails,
                provide_explanation=True,
            )

            # Add evaluation results to DataFrame
            results_df["arize_relevance"] = relevance_results["label"]
            results_df["arize_relevance_explanation"] = relevance_results["explanation"]
            results_df["arize_correctness"] = correctness_results["label"]
            results_df["arize_correctness_explanation"] = correctness_results["explanation"]

            # Display sample explanations
            if len(results_df) > 0:
                print(f"\n   📝 Sample evaluation explanations:")
                for i in range(min(2, len(results_df))):
                    row = results_df.iloc[i]
                    print(f"      Query: {row['query']}")
                    print(
                        f"      Relevance ({row['arize_relevance']}): {row['arize_relevance_explanation'][:100]}..."
                    )
                    print(
                        f"      Correctness ({row['arize_correctness']}): {row['arize_correctness_explanation'][:100]}..."
                    )
                    print()

            print(f"   ✅ LLM evaluations completed")

        except Exception as e:
            print(f"   ❌ Error running LLM evaluations: {e}")

        return results_df


def eval_flight_search_basic():
    """Run basic flight search evaluation with a small set of test queries."""
    print("🔍 Running basic flight search evaluation...")
    print("📋 This evaluation tests:")
    print("   • Agent's ability to understand flight search queries")
    print("   • Quality of responses using flight lookup tools")
    print("   • LLM-based relevance and correctness scoring")

    # Test queries for flight search
    test_inputs = [
        "Find flights from New York to Los Angeles tomorrow",
        "I need a flight from JFK to LAX on Friday",
        "Show me flights from San Francisco to Chicago next week",
        "Book a flight from Miami to Boston for 2 passengers",
        "What are the flight options from Seattle to Denver?",
        "Can you help me find flights from Atlanta to Phoenix?",
    ]

    # Initialize evaluation components
    catalog = agentc.Catalog()
    span = catalog.Span(name="FlightSearchEvaluation")

    evaluator = ArizeFlightSearchEvaluator(catalog, span)

    # Run the evaluation
    results_df = evaluator.run_flight_search_evaluation(test_inputs)

    # Run Arize evaluations if available
    if ARIZE_AVAILABLE:
        results_df = evaluator.run_arize_evaluations(results_df)

    # Calculate metrics
    total_queries = len(results_df)
    queries_with_flight_info = results_df["has_flight_info"].sum()
    queries_with_booking_info = results_df["has_booking_info"].sum()
    queries_with_policy_info = results_df["has_policy_info"].sum()
    appropriate_responses = results_df["appropriate_response"].sum()
    success_rate = (queries_with_flight_info / total_queries) * 100

    # Print summary
    print(f"\n✅ Basic flight search evaluation completed:")
    print(f"   📊 Total queries processed: {total_queries}")
    print(f"   🎯 Queries with flight info: {queries_with_flight_info}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"   ✈️ Queries with booking info: {queries_with_booking_info}")
    print(f"   📋 Queries with policy info: {queries_with_policy_info}")
    print(f"   ✅ Appropriate responses: {appropriate_responses}")

    if ARIZE_AVAILABLE and "arize_relevance" in results_df.columns:
        relevance_scores = results_df["arize_relevance"].value_counts()
        correctness_scores = results_df["arize_correctness"].value_counts()

        # Convert to regular Python dict with int values
        relevance_dict = {k: int(v) for k, v in relevance_scores.items()}
        correctness_dict = {k: int(v) for k, v in correctness_scores.items()}

        print(f"\n🔍 Arize Evaluation Results:")
        print(f"   📋 Relevance: {relevance_dict}")
        print(f"   ✅ Correctness: {correctness_dict}")

    print(f"\n💡 Note: Some errors are expected without full Couchbase setup")

    return results_df


if __name__ == "__main__":
    import sys

    # Check if specific evaluation is requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            eval_flight_search_basic()
        else:
            print("Usage: python eval_arize.py [basic]")
    else:
        # Run basic evaluation by default
        print("🔍 Running basic flight search agent evaluation...")
        eval_flight_search_basic()
