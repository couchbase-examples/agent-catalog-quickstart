#!/usr/bin/env python3
"""
Arize AI Integration for Route Planner Agent Evaluation

This module provides comprehensive evaluation capabilities using Arize AI observability
platform for the route planner agent. It demonstrates how to:

1. Set up Arize observability for LlamaIndex-based agents
2. Create and manage evaluation datasets for route planning scenarios
3. Run automated evaluations with LLM-as-a-judge for route quality
4. Track performance metrics and traces for travel planning systems
5. Monitor tool usage and route optimization effectiveness

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
logging.getLogger("openai").setLevel(logging.WARNING)  # Suppress OpenAI API logs
logging.getLogger("llama_index").setLevel(logging.WARNING)  # Suppress LlamaIndex logs

# Configuration constants
SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
# Note: Developer keys are deprecated in favor of User API Keys
# DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY", "your-developer-key")  # Deprecated
PROJECT_NAME = "route-planner-agent-evaluation"

# Import the route planner agent components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import RouteDataLoader, CouchbaseSetup, RouteplannerAgent

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
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
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


class ArizeRoutePlannerEvaluator:
    """
    Comprehensive evaluation system for route planner agents using Arize AI.
    
    This class provides:
    - Route planning performance evaluation with multiple metrics
    - Tool effectiveness monitoring (search, distance, POI, transport)
    - Route quality assessment and optimization tracking
    - Comparative analysis of different routing strategies
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
            logger.info("Couchbase infrastructure setup complete for evaluation")
            return cluster
        except Exception as e:
            logger.error(f"Error setting up Couchbase infrastructure: {e}")
            raise

    def create_agent(self, catalog: agentc.Catalog, span: agentc.Span):
        """Create a route planner agent for evaluation."""
        logger = logging.getLogger(__name__)
        
        try:
            agent = RouteplannerAgent(span=span)
            logger.info("Route planner agent created for evaluation")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            # Setup tracer provider for Phoenix (local only, no remote collector)
            # Use simple local tracer to avoid connection issues
            self.tracer_provider = TracerProvider()
            
            # Skip console span processor to reduce verbose output
            # Only use it for debugging if needed
            # console_processor = SimpleSpanProcessor(ConsoleSpanExporter())
            # self.tracer_provider.add_span_processor(console_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            print("✅ Local tracing configured successfully")
            
            # Instrument LangChain, OpenAI, and LlamaIndex (only if not already instrumented)
            # Skip instrumentation if auto_instrument was used or if already done
            instrumentors = [
                ("LangChain", LangChainInstrumentor),
                ("OpenAI", OpenAIInstrumentor),
                # Skip LlamaIndex due to private attributes error
                # ("LlamaIndex", LlamaIndexInstrumentor)
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
                    
            # Skip LlamaIndex instrumentation for now due to compatibility issues
            print("ℹ️ LlamaIndex instrumentation skipped (compatibility issue)")
            
            # Initialize Arize datasets client
            # Note: Using API_KEY for both parameters as developer keys are deprecated
            if API_KEY != "your-api-key":
                try:
                    self.arize_client = ArizeDatasetsClient(
                        developer_key=API_KEY,  # Use API key for both
                        api_key=API_KEY
                    )
                    print("✅ Arize datasets client initialized")
                except Exception as e:
                    print(f"⚠️ Could not initialize Arize datasets client: {e}")
                    self.arize_client = None
            
            print("✅ Arize observability configured successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not configure Arize observability: {e}")
            print("   Continuing with local evaluation only...")

    def run_route_planning_evaluation(self, test_inputs: List[str]) -> pd.DataFrame:
        """
        Run route planning evaluation with agent responses.
        
        Args:
            test_inputs: List of test queries to evaluate
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        print(f"🚀 Running evaluation on {len(test_inputs)} queries...")
        
        # Setup infrastructure once
        print("🔧 Setting up Couchbase infrastructure...")
        self.setup_couchbase_infrastructure()
        
        # Create agent once
        print("🤖 Creating route planner agent...")
        agent = self.create_agent(self.catalog, self.span)
        
        for i, query in enumerate(test_inputs, 1):
            print(f"  📝 Query {i}/{len(test_inputs)}: {query[:50]}{'...' if len(query) > 50 else ''}")
            
            try:
                # Get agent response
                print(f"     🔄 Agent processing query (using vector search + LLM)...")
                response = agent.plan_route(query)
                response_text = str(response) if response else "No response"
                
                # Check if response contains route information
                has_route_info = self._check_route_info(response_text)
                
                results.append({
                    "input": query,
                    "output": response_text,
                    "has_route_info": has_route_info,
                })
                
                # Show what kind of info was found
                route_types = []
                if "distance" in response_text.lower() or "miles" in response_text.lower() or "km" in response_text.lower():
                    route_types.append("distance info")
                if any(word in response_text.lower() for word in ["direction", "turn", "north", "south", "east", "west"]):
                    route_types.append("directions")
                if any(word in response_text.lower() for word in ["car", "drive", "flight", "train", "bus"]):
                    route_types.append("transport options")
                
                route_details = f" ({', '.join(route_types)})" if route_types else ""
                print(f"     ✅ Response received ({'with route info' if has_route_info else 'no route info'}){route_details}")
                
                # Show a sample of the response for debugging
                response_preview = response_text[:300].replace('\n', ' ')
                print(f"     💬 Response preview: {response_preview}{'...' if len(response_text) > 300 else ''}")
                
            except Exception as e:
                print(f"     ❌ Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
                results.append({
                    "input": query,
                    "output": f"Error: {e}",
                    "has_route_info": False,
                })
        
        return pd.DataFrame(results)
    
    def _check_route_info(self, response_text: str) -> bool:
        """Check if response contains route-related information."""
        route_keywords = ["route", "distance", "travel", "miles", "km", "minutes", "hours", 
                         "direction", "turn", "north", "south", "east", "west",
                         "car", "drive", "flight", "train", "bus"]
        return any(keyword in response_text.lower() for keyword in route_keywords)

    def run_arize_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run Arize LLM-based evaluations on agent responses.
        
        Args:
            results_df: DataFrame with agent inputs and outputs
            
        Returns:
            DataFrame with Arize evaluation scores added
        """
        if not ARIZE_AVAILABLE:
            print("⚠️ Arize evaluations not available - dependencies missing")
            return results_df
            
        try:
            # Prepare data for evaluation - add reference field and rename columns
            eval_df = results_df.copy()
            # Use the agent's response as the reference text for evaluation
            # This allows the evaluator to check if the response is relevant to the query
            eval_df["reference"] = eval_df["output"]  # Agent response as reference
            eval_df["query"] = eval_df["input"]       # User query
            eval_df["response"] = eval_df["output"]   # Agent response (for correctness evaluation)
            
            print(f"🧠 Running LLM-based evaluations on {len(eval_df)} responses...")
            print("   📋 Evaluation criteria:")
            print("      🔍 Relevance: Does the response address the route planning query?")
            print("      🎯 Correctness: Is the route information accurate and helpful?")
            
            # Show sample evaluation data being sent
            print("\n   🔍 Sample evaluation data:")
            for i, row in eval_df.head(1).iterrows():  # Show first example
                print(f"      Query: {row['query']}")
                print(f"      Reference: {row['reference'][:100]}{'...' if len(row['reference']) > 100 else ''}")
                print(f"      Response: {row['response'][:100]}{'...' if len(row['response']) > 100 else ''}")
                break
            
            # Relevance evaluation
            print("\n   🔍 Evaluating relevance...")
            relevance_eval_df = llm_classify(
                data=eval_df,
                template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                model=self.evaluator_llm,
                rails=self.relevance_rails,
                provide_explanation=True,
                include_prompt=False,  # Reduce output verbosity
                concurrency=2,
                verbose=False,  # Reduce progress bar verbosity
            )
            
            # Correctness evaluation
            print("   🎯 Evaluating correctness...")
            correctness_eval_df = llm_classify(
                data=eval_df,
                template=QA_PROMPT_TEMPLATE,
                model=self.evaluator_llm,
                rails=self.qa_rails,
                provide_explanation=True,
                include_prompt=False,  # Reduce output verbosity
                concurrency=2,
                verbose=False,  # Reduce progress bar verbosity
            )
            
            # Merge evaluations
            merged_df = results_df.copy()
            merged_df["arize_relevance"] = relevance_eval_df.get("label", "unknown")
            merged_df["arize_relevance_explanation"] = relevance_eval_df.get("explanation", "")
            merged_df["arize_correctness"] = correctness_eval_df.get("label", "unknown")
            merged_df["arize_correctness_explanation"] = correctness_eval_df.get("explanation", "")
            
            # Show sample evaluation explanations
            print("\n   📝 Sample evaluation explanations:")
            for i, row in merged_df.head(2).iterrows():  # Show first 2 examples
                print(f"      Query: {row['input']}")
                print(f"      Relevance ({row['arize_relevance']}): {row['arize_relevance_explanation'][:200]}{'...' if len(row['arize_relevance_explanation']) > 200 else ''}")
                print(f"      Correctness ({row['arize_correctness']}): {row['arize_correctness_explanation'][:200]}{'...' if len(row['arize_correctness_explanation']) > 200 else ''}")
                print()
            
            print("   ✅ LLM evaluations completed")
            return merged_df
            
        except Exception as e:
            print(f"❌ Error running Arize evaluations: {e}")
            return results_df

    def evaluate_route_planning_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive evaluation on different route planning scenarios.
        
        Returns:
            Dictionary of scenario names to evaluation results
        """
        scenarios = {
            "basic_routing": [
                "Plan a route from New York to Boston",
                "I need directions from Los Angeles to San Francisco",
                "Route from Chicago to Detroit",
                "How do I get from Miami to Orlando?",
            ],
            "multi_stop_journeys": [
                "Plan a road trip from Seattle to Portland to San Francisco",
                "Route with stops in New York, Philadelphia, and Washington DC",
                "Multi-city tour: Los Angeles, Las Vegas, Grand Canyon",
                "Travel itinerary covering Boston, New York, and Philadelphia",
            ],
            "transportation_comparisons": [
                "Compare driving vs flying from New York to Los Angeles",
                "Best way to travel from Boston to Washington DC",
                "Train vs car from San Francisco to Los Angeles",
                "Most efficient transport from Chicago to New York",
            ],
            "scenic_routes": [
                "Scenic route from Denver to Salt Lake City",
                "Beautiful drive from San Francisco to Los Angeles",
                "Scenic highway from Seattle to Vancouver",
                "Picturesque route through the Rocky Mountains",
            ],
            "distance_calculations": [
                "How far is it from New York to Miami?",
                "Distance and travel time from Los Angeles to Seattle",
                "Calculate driving distance from Chicago to Houston",
                "Flight distance between Boston and Los Angeles",
            ],
            "poi_discovery": [
                "Points of interest between New York and Boston",
                "Attractions along the route from LA to San Francisco",
                "Restaurants and hotels on the way to Las Vegas",
                "Tourist spots between Chicago and Detroit",
            ],
            "irrelevant_queries": [
                "What's the weather like today?",
                "How do I cook dinner?",
                "Tell me about stock markets",
                "What is quantum physics?",
            ],
        }
        
        results = {}
        
        for scenario_name, queries in scenarios.items():
            print(f"\n🚀 Evaluating scenario: {scenario_name}")
            print(f"   Running {len(queries)} queries...")
            
            # Run route planning evaluation
            agent_results = self.run_route_planning_evaluation(queries)
            
            # Run Arize evaluations if available
            if ARIZE_AVAILABLE:
                evaluated_results = self.run_arize_evaluations(agent_results)
            else:
                evaluated_results = agent_results
            
            results[scenario_name] = evaluated_results
            
            # Print summary
            total_queries = len(evaluated_results)
            route_info_count = sum(evaluated_results["has_route_info"])
            directions_count = sum(evaluated_results["has_directions"])
            transport_info_count = sum(evaluated_results["has_transport_info"])
            poi_info_count = sum(evaluated_results["has_poi_info"])
            
            print(f"   ✅ Completed: {route_info_count}/{total_queries} queries with route info")
            print(f"   🧭 Directions: {directions_count}/{total_queries} queries")
            print(f"   🚗 Transport info: {transport_info_count}/{total_queries} queries")
            print(f"   📍 POI info: {poi_info_count}/{total_queries} queries")
            
            if ARIZE_AVAILABLE and "arize_relevance" in evaluated_results.columns:
                avg_relevance = self._calculate_average_score(evaluated_results["arize_relevance"])
                avg_correctness = self._calculate_average_score(evaluated_results["arize_correctness"])
                print(f"   📊 Avg Relevance: {avg_relevance:.2f}")
                print(f"   📊 Avg Correctness: {avg_correctness:.2f}")
        
        return results

    def _calculate_average_score(self, scores: List[str]) -> float:
        """Calculate average score from evaluation labels."""
        if not scores:
            return 0.0
        
        # Map evaluation labels to numeric scores
        score_map = {
            "correct": 1.0,
            "incorrect": 0.0,
            "relevant": 1.0,
            "irrelevant": 0.0,
            "unknown": 0.5,
        }
        
        numeric_scores = [score_map.get(str(score).lower(), 0.5) for score in scores]
        return sum(numeric_scores) / len(numeric_scores)

    def generate_evaluation_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Dictionary of scenario results
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("# Route Planner Agent Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        total_queries = 0
        total_route_info = 0
        total_directions = 0
        total_transport_info = 0
        total_poi_info = 0
        
        for scenario_name, df in results.items():
            report.append(f"## {scenario_name.replace('_', ' ').title()}")
            report.append(f"- Total Queries: {len(df)}")
            report.append(f"- With Route Info: {sum(df['has_route_info'])}")
            report.append(f"- With Directions: {sum(df['has_directions'])}")
            report.append(f"- With Transport Info: {sum(df['has_transport_info'])}")
            report.append(f"- With POI Info: {sum(df['has_poi_info'])}")
            report.append(f"- Success Rate: {sum(df['has_route_info']) / len(df) * 100:.1f}%")
            
            # Arize metrics
            if ARIZE_AVAILABLE and "arize_relevance" in df.columns:
                avg_relevance = self._calculate_average_score(df["arize_relevance"])
                avg_correctness = self._calculate_average_score(df["arize_correctness"])
                report.append(f"- Avg Relevance: {avg_relevance:.2f}")
                report.append(f"- Avg Correctness: {avg_correctness:.2f}")
            
            # Response quality metrics
            avg_response_length = df["response_length"].mean()
            report.append(f"- Avg Response Length: {avg_response_length:.0f} chars")
            
            report.append("")
            
            total_queries += len(df)
            total_route_info += sum(df["has_route_info"])
            total_directions += sum(df["has_directions"])
            total_transport_info += sum(df["has_transport_info"])
            total_poi_info += sum(df["has_poi_info"])
        
        report.append("## Overall Summary")
        report.append(f"- Total Queries Evaluated: {total_queries}")
        report.append(f"- Total With Route Info: {total_route_info}")
        report.append(f"- Overall Success Rate: {total_route_info / total_queries * 100:.1f}%")
        report.append(f"- Directions Coverage: {total_directions / total_queries * 100:.1f}%")
        report.append(f"- Transport Info Coverage: {total_transport_info / total_queries * 100:.1f}%")
        report.append(f"- POI Info Coverage: {total_poi_info / total_queries * 100:.1f}%")
        report.append("")
        
        if ARIZE_AVAILABLE and self.arize_client:
            report.append("## Arize AI Dashboard")
            report.append("View detailed traces and metrics in your Arize workspace:")
            report.append(f"- Project: {PROJECT_NAME}")
            report.append(f"- Space ID: {SPACE_ID}")
            report.append("")
        
        return "\n".join(report)

    def log_experiment_to_arize(self, results_df: pd.DataFrame, experiment_name: str) -> str:
        """
        Log experiment results to Arize for analysis and comparison.
        
        Args:
            results_df: DataFrame with evaluation results
            experiment_name: Name for the experiment
            
        Returns:
            Experiment ID if successful, None otherwise
        """
        try:
            if not self.arize_client or not ARIZE_AVAILABLE:
                print("⚠️ No Arize client available, skipping logging")
                return None
            
            # Define column mappings
            task_cols = ExperimentTaskResultColumnNames(
                example_id="example_id",
                result="output"
            )
            
            evaluator_columns = {}
            
            if "arize_relevance" in results_df.columns:
                evaluator_columns["arize_relevance"] = EvaluationResultColumnNames(
                    label="arize_relevance",
                    explanation="arize_relevance_explanation",
                )
            
            if "arize_correctness" in results_df.columns:
                evaluator_columns["arize_correctness"] = EvaluationResultColumnNames(
                    label="arize_correctness",
                    explanation="arize_correctness_explanation",
                )
            
            # Log experiment to Arize
            experiment_id = self.arize_client.log_experiment(
                space_id=SPACE_ID,
                experiment_name=f"{experiment_name}-{str(uuid4())[:4]}",
                experiment_df=results_df,
                task_columns=task_cols,
                evaluator_columns=evaluator_columns,
                dataset_name=f"route-planner-eval-{str(uuid4())[:8]}",
            )
            
            print(f"✅ Logged experiment to Arize: {experiment_name}")
            return experiment_id
            
        except Exception as e:
            print(f"❌ Error logging experiment to Arize: {e}")
            return None


# Standalone helper functions for external use
def setup_couchbase_infrastructure():
    """Standalone function to setup Couchbase infrastructure for evaluation."""
    logger = logging.getLogger(__name__)
    
    try:
        couchbase_setup = CouchbaseSetup()
        couchbase_setup.setup_environment()
        cluster = couchbase_setup.setup_couchbase_connection()
        collection = couchbase_setup.setup_bucket_scope_collection()
        couchbase_setup.setup_vector_search_index()
        logger.info("Couchbase infrastructure setup complete for evaluation")
        return cluster
    except Exception as e:
        logger.error(f"Error setting up Couchbase infrastructure: {e}")
        raise


def create_agent(catalog: agentc.Catalog, span: agentc.Span):
    """Standalone function to create a route planner agent for evaluation."""
    logger = logging.getLogger(__name__)
    
    try:
        agent = RouteplannerAgent(span=span)
        logger.info("Route planner agent created for evaluation")
        return agent
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise


def eval_route_planning_basic():
    """Run basic route planning evaluation with a small set of test queries."""
    print("🔍 Running basic route planning evaluation...")
    print("📋 This evaluation tests:")
    print("   • Agent's ability to understand route planning queries")
    print("   • Quality of responses using vector search + LLM")
    print("   • LLM-based relevance and correctness scoring")
    
    # Basic test queries for route planning
    test_inputs = [
        "Plan a route from New York to Boston",
        "How far is Los Angeles from San Francisco?", 
        "Best way to travel from Chicago to Detroit",
        "Scenic route through Colorado"
    ]
    
    # Initialize evaluation components
    catalog = agentc.Catalog()
    span = catalog.Span(name="RouteplannerEvaluation")
    
    evaluator = ArizeRoutePlannerEvaluator(catalog, span)
    
    # Run the evaluation
    results_df = evaluator.run_route_planning_evaluation(test_inputs)
    
    # Run Arize evaluations if available
    if ARIZE_AVAILABLE:
        results_df = evaluator.run_arize_evaluations(results_df)
    
    # Calculate metrics
    total_queries = len(results_df)
    queries_with_route_info = results_df['has_route_info'].sum()
    success_rate = (queries_with_route_info / total_queries) * 100
    
    # Print summary
    print(f"\n✅ Basic route planning evaluation completed:")
    print(f"   📊 Total queries processed: {total_queries}")
    print(f"   🎯 Queries with route info: {queries_with_route_info}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    
    if ARIZE_AVAILABLE and 'arize_relevance' in results_df.columns:
        relevance_scores = results_df['arize_relevance'].value_counts()
        correctness_scores = results_df['arize_correctness'].value_counts()
        
        # Convert to regular Python dict with int values
        relevance_dict = {k: int(v) for k, v in relevance_scores.items()}
        correctness_dict = {k: int(v) for k, v in correctness_scores.items()}
        
        print(f"\n🔍 Arize Evaluation Results:")
        print(f"   📋 Relevance: {relevance_dict}")
        print(f"   ✅ Correctness: {correctness_dict}")
    
    print(f"\n💡 Note: Some errors are expected without full Couchbase setup")
    
    return results_df


def eval_route_planning_comprehensive():
    """
    Run comprehensive evaluation across all route planning scenarios.
    
    This function provides a complete evaluation suite that tests the
    agent across different types of route planning queries and use cases.
    """
    print("🚀 Starting comprehensive route planner agent evaluation...")
    
    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    root_span = catalog.Span(name="RoutePlannerComprehensiveEval")
    
    # Initialize evaluator
    evaluator = ArizeRoutePlannerEvaluator(catalog=catalog, span=root_span)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_route_planning_scenarios()
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    # Save report
    report_file = pathlib.Path("evals") / "route_planner_evaluation_report.md"
    with report_file.open("w") as f:
        f.write(report)
    
    print(f"✅ Comprehensive evaluation completed!")
    print(f"   📊 Report saved to: {report_file}")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    import sys
    
    # Check if specific evaluation is requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "basic":
            eval_route_planning_basic()
        elif sys.argv[1] == "comprehensive":
            eval_route_planning_comprehensive()
        else:
            print("Usage: python eval_arize.py [basic|comprehensive]")
    else:
        # Run comprehensive evaluation by default
        print("🔍 Running comprehensive route planner agent evaluation...")
        eval_route_planning_comprehensive()