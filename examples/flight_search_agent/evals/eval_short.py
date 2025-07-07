#!/usr/bin/env python3
"""
Arize AI Integration for Flight Search Agent Evaluation

This module provides comprehensive evaluation capabilities using Arize AI observability
platform alongside the existing Agent Catalog infrastructure.
"""

import json
import os
import pathlib
import unittest.mock
from typing import Any, Dict, List
from uuid import uuid4

import agentc
import pandas as pd

# Configuration constants
SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY", "your-developer-key")
PROJECT_NAME = "flight-search-agent-evaluation"

# Import the flight search agent components
from main import FlightSearchGraph, setup_environment

# Try to import Arize dependencies with fallback
try:
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.experiments.types import (
        EvaluationResultColumnNames,
        ExperimentTaskResultColumnNames,
    )
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from arize.otel import register
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

    ARIZE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Arize dependencies not available: {e}")
    print("   Running in local evaluation mode only...")
    ARIZE_AVAILABLE = False


class ArizeFlightSearchEvaluator:
    """
    Comprehensive evaluation system for flight search agents using Arize AI.
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

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            # Setup tracer provider
            self.tracer_provider = register(
                space_id=SPACE_ID,
                api_key=API_KEY,
                project_name=PROJECT_NAME,
            )

            # Instrument LangChain and OpenAI
            LangChainInstrumentor().instrument(tracer_provider=self.tracer_provider)
            OpenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)

            # Initialize Arize datasets client
            self.arize_client = ArizeDatasetsClient(developer_key=DEVELOPER_KEY, api_key=API_KEY)

            print("✅ Arize observability configured successfully")

        except Exception as e:
            print(f"⚠️ Warning: Could not configure Arize observability: {e}")
            print("   Continuing with local evaluation only...")

    def run_agent_evaluation(self, test_inputs: List[str]) -> pd.DataFrame:
        """
        Run the flight search agent on test inputs and collect results.

        Args:
            test_inputs: List of flight search queries to evaluate

        Returns:
            DataFrame with input queries and agent responses
        """
        results = []

        with self.span.new("FlightSearchEvaluation") as eval_span:
            # Setup the agent once
            flight_graph = FlightSearchGraph(catalog=self.catalog, span=eval_span)
            compiled_graph = flight_graph.compile()

            for i, query in enumerate(test_inputs):
                with eval_span.new(f"Query_{i}", query=query) as query_span:
                    try:
                        # Mock input to prevent interactive prompts
                        with unittest.mock.patch("builtins.input", return_value=query):
                            # Create starting state
                            state = FlightSearchGraph.build_starting_state(query=query)

                            # Run the agent
                            result = compiled_graph.invoke(state)

                            # Extract the response
                            response = ""
                            if result.get("messages"):
                                last_message = result["messages"][-1]
                                if hasattr(last_message, "content"):
                                    response = last_message.content

                            results.append(
                                {
                                    "input": query,
                                    "output": response,
                                    "resolved": result.get("resolved", False),
                                    "search_results_count": len(result.get("search_results", [])),
                                    "example_id": i,
                                }
                            )

                            query_span["response"] = response
                            query_span["resolved"] = result.get("resolved", False)

                    except Exception as e:
                        print(f"❌ Error evaluating query '{query}': {e}")
                        results.append(
                            {
                                "input": query,
                                "output": f"Error: {str(e)}",
                                "resolved": False,
                                "search_results_count": 0,
                                "example_id": i,
                            }
                        )
                        query_span["error"] = str(e)

        return pd.DataFrame(results)

    def run_llm_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run LLM-based evaluations on agent responses.

        Args:
            results_df: DataFrame with agent inputs and outputs

        Returns:
            DataFrame with evaluation scores added
        """
        if not ARIZE_AVAILABLE:
            print("⚠️ LLM evaluations not available - Arize dependencies missing")
            return results_df

        try:
            # Relevance evaluation
            print("🔍 Running relevance evaluation...")
            relevance_eval_df = llm_classify(
                dataframe=results_df,
                template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                model=self.evaluator_llm,
                rails=self.relevance_rails,
                provide_explanation=True,
                include_prompt=True,
                concurrency=2,
            )

            # Correctness evaluation
            print("🎯 Running correctness evaluation...")
            correctness_eval_df = llm_classify(
                dataframe=results_df,
                template=QA_PROMPT_TEMPLATE,
                model=self.evaluator_llm,
                rails=self.qa_rails,
                provide_explanation=True,
                include_prompt=True,
                concurrency=2,
            )

            # Merge evaluations
            merged_df = results_df.copy()
            merged_df["relevance"] = relevance_eval_df.get("label", "unknown")
            merged_df["relevance_explanation"] = relevance_eval_df.get("explanation", "")
            merged_df["correctness"] = correctness_eval_df.get("label", "unknown")
            merged_df["correctness_explanation"] = correctness_eval_df.get("explanation", "")

            return merged_df

        except Exception as e:
            print(f"❌ Error running LLM evaluations: {e}")
            return results_df

    def evaluate_flight_search_scenarios(self) -> Dict[str, pd.DataFrame]:
        """
        Run comprehensive evaluation on different flight search scenarios.

        Returns:
            Dictionary of scenario names to evaluation results
        """
        scenarios = {
            "basic_queries": [
                "Find flights from SFO to LAX",
                "I want to book a flight from New York to Los Angeles",
                "Show me flights from Miami to Chicago",
                "Find the cheapest flights from Boston to Seattle",
            ],
            "complex_queries": [
                "I need a flight from San Francisco to Tokyo with a layover",
                "Find round-trip flights from LAX to LHR departing next week",
                "Book a first-class flight from JFK to SFO for 2 passengers",
                "I want to find flights with flexible dates from Chicago to Miami",
            ],
            "policy_queries": [
                "What is the baggage policy for domestic flights?",
                "Can I cancel my flight booking?",
                "What happens if my flight is delayed due to weather?",
                "Do you have any COVID-19 travel restrictions?",
            ],
            "booking_queries": [
                "Retrieve my existing flight bookings",
                "Save this flight booking for later",
                "Cancel my flight reservation",
                "Modify my existing booking",
            ],
            "irrelevant_queries": [
                "What's the weather like today?",
                "How do I bake a cake?",
                "Tell me about the stock market",
                "What's the capital of France?",
            ],
        }

        results = {}

        for scenario_name, queries in scenarios.items():
            print(f"\n🚀 Evaluating scenario: {scenario_name}")
            print(f"   Running {len(queries)} queries...")

            # Run agent evaluation
            agent_results = self.run_agent_evaluation(queries)

            # Run LLM evaluations if available
            if ARIZE_AVAILABLE:
                evaluated_results = self.run_llm_evaluations(agent_results)
            else:
                evaluated_results = agent_results

            results[scenario_name] = evaluated_results

            # Print summary
            total_queries = len(evaluated_results)
            resolved_count = sum(evaluated_results["resolved"])

            print(f"   ✅ Completed: {resolved_count}/{total_queries} queries resolved")

            if ARIZE_AVAILABLE:
                avg_relevance = self._calculate_average_score(
                    evaluated_results.get("relevance", [])
                )
                avg_correctness = self._calculate_average_score(
                    evaluated_results.get("correctness", [])
                )
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

        numeric_scores = [score_map.get(score.lower(), 0.5) for score in scores]
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
        report.append("# Flight Search Agent Evaluation Report")
        report.append("=" * 50)
        report.append("")

        total_queries = 0
        total_resolved = 0

        for scenario_name, df in results.items():
            report.append(f"## {scenario_name.replace('_', ' ').title()}")
            report.append(f"- Total Queries: {len(df)}")
            report.append(f"- Resolved: {sum(df['resolved'])}")
            report.append(f"- Success Rate: {sum(df['resolved']) / len(df) * 100:.1f}%")

            if ARIZE_AVAILABLE and "relevance" in df.columns:
                avg_relevance = self._calculate_average_score(df["relevance"])
                report.append(f"- Avg Relevance: {avg_relevance:.2f}")

            if ARIZE_AVAILABLE and "correctness" in df.columns:
                avg_correctness = self._calculate_average_score(df["correctness"])
                report.append(f"- Avg Correctness: {avg_correctness:.2f}")

            report.append("")

            total_queries += len(df)
            total_resolved += sum(df["resolved"])

        report.append("## Overall Summary")
        report.append(f"- Total Queries Evaluated: {total_queries}")
        report.append(f"- Total Resolved: {total_resolved}")
        report.append(f"- Overall Success Rate: {total_resolved / total_queries * 100:.1f}%")
        report.append("")

        if ARIZE_AVAILABLE and self.arize_client:
            report.append("## Arize AI Dashboard")
            report.append("View detailed traces and metrics in your Arize workspace:")
            report.append(f"- Project: {PROJECT_NAME}")
            report.append(f"- Space ID: {SPACE_ID}")
            report.append("")

        return "\n".join(report)


def eval_bad_intro():
    """
    Evaluate agent behavior with irrelevant queries.
    """
    print("🔍 Running bad intro evaluation...")

    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    root_span = catalog.Span(name="FlightSearchEvalBadIntro")

    # Initialize evaluator
    evaluator = ArizeFlightSearchEvaluator(catalog=catalog, span=root_span)

    # Load test cases
    test_file = pathlib.Path("evals") / "resources" / "bad-intro.jsonl"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    with (
        test_file.open() as fp,
        root_span.new("IrrelevantQueries") as suite_span,
    ):
        test_inputs = []
        for line in fp:
            try:
                test_case = json.loads(line.strip())
                if test_case.get("input"):
                    test_inputs.append(test_case["input"])
            except json.JSONDecodeError:
                continue

        if not test_inputs:
            print("❌ No valid test cases found")
            return

        # Run evaluation
        results = evaluator.run_agent_evaluation(test_inputs)

        # Check if agent correctly identifies irrelevant queries
        correctly_handled = sum(
            1
            for _, row in results.iterrows()
            if "I can help you with flight" in row.get("output", "").lower()
            or "flight search" in row.get("output", "").lower()
        )

        print(f"✅ Bad intro evaluation completed:")
        print(f"   - Total queries: {len(results)}")
        print(f"   - Correctly handled: {correctly_handled}")
        print(f"   - Success rate: {correctly_handled / len(results) * 100:.1f}%")

        # Log results
        suite_span["total_queries"] = len(results)
        suite_span["correctly_handled"] = correctly_handled
        suite_span["success_rate"] = correctly_handled / len(results)


def eval_short_threads():
    """
    Evaluate agent behavior with valid flight search queries.
    """
    print("🔍 Running short threads evaluation...")

    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    root_span = catalog.Span(name="FlightSearchEvalShortThreads")

    # Initialize evaluator
    evaluator = ArizeFlightSearchEvaluator(catalog=catalog, span=root_span)

    # Load test cases
    test_file = pathlib.Path("evals") / "resources" / "short-thread.jsonl"
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return

    with (
        test_file.open() as fp,
        root_span.new("ValidFlightQueries") as suite_span,
    ):
        test_cases = []
        for line in fp:
            try:
                test_case = json.loads(line.strip())
                if test_case.get("input"):
                    # Handle both single queries and conversation threads
                    if isinstance(test_case["input"], list):
                        # Take the first query from conversation threads
                        query = test_case["input"][0]
                    else:
                        query = test_case["input"]

                    test_cases.append(
                        {
                            "input": query,
                            "reference": test_case.get("reference", ""),
                        }
                    )
            except json.JSONDecodeError:
                continue

        if not test_cases:
            print("❌ No valid test cases found")
            return

        # Extract just the queries for evaluation
        queries = [case["input"] for case in test_cases]

        # Run evaluation
        results = evaluator.run_agent_evaluation(queries)

        # Run LLM evaluations if available
        if ARIZE_AVAILABLE:
            evaluated_results = evaluator.run_llm_evaluations(results)
        else:
            evaluated_results = results

        # Calculate metrics
        total_queries = len(evaluated_results)
        resolved_count = sum(evaluated_results["resolved"])

        print(f"✅ Short threads evaluation completed:")
        print(f"   - Total queries: {total_queries}")
        print(f"   - Resolved: {resolved_count}")
        print(f"   - Success rate: {resolved_count / total_queries * 100:.1f}%")

        if ARIZE_AVAILABLE:
            avg_relevance = evaluator._calculate_average_score(
                evaluated_results.get("relevance", [])
            )
            avg_correctness = evaluator._calculate_average_score(
                evaluated_results.get("correctness", [])
            )
            print(f"   - Avg relevance: {avg_relevance:.2f}")
            print(f"   - Avg correctness: {avg_correctness:.2f}")

        # Log results
        suite_span["total_queries"] = total_queries
        suite_span["resolved_count"] = resolved_count
        suite_span["success_rate"] = resolved_count / total_queries


def run_comprehensive_evaluation():
    """
    Run comprehensive evaluation across all scenarios.
    """
    print("🚀 Starting comprehensive flight search agent evaluation...")

    # Setup environment
    setup_environment()

    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    root_span = catalog.Span(name="FlightSearchComprehensiveEval")

    # Initialize evaluator
    evaluator = ArizeFlightSearchEvaluator(catalog=catalog, span=root_span)

    # Run comprehensive evaluation
    results = evaluator.evaluate_flight_search_scenarios()

    # Generate report
    report = evaluator.generate_evaluation_report(results)

    # Save report
    report_file = pathlib.Path("evals") / "evaluation_report.md"
    with report_file.open("w") as f:
        f.write(report)

    print(f"✅ Comprehensive evaluation completed!")
    print(f"   📊 Report saved to: {report_file}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    import sys

    # Check if specific evaluation is requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "bad_intro":
            eval_bad_intro()
        elif sys.argv[1] == "short_threads":
            eval_short_threads()
        elif sys.argv[1] == "comprehensive":
            run_comprehensive_evaluation()
        else:
            print("Usage: python eval_short.py [bad_intro|short_threads|comprehensive]")
    else:
        # Run all evaluations by default
        print("🔍 Running all flight search agent evaluations...")
        eval_bad_intro()
        print("\n" + "=" * 50 + "\n")
        eval_short_threads()
        print("\n" + "=" * 50 + "\n")
        run_comprehensive_evaluation()
