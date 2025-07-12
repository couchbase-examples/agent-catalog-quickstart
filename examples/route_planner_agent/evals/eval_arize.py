#!/usr/bin/env python3
"""
Arize Phoenix Integration for Route Planner Agent

This script demonstrates how to use Arize Phoenix to evaluate the route planner agent
that uses LlamaIndex with Couchbase vector store and has 2 tools:
- search_routes: Semantic search for route information
- calculate_distance: Calculate distance and time between locations

Features:
- Phoenix UI for trace visualization
- LLM-based evaluation with Phoenix evaluators (Relevance + QA)
- Integration with actual route planner agent
- Comprehensive evaluation metrics with route-specific checks
"""

import os
import sys
import logging
import json
import time
import warnings
from typing import List, Dict, Any
import pandas as pd
from dotenv import load_dotenv

# Path-related imports and setup - keep these at the top for sys.path modification
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Environment setup
load_dotenv(dotenv_path=os.path.join(parent_dir, "../../.env"))
load_dotenv(dotenv_path=os.path.join(parent_dir, ".env"), override=True)

# Phoenix imports
try:
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from phoenix.evals import (
        OpenAIModel,
        llm_classify,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        QA_PROMPT_TEMPLATE,
    )
    import urllib3
    PHOENIX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Phoenix dependencies not available: {e}")
    PHOENIX_AVAILABLE = False

# Agent imports
try:
    import agentc
    from main import (
        setup_environment,
        setup_couchbase_connection,
        setup_collection,
        setup_vector_search_index,
        setup_ai_models,
        ingest_route_data,
        create_llamaindex_agent,
    )
    from llama_index.vector_stores.couchbase import CouchbaseSearchVectorStore
    from llama_index.core import Settings
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Agent dependencies not available: {e}")
    AGENT_AVAILABLE = False

# SQLAlchemy imports for handling warnings
try:
    import sqlalchemy
    import sqlalchemy.exc
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("evaluation.log")],
)
logger = logging.getLogger(__name__)

# Keep important third-party logs at reasonable levels
logging.getLogger("requests").setLevel(logging.INFO)
logging.getLogger("urllib3").setLevel(logging.INFO)


class RouteEvaluator:
    """Route planner agent evaluator with Phoenix integration."""

    def __init__(self):
        """Initialize the evaluator."""
        self.phoenix_session = None
        self.tracer_provider = None
        self.agent = None
        self.evaluation_model = None

        # Initialize Phoenix if available
        if PHOENIX_AVAILABLE:
            self.setup_phoenix()

        # Initialize evaluation model
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.evaluation_model = OpenAIModel(model="gpt-4o", temperature=0.1)
            except Exception as e:
                logger.warning(f"Could not initialize evaluation model: {e}")

    def setup_phoenix(self):
        """Setup Phoenix instrumentation and launch UI."""
        try:
            logger.info("🔥 Setting up Phoenix instrumentation...")

            # Set Phoenix environment variables
            os.environ["PHOENIX_HOST"] = "0.0.0.0"
            os.environ["PHOENIX_PORT"] = "6006"

            # Launch Phoenix session
            self.phoenix_session = px.launch_app()
            time.sleep(2)

            # Set up Phoenix instrumentation without external trace export
            current_tracer_provider = trace.get_tracer_provider()
            if hasattr(current_tracer_provider, "__class__") and "TraceProvider" in str(
                current_tracer_provider.__class__
            ):
                logger.info("🔄 Tracer provider already exists, using existing one")
                tracer_provider = current_tracer_provider
            else:
                tracer_provider = register(
                    project_name="route_planner_evaluation",
                    set_global_tracer_provider=False,
                )
                trace.set_tracer_provider(tracer_provider)

            # Instrument LlamaIndex with Phoenix
            if not LlamaIndexInstrumentor().is_instrumented_by_opentelemetry:
                LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

            logger.info("✅ Phoenix instrumentation setup complete")
            logger.info(f"🌐 Phoenix UI available at: http://localhost:6006/")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Phoenix setup failed (evaluation will continue): {e}")
            return False

    def cleanup_phoenix(self):
        """Clean up Phoenix session properly."""
        if self.phoenix_session:
            try:
                if hasattr(self.phoenix_session, "close"):
                    self.phoenix_session.close()
                elif hasattr(self.phoenix_session, "stop"):
                    self.phoenix_session.stop()
                elif hasattr(self.phoenix_session, "shutdown"):
                    self.phoenix_session.shutdown()
                else:
                    logger.info("Phoenix session cleanup - no explicit close method needed")
            except Exception as e:
                logger.warning(f"Phoenix session cleanup warning: {e}")
            finally:
                self.phoenix_session = None

    def setup_agent(self):
        """Setup the route planner agent."""
        try:
            logger.info("🔧 Setting up route planner agent...")

            # Initialize Agent Catalog and span
            catalog = agentc.Catalog()
            span = catalog.Span(name="Route Planner Evaluation Setup")

            # Setup environment
            setup_environment()

            # Setup Couchbase connection
            cluster = setup_couchbase_connection()

            # Setup collection
            setup_collection(
                cluster,
                os.environ["CB_BUCKET_NAME"],
                os.environ["SCOPE_NAME"],
                os.environ["COLLECTION_NAME"],
            )

            # Setup vector search index
            try:
                with open("agentcatalog_index.json", "r") as file:
                    index_definition = json.load(file)
                setup_vector_search_index(cluster, index_definition)
            except Exception as e:
                logger.warning(f"⚠️ Could not setup vector search index: {e}")

            # Setup AI models
            embed_model, llm = setup_ai_models(span)

            # Setup vector store
            vector_store = CouchbaseSearchVectorStore(
                cluster=cluster,
                bucket_name=os.environ["CB_BUCKET_NAME"],
                scope_name=os.environ["SCOPE_NAME"],
                collection_name=os.environ["COLLECTION_NAME"],
                index_name=os.environ["INDEX_NAME"],
            )

            # Ingest route data
            ingest_route_data(vector_store, span)

            # Create agent
            self.agent = create_llamaindex_agent(catalog, span)

            logger.info("✅ Route planner agent setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Agent setup failed: {e}")
            return False

    def run_evaluation(self, test_queries: List[str]) -> pd.DataFrame:
        """Run evaluation on test queries."""
        logger.info(f"🧪 Running evaluation with {len(test_queries)} queries...")

        if not self.agent:
            logger.error("❌ Agent not initialized")
            return pd.DataFrame()

        results = []

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n📝 Query {i}/{len(test_queries)}: {query}")

            try:
                # Get agent response
                response = self.agent.chat(query)
                response_text = self._clean_response(str(response))

                # Create response preview
                preview = (
                    (response_text[:150] + "...") if len(response_text) > 150 else response_text
                )
                logger.info(f"✅ Response ({len(response_text)} chars): {preview}")

                # Evaluate response
                has_route_info = self._check_route_info(response_text)
                has_distance_info = self._check_distance_info(response_text)
                has_travel_time = self._check_travel_time(response_text)
                appropriate_length = self._check_appropriate_length(response_text)
                is_relevant = self._check_relevance(query, response_text)

                results.append(
                    {
                        "query": query,
                        "response": response_text,
                        "response_length": len(response_text),
                        "has_route_info": has_route_info,
                        "has_distance_info": has_distance_info,
                        "has_travel_time": has_travel_time,
                        "appropriate_length": appropriate_length,
                        "is_relevant": is_relevant,
                        "quality_score": self._calculate_quality_score(
                            has_route_info,
                            has_distance_info,
                            has_travel_time,
                            appropriate_length,
                            is_relevant,
                        ),
                    }
                )

            except Exception as e:
                logger.error(f"❌ Error processing query: {e}")
                results.append(
                    {
                        "query": query,
                        "response": f"Error: {str(e)}",
                        "response_length": 0,
                        "has_route_info": False,
                        "has_distance_info": False,
                        "has_travel_time": False,
                        "appropriate_length": False,
                        "is_relevant": False,
                        "quality_score": 0,
                    }
                )

        df = pd.DataFrame(results)

        # Run Phoenix LLM evaluations if available
        if PHOENIX_AVAILABLE and self.evaluation_model:
            df = self._run_phoenix_evaluations(df)

        return df

    def _clean_response(self, response: str) -> str:
        """Clean response by removing debug information and excessive repetition."""
        # Remove common debug prefixes
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # Handle excessive repetition
        sentences = response.split(". ")
        if len(sentences) > 3:
            first_sentence = sentences[0] + "."
            count = response.count(first_sentence)

            if count > 2:
                first_occurrence = response.find(first_sentence)
                if first_occurrence != -1:
                    second_occurrence = response.find(
                        first_sentence, first_occurrence + len(first_sentence)
                    )
                    if second_occurrence != -1:
                        response = response[:second_occurrence].strip()
                        last_period = response.rfind(".")
                        if last_period != -1:
                            response = response[: last_period + 1]

        # Remove trailing incomplete sentences
        if response.endswith(" The ") or response.endswith(" However, "):
            last_period = response.rfind(".")
            if last_period != -1:
                response = response[: last_period + 1]

        return response.strip()

    def _check_route_info(self, response: str) -> bool:
        """Check if response contains route information."""
        route_indicators = [
            "route", "highway", "interstate", "road", "miles", "distance",
            "drive", "travel", "direction", "via", "through", "along",
        ]
        return any(indicator in response.lower() for indicator in route_indicators)

    def _check_distance_info(self, response: str) -> bool:
        """Check if response contains distance information."""
        distance_indicators = ["miles", "km", "kilometers", "distance", "far"]
        return any(indicator in response.lower() for indicator in distance_indicators)

    def _check_travel_time(self, response: str) -> bool:
        """Check if response contains travel time information."""
        time_indicators = ["hours", "minutes", "time", "duration", "takes"]
        return any(indicator in response.lower() for indicator in time_indicators)

    def _check_appropriate_length(self, response: str) -> bool:
        """Check if response has appropriate length."""
        return 50 <= len(response) <= 1000

    def _check_relevance(self, query: str, response: str) -> bool:
        """Check if response is relevant to the query."""
        query_terms = query.lower().split()
        response_lower = response.lower()
        matching_terms = sum(1 for term in query_terms if term in response_lower)
        return matching_terms >= max(1, len(query_terms) * 0.3)

    def _calculate_quality_score(
        self, has_route_info: bool, has_distance_info: bool, has_travel_time: bool,
        appropriate_length: bool, is_relevant: bool,
    ) -> float:
        """Calculate overall quality score (0-10)."""
        score = 0
        if has_route_info: score += 2
        if has_distance_info: score += 2
        if has_travel_time: score += 2
        if appropriate_length: score += 2
        if is_relevant: score += 2
        return score

    def _run_phoenix_evaluations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Phoenix LLM evaluations (Relevance + QA)."""
        try:
            logger.info("🧠 Running Phoenix LLM evaluations...")

            # Prepare evaluation data
            eval_data = []
            for _, row in df.iterrows():
                eval_data.append({
                    "input": row["query"],
                    "output": row["response"],
                    "reference": self._get_reference_answer(row["query"]),
                })

            eval_df = pd.DataFrame(eval_data)

            # Run relevance evaluation
            logger.info("  🔍 Evaluating relevance...")
            relevance_results = llm_classify(
                data=eval_df,
                model=self.evaluation_model,
                template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                rails=["relevant", "irrelevant"],
                provide_explanation=True,
            )

            # Run QA evaluation
            logger.info("  🧠 Evaluating QA correctness...")
            qa_results = llm_classify(
                data=eval_df,
                model=self.evaluation_model,
                template=QA_PROMPT_TEMPLATE,
                rails=["correct", "incorrect"],
                provide_explanation=True,
            )

            # Add results to dataframe
            df["phoenix_relevance"] = relevance_results["label"]
            df["phoenix_relevance_explanation"] = relevance_results["explanation"]
            df["phoenix_qa_correctness"] = qa_results["label"]
            df["phoenix_qa_explanation"] = qa_results["explanation"]

            logger.info("✅ Phoenix evaluations completed")

        except Exception as e:
            logger.error(f"❌ Phoenix evaluation failed: {e}")

        return df

    def _get_reference_answer(self, query: str) -> str:
        """Get reference answer for QA evaluation."""
        # Simple reference answers based on query type
        query_lower = query.lower()
        
        if "new york" in query_lower and "boston" in query_lower:
            return "Route from New York to Boston is approximately 215 miles via I-95, taking about 4-5 hours by car."
        elif "san francisco" in query_lower and "los angeles" in query_lower:
            return "Distance from San Francisco to Los Angeles is about 380-400 miles, taking 6-8 hours by car via Highway 101 or PCH."
        elif "california" in query_lower and "scenic" in query_lower:
            return "Scenic routes in California include Pacific Coast Highway (PCH), Napa Valley wine country, and Highway 1 through Big Sur."
        elif "miami" in query_lower and "new york" in query_lower:
            return "Flight distance from Miami to New York is approximately 1,090 miles, taking about 2.5-3 hours by air."
        elif "colorado" in query_lower and "mountain" in query_lower:
            return "Mountain routes in Colorado include Denver to Aspen (160 miles), I-70 through the Rockies, and scenic drives through Rocky Mountain National Park."
        elif "chicago" in query_lower and "detroit" in query_lower:
            return "Train travel from Chicago to Detroit is approximately 280 miles, taking about 5-6 hours by train."
        else:
            return "This query requires route or distance information based on the available tools."

    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary."""
        total_queries = len(df)
        successful_queries = len(df[df["quality_score"] > 0])
        avg_quality = df["quality_score"].mean()

        print("\n" + "=" * 80)
        print("📊 ARIZE PHOENIX EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\n📈 Overall Results:")
        print(f"   Total queries: {total_queries}")
        print(f"   Successful responses: {successful_queries}/{total_queries}")
        print(f"   Success rate: {successful_queries / total_queries * 100:.1f}%")
        print(f"   Average quality score: {avg_quality:.1f}/10")

        print(f"\n📋 Content Analysis:")
        print(f"   Responses with route info: {df['has_route_info'].sum()}/{total_queries}")
        print(f"   Responses with distance info: {df['has_distance_info'].sum()}/{total_queries}")
        print(f"   Responses with travel time: {df['has_travel_time'].sum()}/{total_queries}")
        print(f"   Appropriate length: {df['appropriate_length'].sum()}/{total_queries}")
        print(f"   Relevant responses: {df['is_relevant'].sum()}/{total_queries}")

        if "phoenix_relevance" in df.columns:
            relevance_counts = df["phoenix_relevance"].value_counts()
            print(f"\n🔍 Phoenix LLM Evaluations:")
            print(f"   Relevance: {dict(relevance_counts)}")
            
            if "phoenix_qa_correctness" in df.columns:
                qa_counts = df["phoenix_qa_correctness"].value_counts()
                print(f"   QA Correctness: {dict(qa_counts)}")

        if self.phoenix_session:
            print(f"\n🔗 Phoenix UI: http://localhost:6006/")
            print("   View detailed traces and evaluation results in Phoenix")

        print("=" * 80)

        # Save results
        output_file = "phoenix_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"💾 Results saved to {output_file}")


def main():
    """Main evaluation function."""
    print("🚀 Starting Route Planner Agent Evaluation with Arize Phoenix")
    print("=" * 80)

    # Test queries focused on the 2 available tools
    test_queries = [
        "Find routes from New York to Boston",
        "Calculate the driving distance from San Francisco to Los Angeles",
        "What are scenic routes in California?",
        "How far is it to fly from Miami to New York?",
        "Find mountain routes in Colorado",
        "Calculate train travel time from Chicago to Detroit",
    ]

    # Initialize evaluator
    evaluator = RouteEvaluator()

    try:
        # Setup agent
        if not evaluator.setup_agent():
            print("❌ Could not setup agent. Exiting.")
            return

        # Run evaluation
        results_df = evaluator.run_evaluation(test_queries)

        # Print summary
        evaluator.print_summary(results_df)

        # Keep Phoenix UI running
        if evaluator.phoenix_session:
            print(f"\n🌟 Phoenix UI is running at: http://localhost:6006/")
            print("Keep this window open to explore traces and evaluations.")
            print("\nPress Enter to close Phoenix session and exit...")
            input()

        print("✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}")
    finally:
        # Cleanup Phoenix session
        if hasattr(evaluator, "cleanup_phoenix"):
            evaluator.cleanup_phoenix()


if __name__ == "__main__":
    main()
