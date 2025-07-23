#!/usr/bin/env python3
"""
Arize Phoenix Integration for Landmark Search Agent

This script demonstrates how to use Arize Phoenix to evaluate the landmark search agent
that uses LlamaIndex with Couchbase vector store and travel-sample.inventory.landmark data.

Features:
- Phoenix UI for trace visualization
- LLM-based evaluation with Phoenix evaluators (Relevance, QA, Hallucination, Toxicity)
- Integration with actual landmark search agent
- Comprehensive evaluation metrics with landmark-specific checks
"""

import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Path-related imports and setup - keep these at the top for sys.path modification
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Environment setup
load_dotenv(dotenv_path=os.path.join(parent_dir, "../../.env"))
load_dotenv(dotenv_path=os.path.join(parent_dir, ".env"), override=True)

# Configure logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import Phoenix/Arize dependencies with proper fallback
try:
    import phoenix as px
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.evals import (
        HALLUCINATION_PROMPT_RAILS_MAP,
        HALLUCINATION_PROMPT_TEMPLATE,
        QA_PROMPT_RAILS_MAP,
        QA_PROMPT_TEMPLATE,
        RAG_RELEVANCY_PROMPT_RAILS_MAP,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        TOXICITY_PROMPT_RAILS_MAP,
        TOXICITY_PROMPT_TEMPLATE,
        OpenAIModel,
        llm_classify,
    )
    from phoenix.otel import register

    PHOENIX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Phoenix not available: {e}")
    PHOENIX_AVAILABLE = False

# Try to import Arize datasets client
try:
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.utils.constants import GENERATIVE

    ARIZE_DATASETS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Arize datasets not available: {e}")
    ARIZE_DATASETS_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation system."""

    # Arize Configuration
    arize_space_id: str = os.getenv("ARIZE_SPACE_ID", "default-space")
    arize_api_key: str = os.getenv("ARIZE_API_KEY", "")
    project_name: str = "landmark-search-agent-evaluation"

    # Phoenix Configuration
    phoenix_base_port: int = 6006
    phoenix_max_port_attempts: int = 5

    # Evaluation Configuration
    evaluator_model: str = "gpt-4o"
    max_queries: int = 10
    evaluation_timeout: int = 300


class PhoenixManager:
    """Manages Phoenix server lifecycle."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.session = None
        self.active_port = None

    def start_phoenix(self) -> bool:
        """Start Phoenix server and return success status."""
        if not PHOENIX_AVAILABLE:
            logger.warning("⚠️ Phoenix dependencies not available")
            return False

        try:
            logger.info("🔧 Setting up Phoenix observability...")

            # Start Phoenix session
            self.session = px.launch_app()

            if self.session:
                logger.info(f"🌐 Phoenix UI: {self.session.url}")
                self.active_port = self.config.phoenix_base_port

            # Register Phoenix OTEL for LlamaIndex
            register(
                project_name=self.config.project_name,
                endpoint=f"http://localhost:{self.active_port}/v1/traces",
            )

            # Instrument LlamaIndex specifically
            LlamaIndexInstrumentor().instrument()

            logger.info("✅ Phoenix setup completed successfully")
            return True

        except Exception as e:
            logger.exception(f"❌ Phoenix setup failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Phoenix resources."""
        try:
            if self.session:
                # Phoenix session cleanup happens automatically
                pass
            logger.info("🔒 Phoenix cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during Phoenix cleanup: {e}")


class LandmarkSearchEvaluator:
    """
    LlamaIndex-specific evaluator for the landmark search agent.

    This evaluator is designed specifically for LlamaIndex ReActAgent:
    - Uses .chat() method for agent invocation
    - Handles LlamaIndex response structure (.response, .source_nodes)
    - Integrates with Phoenix for LlamaIndex tracing
    - Uses Phoenix evaluators for comprehensive assessment
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator with configuration."""
        self.config = config or EvaluationConfig()
        self.phoenix_manager = PhoenixManager(self.config)

        # Agent components
        self.agent = None
        self.client = None

        # Phoenix evaluators
        self.evaluator_llm = None

        if PHOENIX_AVAILABLE:
            self._setup_phoenix_evaluators()

    def _setup_phoenix_evaluators(self) -> None:
        """Setup Phoenix evaluators for LLM-based evaluation."""
        try:
            self.evaluator_llm = OpenAIModel(model=self.config.evaluator_model)
            logger.info("✅ Phoenix evaluators initialized")

            # Start Phoenix
            if self.phoenix_manager.start_phoenix():
                logger.info("✅ Phoenix instrumentation enabled for LlamaIndex")

        except Exception as e:
            logger.warning(f"⚠️ Phoenix evaluators setup failed: {e}")
            self.evaluator_llm = None

    def setup_agent(self) -> bool:
        """Setup landmark search agent using main.py setup function."""
        try:
            logger.info("🔧 Setting up landmark search agent...")

            # Import and setup agent from main.py
            from main import setup_landmark_agent

            self.agent, self.client = setup_landmark_agent()

            logger.info("✅ Landmark search agent setup completed successfully")
            return True

        except Exception as e:
            logger.exception(f"❌ Error setting up landmark search agent: {e}")
            return False

    def _extract_response_content(self, result: Any) -> str:
        """Extract clean response content from LlamaIndex agent result."""
        try:
            # LlamaIndex AgentChatResponse structure
            if hasattr(result, "response"):
                return str(result.response)
            elif hasattr(result, "message"):
                return str(result.message)
            else:
                return str(result)
        except Exception as e:
            logger.warning(f"Error extracting response content: {e}")
            return f"Error extracting response: {e}"

    def _extract_source_nodes(self, result: Any) -> List[str]:
        """Extract source nodes from LlamaIndex response."""
        try:
            if hasattr(result, "source_nodes") and result.source_nodes:
                return [node.text for node in result.source_nodes if hasattr(node, "text")]
            return []
        except Exception as e:
            logger.warning(f"Error extracting source nodes: {e}")
            return []

    def run_single_evaluation(self, query: str) -> Dict[str, Any]:
        """Run evaluation for a single query using LlamaIndex agent."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call setup_agent() first.")

        logger.info(f"🔍 Evaluating query: {query}")

        start_time = time.time()

        try:
            # Use LlamaIndex .chat() method (not .invoke() like LangChain)
            result = self.agent.chat(query, chat_history=[])

            # Extract response content and sources
            response = self._extract_response_content(result)
            sources = self._extract_source_nodes(result)

            # Create evaluation result
            evaluation_result = {
                "query": query,
                "response": response,
                "execution_time": time.time() - start_time,
                "success": True,
                "sources": sources,
                "num_sources": len(sources),
            }

            logger.info(f"✅ Query completed in {evaluation_result['execution_time']:.2f}s")
            logger.info(f"📊 Retrieved {len(sources)} source documents")

            return evaluation_result

        except Exception as e:
            logger.exception(f"❌ Query failed: {e}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "execution_time": time.time() - start_time,
                "success": False,
                "error": str(e),
                "sources": [],
                "num_sources": 0,
            }

    def run_phoenix_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Run Phoenix evaluations on the results."""
        if not PHOENIX_AVAILABLE or not self.evaluator_llm:
            logger.warning("⚠️ Phoenix evaluators not available - skipping LLM evaluations")
            return results_df

        logger.info(f"🧠 Running Phoenix evaluations on {len(results_df)} responses...")
        logger.info("📋 Evaluation criteria:")
        logger.info("   🔍 Relevance: Does the response address the landmark search query?")
        logger.info("   🎯 QA Correctness: Is the landmark information accurate?")
        logger.info("   🚨 Hallucination: Does the response contain fabricated information?")
        logger.info("   ☠️ Toxicity: Is the response harmful or inappropriate?")

        try:
            # Prepare evaluation data
            evaluation_data = []
            for _, row in results_df.iterrows():
                query = row["query"]
                response = row["response"]

                # Get reference answer for this query
                reference = self._get_reference_answer(str(query))

                evaluation_data.append(
                    {
                        "input": query,
                        "output": response,
                        "reference": reference,
                        "context": "; ".join(row["sources"][:3])
                        if row["sources"]
                        else "No context",
                    }
                )

            eval_df = pd.DataFrame(evaluation_data)

            # Run individual Phoenix evaluations
            self._run_individual_phoenix_evaluations(eval_df, results_df)

            logger.info("✅ Phoenix evaluations completed")

        except Exception as e:
            logger.exception(f"❌ Error running Phoenix evaluations: {e}")
            # Add error indicators
            for eval_type in ["relevance", "qa_correctness", "hallucination", "toxicity"]:
                results_df[eval_type] = "error"
                results_df[f"{eval_type}_explanation"] = f"Error: {e}"

        return results_df

    def _get_reference_answer(self, query: str) -> str:
        """Get reference answer for evaluation."""
        try:
            from data.queries import get_reference_answer

            reference_answer = get_reference_answer(query)

            if reference_answer.startswith("No reference answer available"):
                # Create a basic reference based on query
                if "museum" in query.lower() or "gallery" in query.lower():
                    return "Should provide information about museums and galleries with accurate names, addresses, and descriptions."
                elif "restaurant" in query.lower() or "food" in query.lower():
                    return "Should provide information about restaurants and food establishments."
                else:
                    return "Should provide relevant and accurate landmark information."

            return reference_answer

        except Exception as e:
            logger.warning(f"Could not get reference answer for '{query}': {e}")
            return "Should provide relevant and accurate landmark information."

    def _run_individual_phoenix_evaluations(
        self, eval_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> None:
        """Run individual Phoenix evaluations."""
        evaluations = {
            "relevance": {
                "template": RAG_RELEVANCY_PROMPT_TEMPLATE,
                "rails": list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values()),
                "data_cols": ["input", "reference"],
            },
            "qa_correctness": {
                "template": QA_PROMPT_TEMPLATE,
                "rails": list(QA_PROMPT_RAILS_MAP.values()),
                "data_cols": ["input", "output", "reference"],
            },
            "hallucination": {
                "template": HALLUCINATION_PROMPT_TEMPLATE,
                "rails": list(HALLUCINATION_PROMPT_RAILS_MAP.values()),
                "data_cols": ["input", "reference", "output"],
            },
            "toxicity": {
                "template": TOXICITY_PROMPT_TEMPLATE,
                "rails": list(TOXICITY_PROMPT_RAILS_MAP.values()),
                "data_cols": ["input"],
            },
        }

        for eval_name, eval_config in evaluations.items():
            try:
                logger.info(f"   📊 Running {eval_name} evaluation...")

                # Prepare data for this evaluator
                data = eval_df[eval_config["data_cols"]].copy()

                # Run evaluation
                eval_results = llm_classify(
                    data=data,
                    model=self.evaluator_llm,
                    template=eval_config["template"],
                    rails=eval_config["rails"],
                    provide_explanation=True,
                )

                # Process results
                self._process_evaluation_results(eval_results, eval_name, results_df)

            except Exception as e:
                logger.warning(f"⚠️ {eval_name} evaluation failed: {e}")
                results_df[eval_name] = "error"
                results_df[f"{eval_name}_explanation"] = f"Error: {e}"

    def _process_evaluation_results(
        self, eval_results: Any, eval_name: str, results_df: pd.DataFrame
    ) -> None:
        """Process evaluation results and add to results DataFrame."""
        try:
            if eval_results is None:
                logger.warning(f"⚠️ {eval_name} evaluation returned None")
                results_df[eval_name] = "unknown"
                results_df[f"{eval_name}_explanation"] = "Evaluation returned None"
                return

            # Handle DataFrame results (most common case)
            if hasattr(eval_results, "columns"):
                if "label" in eval_results.columns:
                    results_df[eval_name] = eval_results["label"].tolist()
                elif "classification" in eval_results.columns:
                    results_df[eval_name] = eval_results["classification"].tolist()
                else:
                    results_df[eval_name] = "unknown"

                if "explanation" in eval_results.columns:
                    results_df[f"{eval_name}_explanation"] = eval_results["explanation"].tolist()
                else:
                    results_df[f"{eval_name}_explanation"] = "No explanation provided"

                logger.info(f"   ✅ {eval_name} evaluation completed")

            else:
                logger.warning(f"⚠️ {eval_name} evaluation returned unexpected format")
                results_df[eval_name] = "unknown"
                results_df[f"{eval_name}_explanation"] = f"Unexpected format: {type(eval_results)}"

        except Exception as e:
            logger.warning(f"⚠️ Error processing {eval_name} results: {e}")
            results_df[eval_name] = "error"
            results_df[f"{eval_name}_explanation"] = f"Processing error: {e}"

    def run_evaluation(self, queries: List[str]) -> pd.DataFrame:
        """Run complete evaluation pipeline."""
        if not self.setup_agent():
            raise RuntimeError("Failed to setup agent")

        # Limit queries if specified
        if len(queries) > self.config.max_queries:
            queries = queries[: self.config.max_queries]
            logger.info(f"Limited to {self.config.max_queries} queries for evaluation")

        logger.info(
            f"🚀 Starting LlamaIndex landmark search evaluation with {len(queries)} queries"
        )
        logger.info("📋 Evaluation Configuration:")
        logger.info(f"   🤖 Agent: Landmark Search Agent (LlamaIndex)")
        logger.info(f"   🔧 Phoenix Available: {'✅' if PHOENIX_AVAILABLE else '❌'}")
        logger.info(f"   📊 Arize Datasets: {'✅' if ARIZE_DATASETS_AVAILABLE else '❌'}")

        # Run queries
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n📋 Query {i}/{len(queries)}")
            result = self.run_single_evaluation(query)
            results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Run Phoenix evaluations
        results_df = self.run_phoenix_evaluations(results_df)

        # Log summary
        self._log_evaluation_summary(results_df)

        # Create Arize dataset if available
        if ARIZE_DATASETS_AVAILABLE:
            self._create_arize_dataset(results_df)

        return results_df

    def _log_evaluation_summary(self, results_df: pd.DataFrame) -> None:
        """Log evaluation summary."""
        logger.info("\n📊 Evaluation Summary:")
        logger.info(f"  Total queries: {len(results_df)}")
        logger.info(f"  Successful executions: {results_df['success'].sum()}")
        logger.info(f"  Failed executions: {(~results_df['success']).sum()}")
        logger.info(f"  Average execution time: {results_df['execution_time'].mean():.2f}s")
        logger.info(f"  Average sources per query: {results_df['num_sources'].mean():.1f}")

        # Phoenix evaluation results
        if PHOENIX_AVAILABLE and self.evaluator_llm:
            self._format_evaluation_results(results_df)

        # Sample results with FULL detailed explanations for debugging
        if len(results_df) > 0:
            logger.info("\n📝 DETAILED EVALUATION RESULTS (FULL EXPLANATIONS):")
            logger.info("="*80)
            for i in range(min(len(results_df), len(results_df))):  # Show all results, not just 3
                row = results_df.iloc[i]
                logger.info(f"\n🔍 QUERY {i+1}: {row['query']}")
                logger.info("-"*60)
                logger.info(f"📄 RESPONSE: {str(row['response'])[:200]}...")

                for eval_type in ["relevance", "qa_correctness", "hallucination", "toxicity"]:
                    if eval_type in row and row[eval_type] != "error":
                        result = row[eval_type]
                        # Show FULL explanation instead of truncated version
                        full_explanation = str(row.get(f"{eval_type}_explanation", "No explanation provided"))
                        emoji = self._get_result_emoji(eval_type, result)
                        logger.info(f"\n📊 {eval_type.upper()}: {result}")
                        logger.info(f"💭 FULL REASONING:")
                        logger.info(f"{full_explanation}")
                        logger.info("-"*40)
                logger.info("="*80)

    def _get_result_emoji(self, eval_type: str, result: str) -> str:
        """Get appropriate emoji for evaluation result."""
        good_results = {
            "relevance": ["relevant"],
            "qa_correctness": ["correct"],
            "hallucination": ["factual"],
            "toxicity": ["non-toxic"],
        }

        if result in good_results.get(eval_type, []):
            return "✅"
        elif result == "error":
            return "❌"
        else:
            return "⚠️"

    def _format_evaluation_results(self, results_df: pd.DataFrame) -> None:
        """Format evaluation results in a user-friendly way."""
        print("\n" + "=" * 50)
        print("🏛️ LANDMARK SEARCH EVALUATION RESULTS")
        print("=" * 50)

        total_queries = len(results_df)

        evaluation_metrics = {
            "relevance": {
                "name": "🔍 Relevance",
                "description": "Does the response address the landmark query?",
                "good_values": ["relevant"],
            },
            "qa_correctness": {
                "name": "🎯 QA Correctness",
                "description": "Is the landmark information accurate?",
                "good_values": ["correct"],
            },
            "hallucination": {
                "name": "🚨 Hallucination",
                "description": "Does the response contain fabricated info?",
                "good_values": ["factual"],
            },
            "toxicity": {
                "name": "☠️ Toxicity",
                "description": "Is the response harmful or inappropriate?",
                "good_values": ["non-toxic"],
            },
        }

        for metric_name, metric_info in evaluation_metrics.items():
            if metric_name in results_df.columns:
                print(f"\n{metric_info['name']}: {metric_info['description']}")
                print("-" * 40)

                value_counts = results_df[metric_name].value_counts()

                for category, count in value_counts.items():
                    percentage = (count / total_queries) * 100

                    if category in metric_info["good_values"]:
                        status = "✅"
                    elif category == "error":
                        status = "❌"
                    else:
                        status = "⚠️"

                    print(
                        f"  {status} {category.title()}: {count}/{total_queries} ({percentage:.1f}%)"
                    )

        print("\n" + "=" * 50)

        # Performance metrics
        successful_queries = results_df["success"].sum()
        print(f"\n⚡ Performance: {successful_queries}/{total_queries} queries successful")
        print(f"📊 Average response time: {results_df['execution_time'].mean():.2f}s")
        print(f"🔗 Average sources retrieved: {results_df['num_sources'].mean():.1f}")

        print("=" * 50)

    def _create_arize_dataset(self, results_df: pd.DataFrame) -> Optional[str]:
        """Create an Arize dataset from evaluation results."""
        try:
            if not ARIZE_DATASETS_AVAILABLE or not self.config.arize_api_key:
                logger.info("⚠️ Arize datasets not available - skipping dataset creation")
                return None

            # Initialize Arize client
            client = ArizeDatasetsClient(api_key=self.config.arize_api_key)

            # Create dataset name
            dataset_name = f"landmark-search-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            logger.info("📊 Creating Arize dataset...")
            dataset_id = client.create_dataset(
                space_id=self.config.arize_space_id,
                dataset_name=dataset_name,
                dataset_type=GENERATIVE,
                data=results_df,
            )

            if dataset_id:
                logger.info(f"✅ Arize dataset created: {dataset_name} (ID: {dataset_id})")
                return dataset_id
            else:
                logger.warning("⚠️ Dataset creation returned None")
                return None

        except Exception as e:
            logger.warning(f"⚠️ Error creating Arize dataset: {e}")
            return None

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.phoenix_manager.cleanup()


def get_default_queries() -> List[str]:
    """Get default test queries for evaluation."""
    try:
        from data.queries import get_queries_for_evaluation

        return get_queries_for_evaluation(limit=10)
    except ImportError:
        # Fallback queries if import fails
        return [
            "Find museums and galleries in Glasgow",
            "Show me restaurants serving Asian cuisine",
            "What attractions can I see in Glasgow?",
            "Tell me about Monet's House",
            "Find places to eat in Gillingham",
        ]


def run_demo() -> pd.DataFrame:
    """Run a simple evaluation demo with a few queries."""
    logger.info("🔧 Running landmark search evaluation demo...")

    demo_queries = ["Find museums and galleries in Glasgow", "Tell me about Monet's House"]

    evaluator = LandmarkSearchEvaluator()
    try:
        results = evaluator.run_evaluation(demo_queries)
        logger.info("🎉 Evaluation demo complete!")
        if PHOENIX_AVAILABLE:
            logger.info("💡 Visit Phoenix UI to see detailed traces and evaluations")
        return results
    finally:
        evaluator.cleanup()


def main() -> pd.DataFrame:
    """Main evaluation function."""
    evaluator = LandmarkSearchEvaluator()
    try:
        queries = get_default_queries()
        results = evaluator.run_evaluation(queries)
        logger.info("\n✅ Landmark search evaluation complete!")
        return results
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    # Run demo mode for quick testing
    # Uncomment the next line to run demo mode instead of full evaluation
    # run_demo()

    # Run full evaluation
    main()
