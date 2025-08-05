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
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import agentc
import pandas as pd
import nest_asyncio

# Apply the patch to allow nested asyncio event loops
nest_asyncio.apply()


# Add parent directory to path to import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the refactored setup functions
from main import clear_bookings_and_reviews, setup_flight_search_agent

# Import lenient evaluation templates
from templates import (
    LENIENT_QA_PROMPT_TEMPLATE,
    LENIENT_HALLUCINATION_PROMPT_TEMPLATE,
    LENIENT_QA_RAILS,
    LENIENT_HALLUCINATION_RAILS
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import Arize dependencies with fallback
try:
    import phoenix as px
    from arize.experimental.datasets import ArizeDatasetsClient
    from arize.experimental.datasets.utils.constants import GENERATIVE
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.evals import (
        HALLUCINATION_PROMPT_RAILS_MAP,
        HALLUCINATION_PROMPT_TEMPLATE,
        QA_PROMPT_RAILS_MAP,
        QA_PROMPT_TEMPLATE,
        RAG_RELEVANCY_PROMPT_RAILS_MAP,
        RAG_RELEVANCY_PROMPT_TEMPLATE,
        TOXICITY_PROMPT_RAILS_MAP,
        TOXICITY_PROMPT_TEMPLATE,
        HallucinationEvaluator,
        OpenAIModel,
        QAEvaluator,
        RelevanceEvaluator,
        ToxicityEvaluator,
        llm_classify,
        run_evals,
    )
    from phoenix.otel import register

    ARIZE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Arize dependencies not available: {e}")
    logger.warning("Running in local evaluation mode only...")
    ARIZE_AVAILABLE = False


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation system."""

    # Arize Configuration
    arize_space_id: str = os.getenv("ARIZE_SPACE_ID", "your-space-id")
    arize_api_key: str = os.getenv("ARIZE_API_KEY", "your-api-key")
    project_name: str = "flight-search-agent-evaluation"

    # Phoenix Configuration
    phoenix_base_port: int = 6006
    phoenix_grpc_base_port: int = 4317
    phoenix_max_port_attempts: int = 5
    phoenix_startup_timeout: int = 30

    # Evaluation Configuration
    evaluator_model: str = "gpt-4o"
    batch_size: int = 10
    max_retries: int = 3
    evaluation_timeout: int = 300

    # Logging Configuration
    verbose_modules: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize default values that can't be set in dataclass."""
        if self.verbose_modules is None:
            self.verbose_modules = [
                "httpx",
                "opentelemetry",
                "phoenix",
                "openai",
                "langchain",
                "agentc_core",
            ]


class PhoenixManager:
    """Manages Phoenix server lifecycle and port management."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.session = None
        self.active_port = None
        self.tracer_provider = None

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    def _kill_existing_phoenix_processes(self) -> None:
        """Kill any existing Phoenix processes."""
        try:
            subprocess.run(["pkill", "-f", "phoenix"], check=False, capture_output=True)
            time.sleep(2)  # Wait for processes to terminate
        except Exception as e:
            logger.debug(f"Error killing Phoenix processes: {e}")

    def _find_available_port(self) -> Tuple[int, int]:
        """Find available ports for Phoenix."""
        phoenix_port = self.config.phoenix_base_port
        grpc_port = self.config.phoenix_grpc_base_port

        for _ in range(self.config.phoenix_max_port_attempts):
            if not self._is_port_in_use(phoenix_port):
                return phoenix_port, grpc_port
            phoenix_port += 1
            grpc_port += 1

        raise RuntimeError(
            f"Could not find available ports after {self.config.phoenix_max_port_attempts} attempts"
        )

    def start_phoenix(self) -> bool:
        """Start Phoenix server and return success status."""
        if not ARIZE_AVAILABLE:
            logger.warning("⚠️ Phoenix dependencies not available")
            return False

        try:
            logger.info("🔧 Setting up Phoenix observability...")

            # Clean up existing processes
            self._kill_existing_phoenix_processes()

            # Find available ports
            phoenix_port, grpc_port = self._find_available_port()

            # Set environment variables
            os.environ["PHOENIX_PORT"] = str(phoenix_port)
            os.environ["PHOENIX_GRPC_PORT"] = str(grpc_port)

            # Start Phoenix session
            self.session = px.launch_app()
            self.active_port = phoenix_port

            if self.session:
                logger.info(f"🌐 Phoenix UI: {self.session.url}")

            # Register Phoenix OTEL
            self.tracer_provider = register(
                project_name=self.config.project_name,
                endpoint=f"http://localhost:{phoenix_port}/v1/traces",
            )

            logger.info("✅ Phoenix setup completed successfully")
            return True

        except Exception as e:
            logger.exception(f"❌ Phoenix setup failed: {e}")
            return False

    def setup_instrumentation(self) -> bool:
        """Setup OpenTelemetry instrumentation."""
        if not self.tracer_provider or not ARIZE_AVAILABLE:
            return False

        try:
            instrumentors = [
                ("LangChain", LangChainInstrumentor),
                ("OpenAI", OpenAIInstrumentor),
            ]

            for name, instrumentor_class in instrumentors:
                try:
                    instrumentor = instrumentor_class()
                    instrumentor.instrument(tracer_provider=self.tracer_provider)
                    logger.info(f"✅ {name} instrumentation enabled")
                except Exception as e:
                    logger.warning(f"⚠️ {name} instrumentation failed: {e}")

            return True

        except Exception as e:
            logger.exception(f"❌ Instrumentation setup failed: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up Phoenix resources."""
        try:
            # Clean up environment variables
            for var in ["PHOENIX_PORT", "PHOENIX_GRPC_PORT"]:
                if var in os.environ:
                    del os.environ[var]

            logger.info("🔒 Phoenix cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during Phoenix cleanup: {e}")


class ArizeDatasetManager:
    """Manages Arize dataset creation and management."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = None
        self._setup_client()

    def _setup_client(self) -> None:
        """Setup Arize datasets client."""
        if not ARIZE_AVAILABLE:
            return

        if (
            self.config.arize_api_key != "your-api-key"
            and self.config.arize_space_id != "your-space-id"
        ):
            try:
                # Initialize with correct parameters - no space_id needed for datasets client
                self.client = ArizeDatasetsClient(
                    api_key=self.config.arize_api_key
                )
                logger.info("✅ Arize datasets client initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize Arize datasets client: {e}")
                self.client = None
        else:
            logger.warning("⚠️ Arize API credentials not configured")
            self.client = None

    def create_dataset(self, results_df: pd.DataFrame) -> Optional[str]:
        """Create Arize dataset from evaluation results."""
        if not self.client:
            logger.warning("⚠️ Arize client not available - skipping dataset creation")
            return None

        try:
            dataset_name = f"flight-search-evaluation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info("📊 Creating Arize dataset...")
            dataset_id = self.client.create_dataset(
                space_id=self.config.arize_space_id,
                dataset_name=dataset_name,
                dataset_type=GENERATIVE,
                data=results_df,
                convert_dict_to_json=True
            )
            
            if dataset_id:
                logger.info(f"✅ Arize dataset created successfully: {dataset_id}")
                return dataset_id
            else:
                logger.warning("⚠️ Dataset creation returned None")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating Arize dataset: {e}")
            return None


class ArizeFlightSearchEvaluator:
    """
    Streamlined flight search agent evaluator using only Arize Phoenix evaluators.

    This class provides comprehensive evaluation capabilities using:
    - Phoenix RelevanceEvaluator for response relevance
    - Phoenix QAEvaluator for correctness assessment
    - Phoenix HallucinationEvaluator for factual accuracy
    - Phoenix ToxicityEvaluator for safety assessment
    - No manual validation - Phoenix evaluators only
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator with configuration."""
        self.config = config or EvaluationConfig()
        self._setup_logging()

        # Initialize components
        self.phoenix_manager = PhoenixManager(self.config)
        self.dataset_manager = ArizeDatasetManager(self.config)

        # Agent components
        self.agent = None
        self.span = None

        # Phoenix evaluators
        self.evaluators = {}
        self.evaluator_llm = None

        if ARIZE_AVAILABLE:
            self._setup_phoenix_evaluators()

    def _setup_logging(self) -> None:
        """Configure logging to suppress verbose modules."""
        if self.config.verbose_modules:
            for module in self.config.verbose_modules:
                logging.getLogger(module).setLevel(logging.WARNING)

    def _setup_phoenix_evaluators(self) -> None:
        """Setup Phoenix evaluators with robust error handling."""
        if not ARIZE_AVAILABLE:
            logger.warning("⚠️ Phoenix dependencies not available - evaluations will be limited")
            return
            
        try:
            self.evaluator_llm = OpenAIModel(model=self.config.evaluator_model)

            # Initialize all Phoenix evaluators
            self.evaluators = {
                "relevance": RelevanceEvaluator(self.evaluator_llm),
                "qa_correctness": QAEvaluator(self.evaluator_llm),
                "hallucination": HallucinationEvaluator(self.evaluator_llm),
                "toxicity": ToxicityEvaluator(self.evaluator_llm),
            }

            logger.info("✅ Phoenix evaluators initialized successfully")
            logger.info(f"   🤖 Using evaluator model: {self.config.evaluator_model}")
            logger.info(f"   📊 Available evaluators: {list(self.evaluators.keys())}")

            # Setup Phoenix if available
            if self.phoenix_manager.start_phoenix():
                self.phoenix_manager.setup_instrumentation()

        except Exception as e:
            logger.warning(f"⚠️ Phoenix evaluators setup failed: {e}")
            logger.info("Continuing with basic evaluation metrics only...")
            self.evaluators = {}

    def setup_agent(self) -> bool:
        """Setup flight search agent using refactored main.py setup."""
        try:
            logger.info("🔧 Setting up flight search agent...")

            # Use the refactored setup function from main.py
            compiled_graph, application_span = setup_flight_search_agent()

            self.agent = compiled_graph
            self.span = application_span

            logger.info("✅ Flight search agent setup completed successfully")
            return True

        except Exception as e:
            logger.exception(f"❌ Error setting up flight search agent: {e}")
            return False

    def _extract_response_content(self, result: Any) -> str:
        """Extract clean response content from agent result."""
        try:
            if hasattr(result, "messages") and result.messages:
                last_message = result.messages[-1]
                if hasattr(last_message, "content"):
                    return str(last_message.content)

            if hasattr(result, "search_results") and result.search_results:
                return str(result.search_results)

            return str(result)
        except Exception as e:
            return f"Error extracting response: {e}"

    def run_single_evaluation(self, query: str) -> Dict[str, Any]:
        """Run evaluation for a single query - no manual validation."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call setup_agent() first.")

        logger.info(f"🔍 Evaluating query: {query}")

        start_time = time.time()

        try:
            # Build starting state and run query
            from main import FlightSearchGraph

            state = FlightSearchGraph.build_starting_state(query=query)
            result = self.agent.invoke(state)

            # Extract response content
            response = self._extract_response_content(result)

            # Create evaluation result - no manual scoring
            evaluation_result = {
                "query": query,
                "response": response,
                "execution_time": time.time() - start_time,
                "success": True,
            }

            logger.info(f"✅ Query completed in {evaluation_result['execution_time']:.2f}s")
            return evaluation_result

        except Exception as e:
            logger.exception(f"❌ Query failed: {e}")
            return {
                "query": query,
                "response": f"Error: {str(e)}",
                "execution_time": time.time() - start_time,
                "success": False,
                "error": str(e),
            }

    def run_phoenix_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Run Phoenix evaluations on the results."""
        if not ARIZE_AVAILABLE or not self.evaluators:
            logger.warning("⚠️ Phoenix evaluators not available - skipping evaluations")
            return results_df

        logger.info(f"🧠 Running Phoenix evaluations on {len(results_df)} responses...")
        logger.info("📋 Evaluation criteria:")
        logger.info("   🔍 Relevance: Does the response address the flight search query?")
        logger.info("   🎯 QA Correctness: Is the flight information accurate and helpful?")
        logger.info("   🚨 Hallucination: Does the response contain fabricated information?")
        logger.info("   ☠️ Toxicity: Is the response harmful or inappropriate?")

        try:
            # Prepare evaluation data
            evaluation_data = []
            for _, row in results_df.iterrows():
                query = row["query"]
                response = row["response"]

                # Create reference text based on query type
                reference = self._create_reference_text(str(query))

                evaluation_data.append(
                    {
                        "input": query,
                        "output": response,
                        "reference": reference,
                        "query": query,  # For hallucination evaluation
                        "response": response,  # For hallucination evaluation
                        "text": response,  # For toxicity evaluation
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

    def _create_reference_text(self, query: str) -> str:
        """Create reference text for evaluation based on query."""
        from data.queries import get_reference_answer

        # Get the actual reference answer for this query
        reference_answer = get_reference_answer(query)

        if reference_answer.startswith("No reference answer available"):
            raise ValueError(
                f"No reference answer available for query: '{query}'. "
                f"Please add this query to QUERY_REFERENCE_ANSWERS in data/queries.py"
            )

        return reference_answer

    def _run_individual_phoenix_evaluations(
        self, eval_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> None:
        """Run individual Phoenix evaluations."""
        for eval_name, evaluator in self.evaluators.items():
            try:
                logger.info(f"   📊 Running {eval_name} evaluation...")

                # Prepare data based on evaluator requirements
                if eval_name == "relevance":
                    data = eval_df[["input", "reference"]].copy()
                    eval_results = llm_classify(
                        data=data,
                        model=self.evaluator_llm,
                        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                        rails=list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values()),
                        provide_explanation=True,
                    )
                elif eval_name == "qa_correctness":
                    data = eval_df[["input", "output", "reference"]].copy()
                    eval_results = llm_classify(
                        data=data,
                        model=self.evaluator_llm,
                        template=LENIENT_QA_PROMPT_TEMPLATE,
                        rails=LENIENT_QA_RAILS,
                        provide_explanation=True,
                    )
                elif eval_name == "hallucination":
                    data = eval_df[["input", "reference", "output"]].copy()
                    eval_results = llm_classify(
                        data=data,
                        model=self.evaluator_llm,
                        template=LENIENT_HALLUCINATION_PROMPT_TEMPLATE,
                        rails=LENIENT_HALLUCINATION_RAILS,
                        provide_explanation=True,
                    )
                elif eval_name == "toxicity":
                    data = eval_df[["input"]].copy()
                    eval_results = llm_classify(
                        data=data,
                        model=self.evaluator_llm,
                        template=TOXICITY_PROMPT_TEMPLATE,
                        rails=list(TOXICITY_PROMPT_RAILS_MAP.values()),
                        provide_explanation=True,
                    )
                else:
                    logger.warning(f"⚠️ Unknown evaluator: {eval_name}")
                    continue

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

            # Handle DataFrame results
            if hasattr(eval_results, "columns"):
                if "label" in eval_results.columns:
                    results_df[eval_name] = eval_results["label"].tolist()
                elif "classification" in eval_results.columns:
                    results_df[eval_name] = eval_results["classification"].tolist()
                else:
                    results_df[eval_name] = "unknown"

                if "explanation" in eval_results.columns:
                    results_df[f"{eval_name}_explanation"] = eval_results["explanation"].tolist()
                elif "reason" in eval_results.columns:
                    results_df[f"{eval_name}_explanation"] = eval_results["reason"].tolist()
                else:
                    results_df[f"{eval_name}_explanation"] = "No explanation provided"

                logger.info(f"   ✅ {eval_name} evaluation completed")

            # Handle list results
            elif isinstance(eval_results, list) and len(eval_results) > 0:
                if isinstance(eval_results[0], dict):
                    results_df[eval_name] = [item.get("label", "unknown") for item in eval_results]
                    results_df[f"{eval_name}_explanation"] = [
                        item.get("explanation", "No explanation") for item in eval_results
                    ]
                else:
                    results_df[eval_name] = eval_results
                    results_df[f"{eval_name}_explanation"] = "List evaluation result"

                logger.info(f"   ✅ {eval_name} evaluation completed (list format)")

            else:
                logger.warning(f"⚠️ {eval_name} evaluation returned unexpected format")
                results_df[eval_name] = "unknown"
                results_df[f"{eval_name}_explanation"] = f"Unexpected format: {type(eval_results)}"

        except Exception as e:
            logger.warning(f"⚠️ Error processing {eval_name} results: {e}")
            results_df[eval_name] = "error"
            results_df[f"{eval_name}_explanation"] = f"Processing error: {e}"

    def run_evaluation(self, queries: List[str]) -> pd.DataFrame:
        """Run complete evaluation pipeline using only Phoenix evaluators."""
        # Clear existing bookings for a clean test run
        clear_bookings_and_reviews()
        
        if not self.setup_agent():
            raise RuntimeError("Failed to setup agent")

        logger.info(f"🚀 Starting evaluation with {len(queries)} queries")
        
        # Log available features
        logger.info("📋 Evaluation Configuration:")
        logger.info(f"   🤖 Agent: Flight Search Agent (LangGraph)")
        logger.info(f"   🔧 Phoenix Available: {'✅' if ARIZE_AVAILABLE else '❌'}")
        logger.info(f"   📊 Arize Datasets: {'✅' if ARIZE_AVAILABLE and (self.dataset_manager.client is not None) else '❌'}")
        if self.evaluators:
            logger.info(f"   🧠 Phoenix Evaluators: {list(self.evaluators.keys())}")
        else:
            logger.info("   🧠 Phoenix Evaluators: ❌ (basic metrics only)")

        # Run queries (no manual validation)
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n📋 Query {i}/{len(queries)}")
            result = self.run_single_evaluation(query)
            results.append(result)

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Run Phoenix evaluations only
        results_df = self.run_phoenix_evaluations(results_df)

        # Log summary
        self._log_evaluation_summary(results_df)

        # Create Arize dataset
        dataset_id = self.dataset_manager.create_dataset(results_df)
        if dataset_id:
            logger.info(f"📊 Arize dataset created: {dataset_id}")
        else:
            logger.warning("⚠️ Dataset creation failed")

        return results_df

    def _log_evaluation_summary(self, results_df: pd.DataFrame) -> None:
        """Log evaluation summary using Phoenix results only."""
        logger.info("\n📊 Phoenix Evaluation Summary:")
        logger.info(f"  Total queries: {len(results_df)}")
        logger.info(f"  Successful executions: {results_df['success'].sum()}")
        logger.info(f"  Failed executions: {(~results_df['success']).sum()}")
        logger.info(f"  Average execution time: {results_df['execution_time'].mean():.2f}s")

        # Phoenix evaluation results
        if ARIZE_AVAILABLE and self.evaluators:
            logger.info("\n🧠 Phoenix Evaluation Results:")
            for eval_type in ["relevance", "qa_correctness", "hallucination", "toxicity"]:
                if eval_type in results_df.columns:
                    counts = results_df[eval_type].value_counts()
                    logger.info(f"   {eval_type}: {dict(counts)}")

        # Quick scores summary
        if len(results_df) > 0:
            logger.info("\n📊 Quick Scores Summary:")
            for i in range(len(results_df)):
                row = results_df.iloc[i]
                scores = []
                for eval_type in ["relevance", "qa_correctness", "hallucination", "toxicity"]:
                    if eval_type in row:
                        result = row[eval_type]
                        emoji = "✅" if result in ["relevant", "correct", "factual", "non-toxic"] else "❌"
                        scores.append(f"{emoji} {eval_type}: {result}")
                
                logger.info(f"   Query {i+1}: {' | '.join(scores)}")

        # Sample results with FULL detailed explanations for debugging
        if len(results_df) > 0:
            logger.info("\n📝 DETAILED EVALUATION RESULTS (FULL EXPLANATIONS):")
            logger.info("="*80)
            for i in range(min(len(results_df), len(results_df))):  # Show all results
                row = results_df.iloc[i]
                logger.info(f"\n🔍 QUERY {i+1}: {row['query']}")
                logger.info("-"*60)

                for eval_type in ["relevance", "qa_correctness", "hallucination", "toxicity"]:
                    if eval_type in row:
                        result = row[eval_type]
                        # Show FULL explanation instead of processed/truncated version
                        full_explanation = str(row.get(f"{eval_type}_explanation", "No explanation provided"))
                        logger.info(f"\n📊 {eval_type.upper()}: {result}")
                        logger.info(f"💭 FULL REASONING:")
                        logger.info(f"{full_explanation}")
                        logger.info("-"*40)
                logger.info("="*80)

    def cleanup(self) -> None:
        """Clean up all resources."""
        self.phoenix_manager.cleanup()


def get_default_queries() -> List[str]:
    """Get default test queries for evaluation."""
    from data.queries import get_evaluation_queries

    return get_evaluation_queries()


def run_phoenix_demo() -> pd.DataFrame:
    """Run a simple Phoenix evaluation demo."""
    logger.info("🔧 Running Phoenix evaluation demo...")

    demo_queries = [
        "Find flights from JFK to LAX",
        "What do passengers say about SpiceJet's service quality?",
    ]

    evaluator = ArizeFlightSearchEvaluator()
    try:
        results = evaluator.run_evaluation(demo_queries)
        logger.info("🎉 Phoenix evaluation demo complete!")
        logger.info("💡 Visit Phoenix UI to see detailed traces and evaluations")
        return results
    finally:
        evaluator.cleanup()


def main() -> pd.DataFrame:
    """Main evaluation function using only Phoenix evaluators."""
    evaluator = ArizeFlightSearchEvaluator()
    try:
        results = evaluator.run_evaluation(get_default_queries())
        logger.info("\n✅ Phoenix evaluation complete!")
        return results
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    # Run demo mode for quick testing
    # Uncomment the next line to run demo mode instead of full evaluation
    # run_phoenix_demo()

    # Run full evaluation with Phoenix evaluators only
    main()
