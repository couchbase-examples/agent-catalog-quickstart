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

# Add parent directory to path to import main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import the refactored setup functions
from main import setup_flight_search_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import Phoenix/Arize dependencies
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
    evaluator_model: str = "gpt-4o-mini"
    batch_size: int = 10
    max_retries: int = 3
    evaluation_timeout: int = 300
    
    # Logging Configuration
    verbose_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize default values that can't be set in dataclass."""
        if self.verbose_modules is None:
            self.verbose_modules = [
                "httpx", "opentelemetry", "phoenix", "openai", "langchain", "agentc_core"
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
        
        raise RuntimeError(f"Could not find available ports after {self.config.phoenix_max_port_attempts} attempts")
    
    def start_phoenix(self) -> bool:
        """Start Phoenix server and return success status."""
        try:
            if not ARIZE_AVAILABLE:
                logger.warning("⚠️ Phoenix dependencies not available")
                return False
            
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
        if not self.tracer_provider:
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


class ArizeFlightSearchEvaluator:
    """
    Phoenix-based flight search agent evaluator with Arize AI integration.
    
    This class provides comprehensive evaluation capabilities using only
    Phoenix/Arize standard evaluators:
    - HallucinationEvaluator
    - QAEvaluator  
    - RelevanceEvaluator
    - ToxicityEvaluator
    
    No manual evaluation logic - relies entirely on Phoenix LLM-as-a-judge.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluator with configuration."""
        self.config = config or EvaluationConfig()
        self._setup_logging()
        
        # Initialize components
        self.phoenix_manager = PhoenixManager(self.config)
        
        # Agent components
        self.agent = None
        self.span = None
        
        # Phoenix evaluators
        self.evaluator_llm = None
        self.evaluators = {}
        
        if ARIZE_AVAILABLE:
            self._setup_phoenix_components()
    
    def _setup_logging(self) -> None:
        """Configure logging to suppress verbose modules."""
        if self.config.verbose_modules:
            for module in self.config.verbose_modules:
                logging.getLogger(module).setLevel(logging.WARNING)
    
    def _setup_phoenix_components(self) -> None:
        """Setup Phoenix evaluation components."""
        try:
            # Initialize OpenAI model for evaluations
            self.evaluator_llm = OpenAIModel(model=self.config.evaluator_model)
            
            # Initialize Phoenix evaluators
            self.evaluators = {
                "relevance": RelevanceEvaluator(self.evaluator_llm),
                "qa": QAEvaluator(self.evaluator_llm),
                "hallucination": HallucinationEvaluator(self.evaluator_llm),
                "toxicity": ToxicityEvaluator(self.evaluator_llm),
            }
            
            # Setup Phoenix server and instrumentation
            if self.phoenix_manager.start_phoenix():
                self.phoenix_manager.setup_instrumentation()
                
            logger.info("✅ Phoenix evaluators initialized successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ Phoenix components setup failed: {e}")
    
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
            if hasattr(result, 'messages') and result.messages:
                last_message = result.messages[-1]
                if hasattr(last_message, 'content'):
                    return str(last_message.content)
            
            if hasattr(result, 'search_results') and result.search_results:
                return str(result.search_results)
            
            return str(result)
        except Exception as e:
            return f"Error extracting response: {e}"
    
    def run_single_evaluation(self, query: str) -> Dict[str, Any]:
        """Run evaluation for a single query."""
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
            
            # Create evaluation result (no manual scoring)
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
                "error": str(e)
            }
    
    def run_evaluation(self, queries: List[str]) -> pd.DataFrame:
        """Run evaluation on multiple queries and return results."""
        if not self.setup_agent():
            raise RuntimeError("Failed to setup agent")
        
        logger.info(f"🚀 Starting evaluation with {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries, 1):
            logger.info(f"\n📋 Query {i}/{len(queries)}")
            result = self.run_single_evaluation(query)
            results.append(result)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Log basic summary
        self._log_basic_summary(results_df)
        
        # Run Phoenix evaluations if available
        if ARIZE_AVAILABLE and self.evaluators:
            results_df = self._run_phoenix_evaluations(results_df)
        else:
            logger.warning("⚠️ Phoenix evaluators not available - skipping LLM evaluations")
        
        return results_df
    
    def _run_phoenix_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Run Phoenix-based LLM evaluations using standard evaluators."""
        logger.info("🧠 Running Phoenix LLM evaluations...")
        logger.info("   📋 Evaluation criteria:")
        logger.info("      🔍 Relevance: Does the response address the flight search query?")
        logger.info("      🎯 QA: Is the flight information accurate and helpful?")
        logger.info("      🚨 Hallucination: Does the response contain fabricated information?")
        logger.info("      ☠️  Toxicity: Is the response harmful or inappropriate?")
        
        try:
            # Prepare evaluation data for Phoenix evaluators
            evaluation_data = self._prepare_evaluation_data(results_df)
            
            # Run Phoenix evaluations using run_evals
            logger.info("🔄 Running Phoenix evaluations...")
            
            # Use run_evals for comprehensive evaluation
            phoenix_results = run_evals(
                dataframe=evaluation_data,
                evaluators=list(self.evaluators.values()),
                provide_explanation=True,
                verbose=True
            )
            
            # Merge Phoenix results back into our DataFrame
            if isinstance(phoenix_results, list) and len(phoenix_results) > 0:
                # If it's a list of DataFrames, use the first one
                results_df = self._merge_phoenix_results(results_df, phoenix_results[0])
            elif isinstance(phoenix_results, pd.DataFrame):
                # If it's a single DataFrame
                results_df = self._merge_phoenix_results(results_df, phoenix_results)
            else:
                logger.warning("⚠️ Unexpected Phoenix results format")
            
            logger.info("✅ Phoenix evaluations completed successfully")
            
        except Exception as e:
            logger.exception(f"❌ Phoenix evaluations failed: {e}")
            # Add default columns to indicate evaluation failure
            for eval_name in self.evaluators.keys():
                results_df[f"phoenix_{eval_name}"] = "evaluation_failed"
                results_df[f"phoenix_{eval_name}_explanation"] = f"Error: {str(e)}"
        
        return results_df
    
    def _prepare_evaluation_data(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Phoenix evaluators."""
        evaluation_data = []
        
        for _, row in results_df.iterrows():
            query = str(row["query"])
            response = str(row["response"])
            
            # Create reference context based on query type
            reference = self._generate_reference_context(query)
            
            evaluation_data.append({
                "input": query,
                "output": response,
                "reference": reference,
                "query": query,  # For hallucination evaluator
                "response": response,  # For hallucination evaluator
                "text": response,  # For toxicity evaluator
            })
        
        return pd.DataFrame(evaluation_data)
    
    def _generate_reference_context(self, query: str) -> str:
        """Generate reference context for evaluation based on query."""
        query_lower = query.lower()
        
        # Generate context based on query patterns
        if "jfk" in query_lower and "lax" in query_lower:
            return "A relevant response about flights from JFK to LAX with specific airline information and flight details"
        elif "book" in query_lower:
            return "A relevant response about flight booking with passenger details, dates, and confirmation information"
        elif "current" in query_lower or "my" in query_lower:
            return "A relevant response showing current flight bookings with booking IDs and details"
        elif "review" in query_lower or "service" in query_lower:
            return "A relevant response about airline service quality based on passenger reviews"
        else:
            return f"A helpful and accurate response about {query} with specific flight information"
    
    def _merge_phoenix_results(self, results_df: pd.DataFrame, phoenix_results: pd.DataFrame) -> pd.DataFrame:
        """Merge Phoenix evaluation results back into main DataFrame."""
        try:
            # Map Phoenix columns to our naming convention
            phoenix_columns = {
                "label": "phoenix_result",
                "score": "phoenix_score", 
                "explanation": "phoenix_explanation"
            }
            
            # Add Phoenix results for each evaluator
            for eval_name in self.evaluators.keys():
                # Look for evaluator-specific columns
                eval_prefix = f"{eval_name}_"
                
                for phoenix_col, our_col in phoenix_columns.items():
                    phoenix_col_name = f"{eval_prefix}{phoenix_col}"
                    our_col_name = f"phoenix_{eval_name}_{our_col.split('_')[-1]}"
                    
                    if phoenix_col_name in phoenix_results.columns:
                        results_df[our_col_name] = phoenix_results[phoenix_col_name]
                    elif phoenix_col in phoenix_results.columns:
                        results_df[our_col_name] = phoenix_results[phoenix_col]
            
            return results_df
            
        except Exception as e:
            logger.warning(f"⚠️ Error merging Phoenix results: {e}")
            return results_df
    
    def _log_basic_summary(self, results_df: pd.DataFrame) -> None:
        """Log basic evaluation summary."""
        logger.info("\n📊 Basic Evaluation Summary:")
        logger.info(f"  Total queries: {len(results_df)}")
        logger.info(f"  Successful: {results_df['success'].sum()}")
        logger.info(f"  Failed: {(~results_df['success']).sum()}")
        logger.info(f"  Average execution time: {results_df['execution_time'].mean():.2f}s")
        
        # Note: No manual quality scoring - relying on Phoenix evaluations
        logger.info("  Quality assessment: Will be provided by Phoenix evaluators")
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.phoenix_manager.cleanup()


def get_default_queries() -> List[str]:
    """Get default test queries for evaluation."""
    return [
        "Find flights from JFK to LAX",
        "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
        "Book an economy flight from JFK to MIA for next week, 1 passenger",
        "Show me my current flight bookings",
        "What do passengers say about IndiGo's service quality?",
    ]


def run_phoenix_demo() -> pd.DataFrame:
    """Run a simple Phoenix evaluation demo."""
    logger.info("🔧 Running Phoenix evaluation demo...")
    
    demo_queries = [
        "Find flights from JFK to LAX",
        "What do passengers say about IndiGo's service quality?"
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
    """Main evaluation function."""
    if not ARIZE_AVAILABLE:
        logger.error("❌ Phoenix/Arize dependencies not available. Please install them to run evaluations.")
        return pd.DataFrame()
    
    evaluator = ArizeFlightSearchEvaluator()
    try:
        results = evaluator.run_evaluation(get_default_queries())
        logger.info("\n✅ Evaluation complete!")
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phoenix_evaluation_results_{timestamp}.csv"
        results.to_csv(filename, index=False)
        logger.info(f"💾 Results saved to: {filename}")
        
        return results
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    # Run demo mode for quick testing
    # Uncomment the next line to run demo mode instead of full evaluation
    # run_phoenix_demo()
    
    # Run full evaluation
    main()
