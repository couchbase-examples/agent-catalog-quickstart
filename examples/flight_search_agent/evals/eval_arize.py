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
import json

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

# Import the refactored setup functions
from main import setup_environment, CouchbaseClient, FlightSearchGraph

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
        HALLUCINATION_PROMPT_RAILS_MAP,
        HALLUCINATION_PROMPT_TEMPLATE,
        TOXICITY_PROMPT_RAILS_MAP,
        TOXICITY_PROMPT_TEMPLATE,
        HallucinationEvaluator,
        ToxicityEvaluator,
        RelevanceEvaluator,
        QAEvaluator,
        OpenAIModel,
        llm_classify,
        run_evals,
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
            self.hallucination_rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())
            self.toxicity_rails = list(TOXICITY_PROMPT_RAILS_MAP.values())
        else:
            logger.warning("⚠️ Arize not available - running basic evaluation only")

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            logger.info("🔧 Setting up Arize observability...")

            # Check if Phoenix is already running and kill existing processes
            import subprocess
            import socket
            
            def is_port_in_use(port):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    return s.connect_ex(('localhost', port)) == 0
            
            # Kill any existing Phoenix processes
            try:
                subprocess.run(['pkill', '-f', 'phoenix'], capture_output=True)
                time.sleep(2)  # Wait for processes to terminate
            except:
                pass

            # Try alternative ports if 6006 is occupied
            phoenix_port = 6006
            grpc_port = 4317
            max_attempts = 5
            
            for attempt in range(max_attempts):
                try:
                    if is_port_in_use(phoenix_port):
                        phoenix_port += 1
                        grpc_port += 1
                        logger.info(f"🔄 Port {phoenix_port-1} in use, trying {phoenix_port}")
                        continue
                    
                    # Set environment variables for Phoenix ports
                    os.environ['PHOENIX_PORT'] = str(phoenix_port)
                    os.environ['PHOENIX_GRPC_PORT'] = str(grpc_port)
                    
                    # Start Phoenix session (using environment variables, removing deprecated port parameter)
                    self.phoenix_session = px.launch_app()
                    logger.info(f"🌐 Phoenix UI: {self.phoenix_session.url}")
                    break
                except Exception as e:
                    logger.warning(f"🔄 Phoenix startup attempt {attempt+1} failed: {e}")
                    if attempt == max_attempts - 1:
                        logger.error(f"💥 Phoenix failed to start after {max_attempts} attempts")
                        self.phoenix_session = None
                        return
                    phoenix_port += 1
                    grpc_port += 1

            if self.phoenix_session is None:
                logger.error("💥 Phoenix failed to start. Continuing with local evaluation only.")
                return

            # Register Phoenix OTEL and get tracer provider
            try:
                self.tracer_provider = register(
                    project_name=PROJECT_NAME,
                    endpoint=f"http://localhost:{phoenix_port}/v1/traces",
                )
                logger.info("✅ Phoenix OTEL registered successfully")
            except Exception as e:
                logger.warning(f"⚠️ Phoenix OTEL registration failed: {e}")
                self.tracer_provider = None

            # Instrument LangChain and OpenAI with new approach
            if self.tracer_provider:
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
            self.phoenix_session = None

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

            # Setup Couchbase infrastructure
            logger.info("🔧 Setting up Couchbase infrastructure...")
            client = CouchbaseClient(
                conn_string=os.environ['CB_CONN_STRING'],
                username=os.environ['CB_USERNAME'],
                password=os.environ['CB_PASSWORD'],
                bucket_name=os.environ['CB_BUCKET']
            )
            
            client.connect()
            client.setup_collection(os.environ['CB_SCOPE'], os.environ['CB_COLLECTION'])
            
            # Setup vector search index
            try:
                with open("agentcatalog_index.json", "r") as file:
                    index_definition = json.load(file)
                client.setup_vector_search_index(index_definition, os.environ['CB_SCOPE'])
            except Exception as e:
                logger.warning(f"⚠️ Could not setup vector search index: {e}")

            # Setup embeddings and vector store
            try:
                if (
                    os.getenv("CB_USERNAME")
                    and os.getenv("CB_PASSWORD")
                    and os.getenv("CAPELLA_API_ENDPOINT")
                    and os.getenv("CAPELLA_API_EMBEDDING_MODEL")
                ):
                    # Create API key for Capella AI
                    import base64
                    from langchain_openai import OpenAIEmbeddings
                    
                    api_key = base64.b64encode(
                        f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
                    ).decode()

                    # Use OpenAI embeddings client with Capella endpoint
                    embeddings = OpenAIEmbeddings(
                        model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                        api_key=api_key,
                        base_url=f"{os.getenv('CAPELLA_API_ENDPOINT')}/v1",
                    )
                    logger.info("✅ Using Capella AI for embeddings (4096 dimensions)")
                else:
                    raise ValueError("Capella AI credentials not available")
            except Exception as e:
                logger.warning(f"⚠️ Capella AI embeddings failed: {e}")
                # Fall back to OpenAI for evaluation purposes
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                logger.info("⚠️ Using OpenAI embeddings as fallback for evaluation")
            
            client.setup_vector_store(
                scope_name=os.environ['CB_SCOPE'],
                collection_name=os.environ['CB_COLLECTION'],
                index_name=os.environ['CB_INDEX'],
                embeddings=embeddings
            )
            logger.info("✅ Couchbase infrastructure setup completed")

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
        Run comprehensive Arize-based LLM evaluations on the results using Phoenix evaluators.

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            DataFrame with additional Arize evaluation columns
        """
        if not ARIZE_AVAILABLE:
            logger.warning("⚠️ Arize not available - skipping LLM evaluations")
            return results_df

        logger.info(f"🧠 Running Comprehensive Phoenix Evaluations on {len(results_df)} responses...")
        logger.info("   📋 Evaluation criteria:")
        logger.info("      🔍 Relevance: Does the response address the flight search query?")
        logger.info("      🎯 Correctness: Is the flight information accurate and helpful?")
        logger.info("      🚨 Hallucination: Does the response contain fabricated information?")
        logger.info("      ☠️  Toxicity: Is the response harmful or inappropriate?")

        # Sample evaluation data preview
        if len(results_df) > 0:
            sample_row = results_df.iloc[0]
            logger.info(f"\n   🔍 Sample evaluation data:")
            logger.info(f"      Query: {sample_row['query']}")
            logger.info(f"      Response: {sample_row['response'][:100]}...")

        try:
            # Prepare data for evaluation with improved reference texts
            evaluation_data = []
            for _, row in results_df.iterrows():
                query = row["query"]
                query_type = row['query_type']
                
                # Create more specific reference text based on query content
                if "jfk" in query.lower() and "lax" in query.lower():
                    reference = "A relevant response listing specific flights from JFK to LAX with airline codes like AS, B6, DL, QF, AA, UA, US, VX and aircraft details"
                elif "lax" in query.lower() and "jfk" in query.lower():
                    reference = "A relevant response listing specific flights from LAX to JFK with airline codes and aircraft details"
                elif "baggage" in query.lower() or "policy" in query.lower():
                    reference = "A relevant response providing specific airline baggage policies and fee information"
                elif "booking" in query.lower() or "current" in query.lower():
                    reference = "A relevant response showing current flight bookings with booking IDs, routes, dates, and prices"
                elif query_type == "flight_search":
                    reference = f"A relevant response about {query.lower()} with specific flight information including airline codes and aircraft details"
                else:
                    reference = f"A helpful and accurate response about {query_type.replace('_', ' ')} with specific information and no fabricated details"
                
                evaluation_data.append(
                    {
                        # Standard columns for all evaluations
                        "input": query,
                        "output": row["response"],
                        "reference": reference,
                        
                        # Specific columns for different evaluations
                        "query": query,  # For hallucination evaluation
                        "response": row["response"],  # For hallucination evaluation
                        "text": row["response"],  # For toxicity evaluation
                    }
                )

            eval_df = pd.DataFrame(evaluation_data)

            # Initialize evaluators with the model
            evaluators = {
                'relevance': RelevanceEvaluator(self.evaluator_llm),
                'qa': QAEvaluator(self.evaluator_llm),
                'hallucination': HallucinationEvaluator(self.evaluator_llm), 
                'toxicity': ToxicityEvaluator(self.evaluator_llm),
            }

            # Run comprehensive evaluations using Phoenix evaluators
            logger.info(f"\n   🧠 Running advanced Phoenix evaluations...")

            try:
                # Run individual evaluations with proper column mapping
                logger.info(f"   🔄 Running evaluations individually for better reliability...")
                self._run_individual_evaluations(eval_df, results_df, evaluators)

            except Exception as e:
                logger.warning(f"   ⚠️ Individual evaluations failed: {e}")
                # Set default values if all evaluations fail
                for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
                    results_df[f"arize_{eval_type}"] = "not_evaluated"
                    results_df[f"arize_{eval_type}_explanation"] = f"Evaluation failed: {e}"

            # Display sample evaluation results
            if len(results_df) > 0:
                logger.info(f"\n   📝 Sample evaluation results:")
                for i in range(min(2, len(results_df))):
                    row = results_df.iloc[i]
                    logger.info(f"      Query: {row['query']}")
                    
                    for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
                        eval_col = f"arize_{eval_type}"
                        explanation_col = f"arize_{eval_type}_explanation"
                        
                        if eval_col in row:
                            evaluation = row[eval_col]
                            explanation = str(row.get(explanation_col, "No explanation"))[:80] + "..."
                            logger.info(f"      {eval_type.title()}: {evaluation} - {explanation}")
                    logger.info("")

            logger.info(f"   ✅ All Phoenix evaluations completed")

        except Exception as e:
            logger.error(f"   ❌ Error running Phoenix evaluations: {e}")
            # Add default values if evaluation fails
            for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
                results_df[f"arize_{eval_type}"] = "unknown"
                results_df[f"arize_{eval_type}_explanation"] = f"Error: {e}"

        return results_df

    def _run_individual_evaluations(self, eval_df: pd.DataFrame, results_df: pd.DataFrame, evaluators: dict):
        """Run individual evaluations with proper column mapping and error handling."""
        logger.info(f"   🔄 Running individual evaluations...")

        for eval_name, evaluator in evaluators.items():
            try:
                logger.info(f"      📊 Running {eval_name} evaluation...")
                
                # Prepare data with proper column names for each evaluator
                if eval_name == 'relevance':
                    # Relevance evaluator expects 'input' and 'reference' columns
                    relevance_data = eval_df[["input", "reference"]].copy()
                    eval_results = llm_classify(
                        data=relevance_data,  # Fixed deprecated parameter name
                        model=self.evaluator_llm,
                        template=RAG_RELEVANCY_PROMPT_TEMPLATE,
                        rails=self.relevance_rails,
                        provide_explanation=True,
                    )
                elif eval_name == 'qa':
                    # QA evaluator expects 'input', 'output', and 'reference' columns
                    qa_data = eval_df[["input", "output", "reference"]].copy()
                    eval_results = llm_classify(
                        data=qa_data,  # Fixed deprecated parameter name
                        model=self.evaluator_llm,
                        template=QA_PROMPT_TEMPLATE,
                        rails=self.qa_rails,
                        provide_explanation=True,
                    )
                elif eval_name == 'hallucination':
                    # Hallucination evaluator expects 'input', 'reference', and 'output' columns
                    hallucination_data = eval_df[["input", "reference", "output"]].copy()
                    eval_results = llm_classify(
                        data=hallucination_data,  # Fixed deprecated parameter name
                        model=self.evaluator_llm,
                        template=HALLUCINATION_PROMPT_TEMPLATE,
                        rails=self.hallucination_rails,
                        provide_explanation=True,
                    )
                elif eval_name == 'toxicity':
                    # Toxicity evaluator expects 'input' column (renamed from 'text')
                    toxicity_data = eval_df[["input"]].copy()
                    eval_results = llm_classify(
                        data=toxicity_data,  # Fixed deprecated parameter name
                        model=self.evaluator_llm,
                        template=TOXICITY_PROMPT_TEMPLATE,
                        rails=self.toxicity_rails,
                        provide_explanation=True,
                    )
                else:
                    # Fallback for unknown evaluators
                    logger.warning(f"      ⚠️ Unknown evaluator {eval_name}, setting defaults")
                    results_df[f"arize_{eval_name}"] = "not_evaluated"
                    results_df[f"arize_{eval_name}_explanation"] = f"{eval_name} evaluation not implemented"
                    continue

                # Add results to our DataFrame with proper error handling
                if eval_results is not None:
                    # Handle both DataFrame and list return types
                    if hasattr(eval_results, 'columns'):
                        # DataFrame case
                        if 'label' in eval_results.columns:
                            results_df[f"arize_{eval_name}"] = eval_results['label'].tolist()
                        elif 'classification' in eval_results.columns:
                            results_df[f"arize_{eval_name}"] = eval_results['classification'].tolist()
                        else:
                            results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                        
                        if 'score' in eval_results.columns:
                            results_df[f"arize_{eval_name}_score"] = eval_results['score'].tolist()
                        
                        if 'explanation' in eval_results.columns:
                            results_df[f"arize_{eval_name}_explanation"] = eval_results['explanation'].tolist()
                        elif 'reason' in eval_results.columns:
                            results_df[f"arize_{eval_name}_explanation"] = eval_results['reason'].tolist()
                        else:
                            results_df[f"arize_{eval_name}_explanation"] = ["No explanation provided"] * len(results_df)
                        
                        logger.info(f"      ✅ {eval_name} evaluation completed with {len(eval_results)} results")
                    elif isinstance(eval_results, list):
                        # List case - extract values from list of dictionaries  
                        if len(eval_results) > 0 and isinstance(eval_results[0], dict):
                            # Extract labels/classifications
                            if 'label' in eval_results[0]:
                                results_df[f"arize_{eval_name}"] = [item.get('label', 'not_evaluated') for item in eval_results]
                            elif 'classification' in eval_results[0]:
                                results_df[f"arize_{eval_name}"] = [item.get('classification', 'not_evaluated') for item in eval_results]
                            else:
                                results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                            
                            # Extract scores if available
                            if 'score' in eval_results[0]:
                                results_df[f"arize_{eval_name}_score"] = [item.get('score', 0) for item in eval_results]
                            
                            # Extract explanations
                            if 'explanation' in eval_results[0]:
                                results_df[f"arize_{eval_name}_explanation"] = [item.get('explanation', 'No explanation') for item in eval_results]
                            elif 'reason' in eval_results[0]:
                                results_df[f"arize_{eval_name}_explanation"] = [item.get('reason', 'No explanation') for item in eval_results]
                            else:
                                results_df[f"arize_{eval_name}_explanation"] = ["No explanation provided"] * len(results_df)
                        else:
                            # List of simple values
                            results_df[f"arize_{eval_name}"] = eval_results if len(eval_results) == len(results_df) else ["not_evaluated"] * len(results_df)
                            results_df[f"arize_{eval_name}_explanation"] = ["List evaluation result"] * len(results_df)
                        
                        logger.info(f"      ✅ {eval_name} evaluation completed with {len(eval_results)} results (list format)")
                    else:
                        # Single value or unexpected format
                        logger.warning(f"      ⚠️ {eval_name} evaluation returned unexpected format: {type(eval_results)}")
                        results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                        results_df[f"arize_{eval_name}_explanation"] = [f"Unexpected format: {type(eval_results)}"] * len(results_df)
                else:
                    # None result
                    logger.warning(f"      ⚠️ {eval_name} evaluation returned None")
                    results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                    results_df[f"arize_{eval_name}_explanation"] = ["Evaluation returned None"] * len(results_df)

            except Exception as e:
                logger.warning(f"      ⚠️ {eval_name} evaluation failed: {e}")
                results_df[f"arize_{eval_name}"] = ["error"] * len(results_df)
                results_df[f"arize_{eval_name}_explanation"] = [f"Error: {str(e)}"] * len(results_df)

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
                        "arize_qa": row.get("arize_qa", "unknown"),
                        "arize_hallucination": row.get("arize_hallucination", "unknown"),
                        "arize_toxicity": row.get("arize_toxicity", "unknown"),
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
            logger.info(f"\n🔍 Arize LLM Evaluation Results:")
            logger.info(f"   📋 Relevance: {dict(relevance_counts)}")
            
            # Check for QA (correctness) results
            if "arize_qa" in results_df.columns:
                qa_counts = results_df["arize_qa"].value_counts()
                logger.info(f"   ✅ QA/Correctness: {dict(qa_counts)}")
            
            # Check for hallucination results
            if "arize_hallucination" in results_df.columns:
                hallucination_counts = results_df["arize_hallucination"].value_counts()
                logger.info(f"   🚨 Hallucination: {dict(hallucination_counts)}")
            
            # Check for toxicity results  
            if "arize_toxicity" in results_df.columns:
                toxicity_counts = results_df["arize_toxicity"].value_counts()
                logger.info(f"   ☠️  Toxicity: {dict(toxicity_counts)}")

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
        """Clean up resources and close Phoenix session."""
        try:
            # Clean up environment variables
            import os
            for var in ['PHOENIX_PORT', 'PHOENIX_GRPC_PORT']:
                if var in os.environ:
                    del os.environ[var]
                    
            logger.info("🔒 Phoenix cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Error during Phoenix cleanup: {e}")


def run_phoenix_demo():
    """Run a simple Phoenix evaluation demo for flight search agent (matches notebook functionality)."""
    logger.info("🔧 Running Phoenix evaluation demo for flight search agent...")
    
    # Demo queries - simple search only (no bookings or complex operations)
    demo_queries = [
        "Find flights from JFK to LAX",
        "What flights are available from Miami to New York?"
    ]
    
    evaluator = ArizeFlightSearchEvaluator()
    try:
        results = evaluator.run_evaluation(demo_queries)
        logger.info("🎉 Phoenix evaluation demo complete for flight search agent!")
        logger.info("💡 Visit http://localhost:6006 to see detailed traces and evaluations")
        logger.info("📊 The Phoenix UI shows LangGraph execution, tool calls, and evaluation scores")
        logger.info("🔧 Compare hotel vs flight agent performance using the evaluation metrics")
        return results
    finally:
        evaluator.cleanup()


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
    # Run demo mode for quick testing (matches notebook demo)
    # Uncomment the next line to run demo mode instead of full evaluation
    # run_phoenix_demo()
    
    # Run full evaluation
    main()
