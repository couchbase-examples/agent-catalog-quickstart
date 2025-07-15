#!/usr/bin/env python3
"""
Arize AI Integration for Hotel Support Agent Evaluation

This module provides comprehensive evaluation capabilities using Arize AI observability
platform for the hotel support agent. It demonstrates how to:

1. Set up Arize observability for LangChain-based agents
2. Create and manage evaluation datasets for hotel search scenarios
3. Run automated evaluations with LLM-as-a-judge for response quality
4. Track performance metrics and traces for hotel search systems
5. Monitor tool usage and search effectiveness

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

# Configuration constants
SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
PROJECT_NAME = "hotel-support-agent-evaluation"

# Import the hotel support agent components
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import setup_environment, CouchbaseClient

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
    from langchain_openai import OpenAIEmbeddings

    ARIZE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Arize dependencies not available: {e}")
    print("   Running in local evaluation mode only...")
    ARIZE_AVAILABLE = False


# CouchbaseSetup class removed - now using functions imported from main.py instead of duplicating code


class ArizeHotelSupportEvaluator:
    """
    Comprehensive evaluation system for hotel support agents using Arize AI.

    This class provides:
    - Hotel search performance evaluation with multiple metrics
    - Tool effectiveness monitoring (search, details, booking)
    - Response quality assessment and search accuracy tracking
    - Comparative analysis of different hotel search strategies
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
            self.hallucination_rails = list(HALLUCINATION_PROMPT_RAILS_MAP.values())
            self.toxicity_rails = list(TOXICITY_PROMPT_RAILS_MAP.values())
        else:
            print("⚠️ Arize not available - running basic evaluation only")

    def setup_couchbase_infrastructure(self):
        """Setup Couchbase infrastructure for evaluation using CouchbaseClient from main.py."""
        logger = logging.getLogger(__name__)

        try:
            # Setup environment - this handles all environment variables and defaults
            logger.info("🔧 Setting up environment...")
            setup_environment()
            
            # Initialize Couchbase client with travel-sample bucket
            logger.info("🔧 Creating Couchbase client...")
            client = CouchbaseClient(
                conn_string=os.environ['CB_CONN_STRING'],
                username=os.environ['CB_USERNAME'],
                password=os.environ['CB_PASSWORD'],
                bucket_name=os.environ.get('CB_BUCKET', 'travel-sample')
            )
            
            logger.info("🔧 Connecting to Couchbase...")
            client.connect()
            
            logger.info("🔧 Setting up collection...")
            client.setup_collection(
                os.environ.get('CB_SCOPE', 'agentc_data'), 
                os.environ.get('CB_COLLECTION', 'hotel_data')
            )
            
            # Load index definition for vector search setup
            index_file = "agentcatalog_index.json"
            if not os.path.exists(index_file):
                parent_dir = os.path.dirname(os.path.dirname(__file__))
                index_file = os.path.join(parent_dir, "agentcatalog_index.json")
            
            if os.path.exists(index_file):
                logger.info("🔧 Setting up vector search index...")
                with open(index_file, "r") as f:
                    index_definition = json.load(f)
                client.setup_vector_search_index(index_definition, os.environ.get('CB_SCOPE', 'agentc_data'))
            else:
                logger.warning(f"Index definition file {index_file} not found")

            # Setup vector store using client
            logger.info("🔧 Setting up vector store...")
            
            # Create embeddings using the same pattern as main.py
            import base64
            api_key = base64.b64encode(
                f"{os.getenv('CB_USERNAME')}:{os.getenv('CB_PASSWORD')}".encode()
            ).decode()

            embeddings = OpenAIEmbeddings(
                model=os.getenv("CAPELLA_API_EMBEDDING_MODEL"),
                api_key=api_key,
                base_url=os.getenv("CAPELLA_API_ENDPOINT"),
            )
            
            vector_store = client.setup_vector_store(
                scope_name=os.environ.get('CB_SCOPE', 'agentc_data'),
                collection_name=os.environ.get('CB_COLLECTION', 'hotel_data'),
                index_name=os.environ.get('CB_INDEX', 'hotel_data_index'),
                embeddings=embeddings
            )

            logger.info("✅ Couchbase infrastructure setup complete for evaluation")
            return client.cluster, vector_store
        except Exception as e:
            logger.error(f"❌ Error setting up Couchbase infrastructure: {e}")
            raise

    def create_agent(self, catalog: agentc.Catalog, span: agentc.Span):
        """Create a hotel support agent for evaluation."""
        logger = logging.getLogger(__name__)

        try:
            # Import required components
            from langchain.agents import AgentExecutor, create_react_agent
            from langchain_openai import ChatOpenAI
            from langchain_core.tools import Tool
            from langchain_core.prompts import PromptTemplate
            import agentc_langchain

            # Setup LLM with Agent Catalog callback
            llm = ChatOpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                model="gpt-4o",
                temperature=0,
                callbacks=[agentc_langchain.chat.Callback(span=span)],
            )

            # Load only the search tool from Agent Catalog
            tool_search = catalog.find("tool", name="search_vector_database")

            if not tool_search:
                raise ValueError(
                    "Could not find search_vector_database tool. Make sure it's indexed with 'agentc index tools/'"
                )

            # Create single tool list
            tools = [
                Tool(
                    name=tool_search.meta.name,
                    description=tool_search.meta.description,
                    func=tool_search.func,
                ),
            ]

            # Get prompt from Agent Catalog
            hotel_prompt = catalog.find("prompt", name="hotel_search_assistant")
            if not hotel_prompt:
                raise ValueError(
                    "Could not find hotel_search_assistant prompt in catalog. Make sure it's indexed with 'agentc index prompts/'"
                )

            # Create a custom prompt using the catalog prompt content
            prompt_content = hotel_prompt.content.strip()

            custom_prompt = PromptTemplate(
                template=prompt_content,
                input_variables=["input", "agent_scratchpad"],
                partial_variables={
                    "tools": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                    "tool_names": ", ".join([tool.name for tool in tools]),
                },
            )

            # Create ReAct agent
            agent = create_react_agent(llm, tools, custom_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                handle_parsing_errors=True,
                max_iterations=3,  # Reduced from 10 to 3 to prevent infinite loops
                max_execution_time=30,  # Add execution timeout
                return_intermediate_steps=True
            )

            logger.info("Hotel support agent created for evaluation")
            return agent_executor
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            raise

    def _setup_arize_observability(self):
        """Configure Arize observability with OpenTelemetry instrumentation."""
        try:
            print("🔧 Setting up Arize observability...")

            # Start Phoenix session with environment variable approach (no deprecated port parameter)
            import os
            import phoenix as px
            
            # Kill any existing Phoenix processes on common ports
            try:
                import subprocess
                for port in [4317, 6006, 6007, 6008]:
                    try:
                        subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, check=True)
                        subprocess.run(['kill', '-9'] + subprocess.run(['lsof', '-ti', f':{port}'], capture_output=True, text=True).stdout.strip().split(), capture_output=True)
                        print(f"🔄 Killed existing process on port {port}")
                    except:
                        pass  # Port not in use, continue
            except:
                pass  # lsof not available or other error
            
            # Try to start Phoenix with environment variables (preferred method)
            ports_to_try = [6006, 6007, 6008, 6009]
            self.phoenix_session = None
            
            for port in ports_to_try:
                try:
                    # Set environment variables for Phoenix
                    os.environ['PHOENIX_PORT'] = str(port)
                    os.environ['PHOENIX_GRPC_PORT'] = str(4317 + (port - 6006))  # Use different GRPC ports
                    
                    self.phoenix_session = px.launch_app()
                    print(f"🌐 Phoenix UI: http://localhost:{port}")
                    break
                except Exception as e:
                    print(f"⚠️ Phoenix failed on port {port}: {e}")
                    continue
            
            if not self.phoenix_session:
                print("❌ Could not start Phoenix on any port, continuing without Phoenix UI")

            # Register Phoenix OTEL and get tracer provider  
            if self.phoenix_session:
                try:
                    self.tracer_provider = register(
                        project_name=PROJECT_NAME,
                        endpoint=f"http://localhost:{self.phoenix_session.port}/v1/traces",
                    )
                    print("✅ Phoenix OTEL registered successfully")
                except Exception as e:
                    print(f"⚠️ Phoenix OTEL registration failed: {e}")
                    # Fallback to local tracer
                    self.tracer_provider = TracerProvider()
                    trace.set_tracer_provider(self.tracer_provider)
            else:
                # Fallback to local tracer
                self.tracer_provider = TracerProvider()
                trace.set_tracer_provider(self.tracer_provider)
                print("✅ Local tracing configured successfully")

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
                        print(f"✅ {name} instrumented successfully")
                    else:
                        print(f"ℹ️ {name} already instrumented, skipping")
                except Exception as e:
                    print(f"⚠️ {name} instrumentation failed: {e}")
                    continue

            # Initialize Arize datasets client
            if API_KEY != "your-api-key":
                try:
                    self.arize_client = ArizeDatasetsClient(
                        developer_key=API_KEY, api_key=API_KEY
                    )
                    print("✅ Arize datasets client initialized")
                except Exception as e:
                    print(f"⚠️ Could not initialize Arize datasets client: {e}")
                    self.arize_client = None

            print("✅ Arize observability configured successfully")

        except Exception as e:
            print(f"⚠️ Error setting up Arize observability: {e}")

    def cleanup(self):
        """Clean up resources and close Phoenix session."""
        try:
            # Clean up environment variables
            import os
            for var in ['PHOENIX_PORT', 'PHOENIX_GRPC_PORT']:
                if var in os.environ:
                    del os.environ[var]
                    
            print("🔒 Phoenix cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during Phoenix cleanup: {e}")

    def run_hotel_search_evaluation(self, test_inputs: List[str]) -> pd.DataFrame:
        """
        Run hotel search evaluation on a set of test inputs.

        Args:
            test_inputs: List of hotel search queries to evaluate

        Returns:
            DataFrame with evaluation results
        """
        print(f"🚀 Running evaluation on {len(test_inputs)} queries...")

        # Setup infrastructure
        print("🔧 Setting up Couchbase infrastructure...")
        cluster, vector_store = self.setup_couchbase_infrastructure()
        print("✅ Couchbase infrastructure setup complete")

        # Create agent
        print("🤖 Creating hotel support agent...")
        agent = self.create_agent(self.catalog, self.span)
        print("✅ Hotel support agent created")

        results = []

        for i, query in enumerate(test_inputs, 1):
            print(f"  📝 Query {i}/{len(test_inputs)}: {query}")

            try:
                print(f"     🔄 Agent processing query...")
                response_dict = agent.invoke({"input": query})
                response = (
                    response_dict.get("output", str(response_dict))
                    if response_dict
                    else "No response"
                )
                print(f"     ✅ Response received")

                # Analyze response
                has_hotel_info = self._check_hotel_info(response)
                has_recommendations = self._check_recommendations(response)
                has_amenities = self._check_amenities(response)
                appropriate_response = self._check_appropriate_response(query, response)

                # Create response preview
                response_preview = (
                    (response[:150] + "...") if len(response) > 150 else response
                )
                print(f"     💬 Response preview: {response_preview}")

                results.append(
                    {
                        "example_id": f"hotel_query_{i}",
                        "query": query,
                        "output": response,
                        "has_hotel_info": has_hotel_info,
                        "has_recommendations": has_recommendations,
                        "has_amenities": has_amenities,
                        "appropriate_response": appropriate_response,
                        "response_length": len(response),
                    }
                )

            except Exception as e:
                print(f"     ❌ Error processing query: {e}")
                results.append(
                    {
                        "example_id": f"hotel_query_{i}",
                        "query": query,
                        "output": f"Error: {str(e)}",
                        "has_hotel_info": False,
                        "has_recommendations": False,
                        "has_amenities": False,
                        "appropriate_response": False,
                        "response_length": 0,
                    }
                )

        return pd.DataFrame(results)

    def _check_hotel_info(self, response_text: str) -> bool:
        """Check if response contains hotel information."""
        hotel_indicators = [
            "hotel",
            "room",
            "accommodation",
            "stay",
            "booking",
            "reservation",
            "check-in",
            "check-out",
        ]
        return any(indicator in response_text.lower() for indicator in hotel_indicators)

    def _check_recommendations(self, response_text: str) -> bool:
        """Check if response contains hotel recommendations."""
        recommendation_indicators = [
            "recommend",
            "suggest",
            "option",
            "choice",
            "available",
            "found",
            "located",
        ]
        return any(
            indicator in response_text.lower()
            for indicator in recommendation_indicators
        )

    def _check_amenities(self, response_text: str) -> bool:
        """Check if response mentions amenities."""
        amenity_indicators = [
            "amenities",
            "pool",
            "gym",
            "wifi",
            "breakfast",
            "parking",
            "spa",
            "restaurant",
            "bar",
        ]
        return any(
            indicator in response_text.lower() for indicator in amenity_indicators
        )

    def _check_appropriate_response(self, query: str, response: str) -> bool:
        """Check if response is appropriate for the query."""
        # Check for hallucination indicators
        hallucination_indicators = [
            "mars",
            "pluto",
            "atlantis",
            "fictional",
            "not possible",
            "not available",
        ]
        has_hallucination = any(
            indicator in response.lower() for indicator in hallucination_indicators
        )

        # Check for reasonable response length
        reasonable_length = 50 < len(response) < 2000

        # Check for hotel-related content
        has_hotel_content = self._check_hotel_info(response)

        return not has_hallucination and reasonable_length and has_hotel_content

    def run_arize_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run comprehensive Arize-based LLM evaluations on the results using Phoenix evaluators.

        Args:
            results_df: DataFrame with evaluation results

        Returns:
            DataFrame with additional Arize evaluation columns
        """
        if not ARIZE_AVAILABLE:
            print("⚠️ Arize not available - skipping LLM evaluations")
            return results_df

        print(f"🧠 Running Comprehensive Phoenix Evaluations on {len(results_df)} responses...")
        print("   📋 Evaluation criteria:")
        print("      🔍 Relevance: Does the response address the hotel search query?")
        print("      🎯 Correctness: Is the hotel information accurate and helpful?")
        print("      🚨 Hallucination: Does the response contain fabricated information?")
        print("      ☠️  Toxicity: Is the response harmful or inappropriate?")

        # Sample evaluation data preview
        if len(results_df) > 0:
            sample_row = results_df.iloc[0]
            print(f"\n   🔍 Sample evaluation data:")
            print(f"      Query: {sample_row['query']}")
            print(f"      Response: {sample_row['output'][:100]}...")

        try:
            # Prepare data for evaluation with improved reference texts
            evaluation_data = []
            for _, row in results_df.iterrows():
                query = row["query"]
                output = row["output"]
                
                # Create more specific reference text based on query content
                if "luxury" in query.lower() and "amenities" in query.lower():
                    reference = "A relevant response listing luxury hotels with specific amenities like concierge, spa, restaurant, fitness center, and pricing details"
                elif "fitness center" in query.lower() or "gym" in query.lower():
                    reference = "A relevant response about hotels with fitness centers, including 24/7 fitness facilities and other health amenities"
                elif "miami" in query.lower() and "beach" in query.lower():
                    reference = "A relevant response about Miami beach resorts with ocean views, pools, spa amenities, and beachfront locations"
                elif "business" in query.lower():
                    reference = "A relevant response about business hotels with conference facilities, business centers, and executive amenities"
                else:
                    reference = f"A helpful and accurate response about {query.lower()} with specific hotel details, amenities, pricing, and locations"
                
                evaluation_data.append(
                    {
                        # Standard columns for all evaluations
                        "input": query,
                        "output": output,
                        "reference": reference,
                        
                        # Specific columns for different evaluations
                        "query": query,  # For hallucination evaluation
                        "response": output,  # For hallucination evaluation
                        "text": output,  # For toxicity evaluation
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
            print(f"\n   🧠 Running advanced Phoenix evaluations...")

            try:
                # Run all evaluations using run_evals for comprehensive analysis
                # Note: run_evals API changed, using correct parameter name
                print(f"   🔍 Running evaluations with {len(evaluators)} evaluators on {len(eval_df)} samples")
                print(f"   📋 Evaluators: {list(evaluators.keys())}")
                print(f"   📊 Evaluation DataFrame columns: {list(eval_df.columns)}")
                
                evaluation_results = run_evals(
                    dataframe=eval_df,  # Changed back to dataframe parameter
                    evaluators=list(evaluators.values()),
                    provide_explanation=True,
                )
                
                print(f"   ✅ run_evals completed successfully")

                # Add evaluation results to our DataFrame
                if evaluation_results is not None:
                    print(f"   📊 Evaluation results type: {type(evaluation_results)}")
                    
                    # Handle DataFrame results from run_evals
                    if isinstance(evaluation_results, pd.DataFrame) and not evaluation_results.empty:
                        print(f"   📋 DataFrame columns: {list(evaluation_results.columns)}")
                        print(f"   📋 DataFrame shape: {evaluation_results.shape}")
                        
                        # Phoenix evaluators return results with specific column patterns
                        for col in evaluation_results.columns:
                            if col.endswith('_eval'):
                                # Extract evaluator name from column (e.g., 'relevance_eval' -> 'relevance')
                                eval_name = col.replace('_eval', '')
                                results_df[f"arize_{eval_name}"] = evaluation_results[col].tolist()
                                print(f"      ✅ Extracted {eval_name} evaluations")
                            elif col.endswith('_explanation'):
                                # Extract evaluator name from column (e.g., 'relevance_explanation' -> 'relevance')
                                eval_name = col.replace('_explanation', '')
                                results_df[f"arize_{eval_name}_explanation"] = evaluation_results[col].tolist()
                                print(f"      ✅ Extracted {eval_name} explanations")
                            elif col.endswith('_score'):
                                # Extract evaluator name from column (e.g., 'relevance_score' -> 'relevance')
                                eval_name = col.replace('_score', '')
                                results_df[f"arize_{eval_name}_score"] = evaluation_results[col].tolist()
                                print(f"      ✅ Extracted {eval_name} scores")
                        
                        print(f"   ✅ Comprehensive evaluations completed successfully")
                    elif isinstance(evaluation_results, list) and len(evaluation_results) > 0:
                        # It's a list - convert to DataFrame format that we can use
                        print(f"   🔄 Converting list results to DataFrame format...")
                        print(f"   📋 List length: {len(evaluation_results)}")
                        
                        # Debug: Check what's in the list
                        for i, eval_result in enumerate(evaluation_results):
                            print(f"   📋 Result {i}: type={type(eval_result)}")
                            if hasattr(eval_result, '__dict__'):
                                print(f"        attributes: {list(eval_result.__dict__.keys())}")
                            elif isinstance(eval_result, dict):
                                print(f"        keys: {list(eval_result.keys())}")
                        
                        # Try to convert results - check if they're DataFrames
                        combined_results = {}
                        evaluator_names = list(evaluators.keys())
                        
                        for i, eval_result in enumerate(evaluation_results):
                            if isinstance(eval_result, pd.DataFrame):
                                print(f"   📊 Processing DataFrame result with columns: {list(eval_result.columns)}")
                                
                                # Get the evaluator name from the order (Phoenix returns results in same order as evaluators)
                                eval_name = evaluator_names[i] if i < len(evaluator_names) else f"eval_{i}"
                                
                                # Extract the standard columns from Phoenix evaluators
                                if 'label' in eval_result.columns:
                                    combined_results[f"arize_{eval_name}"] = eval_result['label'].tolist()
                                    print(f"      ✅ Extracted {eval_name} labels")
                                
                                if 'score' in eval_result.columns:
                                    combined_results[f"arize_{eval_name}_score"] = eval_result['score'].tolist()
                                    print(f"      ✅ Extracted {eval_name} scores")
                                
                                if 'explanation' in eval_result.columns:
                                    combined_results[f"arize_{eval_name}_explanation"] = eval_result['explanation'].tolist()
                                    print(f"      ✅ Extracted {eval_name} explanations")
                        
                        # Apply the combined results to our DataFrame
                        for key, values in combined_results.items():
                            if len(values) == len(results_df):
                                results_df[key] = values
                                print(f"      ✅ Added {key} to results")
                            else:
                                print(f"      ⚠️ Length mismatch for {key}: {len(values)} vs {len(results_df)}")
                        
                        print(f"   ✅ List results converted successfully")
                    else:
                        print(f"   ⚠️ Phoenix evaluators returned unexpected format: {type(evaluation_results)}")
                        print(f"   🔍 Result details: {evaluation_results}")
                        # Fall back to individual evaluations
                        self._run_individual_evaluations(eval_df, results_df, evaluators)
                else:
                    print(f"   ⚠️ Phoenix evaluators returned None")
                    # Fall back to individual evaluations
                    self._run_individual_evaluations(eval_df, results_df, evaluators)

            except Exception as e:
                print(f"   ⚠️ run_evals failed: {e}, falling back to individual evaluations")
                self._run_individual_evaluations(eval_df, results_df, evaluators)

            # Display sample evaluation results
            if len(results_df) > 0:
                print(f"\n   📝 Sample evaluation results:")
                for i in range(min(2, len(results_df))):
                    row = results_df.iloc[i]
                    print(f"      Query: {row['query']}")
                    
                    for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
                        eval_col = f"arize_{eval_type}"
                        explanation_col = f"arize_{eval_type}_explanation"
                        
                        if eval_col in row:
                            evaluation = row[eval_col]
                            explanation = str(row.get(explanation_col, "No explanation"))[:80] + "..."
                            print(f"      {eval_type.title()}: {evaluation} - {explanation}")
                    print("")

            print(f"   ✅ All Phoenix evaluations completed")

        except Exception as e:
            print(f"   ❌ Error running Phoenix evaluations: {e}")
            # Add default values if evaluation fails
            for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
                results_df[f"arize_{eval_type}"] = "unknown"
                results_df[f"arize_{eval_type}_explanation"] = f"Error: {e}"

        return results_df

    def _run_individual_evaluations(self, eval_df: pd.DataFrame, results_df: pd.DataFrame, evaluators: dict):
        """Run individual evaluations with proper column mapping and error handling."""
        print(f"   🔄 Running individual evaluations...")

        for eval_name, evaluator in evaluators.items():
            try:
                print(f"      📊 Running {eval_name} evaluation...")
                
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
                    print(f"      ⚠️ Unknown evaluator {eval_name}, setting defaults")
                    results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                    results_df[f"arize_{eval_name}_explanation"] = [f"{eval_name} evaluation not implemented"] * len(results_df)
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
                        
                        print(f"      ✅ {eval_name} evaluation completed with {len(eval_results)} results")
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
                        
                        print(f"      ✅ {eval_name} evaluation completed with {len(eval_results)} results (list format)")
                    else:
                        # Single value or unexpected format
                        print(f"      ⚠️ {eval_name} evaluation returned unexpected format: {type(eval_results)}")
                        results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                        results_df[f"arize_{eval_name}_explanation"] = [f"Unexpected format: {type(eval_results)}"] * len(results_df)
                else:
                    # None result
                    print(f"      ⚠️ {eval_name} evaluation returned None")
                    results_df[f"arize_{eval_name}"] = ["not_evaluated"] * len(results_df)
                    results_df[f"arize_{eval_name}_explanation"] = ["Evaluation returned None"] * len(results_df)

            except Exception as e:
                print(f"      ⚠️ {eval_name} evaluation failed: {e}")
                results_df[f"arize_{eval_name}"] = ["error"] * len(results_df)
                results_df[f"arize_{eval_name}_explanation"] = [f"Error: {str(e)}"] * len(results_df)


def eval_hotel_search_basic():
    """Run basic hotel search evaluation with a small set of test queries."""
    print("🔍 Running basic hotel search evaluation...")
    print("📋 This evaluation tests:")
    print("   • Agent's ability to understand hotel search queries")
    print("   • Quality of responses using vector search + LLM")
    print("   • LLM-based relevance and correctness scoring")

    # Import shared queries
    from data.queries import get_evaluation_queries
    
    # Use shared evaluation queries
    test_inputs = get_evaluation_queries()

    # Initialize evaluation components
    catalog = agentc.Catalog()
    span = catalog.Span(name="HotelSupportEvaluation")

    evaluator = ArizeHotelSupportEvaluator(catalog, span)

    # Run the evaluation
    results_df = evaluator.run_hotel_search_evaluation(test_inputs)

    # Run Arize evaluations if available
    if ARIZE_AVAILABLE:
        results_df = evaluator.run_arize_evaluations(results_df)

 

    # Calculate metrics
    total_queries = len(results_df)
    queries_with_hotel_info = results_df["has_hotel_info"].sum()
    queries_with_recommendations = results_df["has_recommendations"].sum()
    queries_with_amenities = results_df["has_amenities"].sum()
    appropriate_responses = results_df["appropriate_response"].sum()
    success_rate = (queries_with_hotel_info / total_queries) * 100

    # Print summary
    print(f"\n✅ Basic hotel search evaluation completed:")
    print(f"   📊 Total queries processed: {total_queries}")
    print(f"   🎯 Queries with hotel info: {queries_with_hotel_info}")
    print(f"   📈 Success rate: {success_rate:.1f}%")
    print(f"   🏨 Queries with recommendations: {queries_with_recommendations}")
    print(f"   🛏️ Queries with amenities: {queries_with_amenities}")
    print(f"   ✅ Appropriate responses: {appropriate_responses}")

    if ARIZE_AVAILABLE:
        arize_results = {}
        
        # Check which Arize evaluation columns exist and process them
        for eval_type in ['relevance', 'qa', 'hallucination', 'toxicity']:
            col_name = f"arize_{eval_type}"
            if col_name in results_df.columns:
                try:
                    scores = results_df[col_name].value_counts()
                    arize_results[eval_type] = {k: int(v) for k, v in scores.items()}
                except Exception as e:
                    print(f"   ⚠️ Error processing {eval_type} results: {e}")
        
        if arize_results:
            print(f"\n🔍 Arize Evaluation Results:")
            for eval_type, scores in arize_results.items():
                print(f"   📋 {eval_type.title()}: {scores}")
        else:
            print(f"\n⚠️ No Arize evaluation results available (evaluations may have failed)")
            
        # Debug: Print all available columns to understand what we have
        print(f"\n🔍 Available result columns:")
        for col in results_df.columns:
            if col.startswith('arize_'):
                print(f"   📋 {col}")
        
    # Show detailed results for debugging
        print(f"\n📊 Sample detailed results:")
        for i in range(len(results_df)):
            row = results_df.iloc[i]
            print(f"   Query {i+1}: {row['query']}")
            for col in results_df.columns:
                if col.startswith('arize_') and not col.endswith('_explanation'):
                    value = row.get(col, 'N/A')
                    print(f"      {col}: {value}")
            print("")

    print(f"\n💡 Note: Some errors are expected without full Couchbase setup")
    
    # Cleanup evaluator to prevent event loop errors
    if 'evaluator' in locals():
        try:
            evaluator.cleanup()
        except:
            pass

    return results_df


def run_phoenix_demo():
    """Run a simple Phoenix evaluation demo for hotel search agent."""
    print("🔧 Setting up Phoenix evaluation demo...")
    
    # Initialize evaluation components (simplified for demo)
    catalog = agentc.Catalog()
    span = catalog.Span(name="HotelSearchDemo")
    
    # Simple test queries for demo
    demo_queries = [
        "Find me a hotel in New York with a pool",
        "Show me budget hotels in San Francisco"
    ]
    
    print("🚀 Running demo hotel search queries...")
    
    # Mock results for demo (would normally run actual agent)
    demo_results = []
    for query in demo_queries:
        demo_results.append({
            "query": query,
            "response": f"Here are some hotel recommendations for '{query}' with details about amenities and pricing.",
            "has_hotel_info": True,
            "has_recommendations": True
        })
    
    print("📊 Demo results collected successfully!")
    print(f"✅ Processed {len(demo_results)} queries")
    
    # Phoenix evaluations would run here in full implementation
    print("🧠 Phoenix evaluations would run on collected data")
    print("💡 Visit http://localhost:6006 to see detailed traces and evaluations")
    print("📊 The Phoenix UI shows conversation flows, tool calls, and evaluation scores")

def main():
    """Main evaluation function with Arize AI integration."""
    import sys

    # Check if specific evaluation is requested
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_phoenix_demo()
        elif sys.argv[1] == "basic":
            eval_hotel_search_basic()
        else:
            print("Usage: python eval_arize.py [demo|basic]")
    else:
        # Run basic evaluation by default
        print("🔍 Running basic hotel search agent evaluation...")
        eval_hotel_search_basic()

if __name__ == "__main__":
    # Run demo mode for quick testing (matches notebook demo)
    # Uncomment the next line to run demo mode instead of full evaluation
    # run_phoenix_demo()
    
    # Run main evaluation
    main()
