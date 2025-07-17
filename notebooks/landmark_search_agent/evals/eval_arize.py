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

# Import the refactored setup functions
from main import setup_environment, CouchbaseClient

# Import queries and reference answers
from data.queries import (
    LANDMARK_SEARCH_QUERIES,
    QUERY_REFERENCE_ANSWERS,
    get_queries_for_evaluation,
    get_reference_answer
)

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
        TOXICITY_PROMPT_TEMPLATE,
        HALLUCINATION_PROMPT_TEMPLATE,
    )
    from phoenix.trace.dsl import SpanQuery
    from phoenix.trace.exporter import HttpExporter
    from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
    PHOENIX_AVAILABLE = True
except ImportError as e:
    print(f"Phoenix not available: {e}")
    PHOENIX_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_phoenix_session():
    """Setup Phoenix session for tracing and evaluation."""
    if not PHOENIX_AVAILABLE:
        logger.error("Phoenix is not available. Please install phoenix-ai package.")
        return None
        
    try:
        # Launch Phoenix UI
        session = px.launch_app()
        logger.info(f"Phoenix UI launched at: {session.url}")
        
        # Setup tracing
        register(
            project_name="landmark-search-agent",
            endpoint="http://localhost:6006/v1/traces",
        )
        
        # Instrument LlamaIndex
        LlamaIndexInstrumentor().instrument()
        
        return session
    except Exception as e:
        logger.error(f"Failed to setup Phoenix session: {e}")
        return None


def run_landmark_queries(agent, queries: List[str]) -> List[Dict[str, Any]]:
    """Run landmark search queries and collect results."""
    results = []
    
    for i, query in enumerate(queries):
        logger.info(f"Running query {i+1}/{len(queries)}: {query}")
        
        try:
            # Execute the query
            start_time = time.time()
            response = agent.chat(query, chat_history=[])
            end_time = time.time()
            
            result = {
                "query": query,
                "response": response.response,
                "response_time": end_time - start_time,
                "sources": [node.text for node in response.source_nodes] if hasattr(response, 'source_nodes') else [],
                "metadata": [node.metadata for node in response.source_nodes] if hasattr(response, 'source_nodes') else []
            }
            
            results.append(result)
            logger.info(f"Query completed in {result['response_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Error running query '{query}': {e}")
            results.append({
                "query": query,
                "response": f"Error: {str(e)}",
                "response_time": 0,
                "sources": [],
                "metadata": []
            })
    
    return results


def evaluate_with_phoenix(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate results using Phoenix evaluators."""
    if not PHOENIX_AVAILABLE:
        logger.error("Phoenix not available for evaluation")
        return {}
    
    try:
        # Setup OpenAI model for evaluation
        eval_model = OpenAIModel(
            model="gpt-4",
            temperature=0.0,
        )
        
        # Create evaluation dataset
        eval_data = []
        for result in results:
            query = result["query"]
            response = result["response"]
            reference = get_reference_answer(query)
            
            # Validate reference answer exists
            if reference == "No reference answer available for this query.":
                logger.warning(f"No reference answer found for query: {query}")
                # Skip queries without reference answers for evaluation
                continue
                
            eval_data.append({
                "query": query,
                "response": response,
                "reference": reference,
                "context": "; ".join(result["sources"][:3]) if result["sources"] else "No context"
            })
        
        if not eval_data:
            logger.error("No evaluation data available - no queries have reference answers")
            return {}
            
        eval_df = pd.DataFrame(eval_data)
        
        # Run evaluations
        evaluations = {}
        
        # 1. Relevance Evaluation
        logger.info("Running relevance evaluation...")
        relevance_results = llm_classify(
            dataframe=eval_df,
            model=eval_model,
            template=RAG_RELEVANCY_PROMPT_TEMPLATE,
            rails=["relevant", "irrelevant"],
            provide_explanation=True,
        )
        evaluations["relevance"] = relevance_results
        
        # 2. QA Correctness Evaluation
        logger.info("Running QA correctness evaluation...")
        qa_results = llm_classify(
            dataframe=eval_df,
            model=eval_model,
            template=QA_PROMPT_TEMPLATE,
            rails=["correct", "incorrect"],
            provide_explanation=True,
        )
        evaluations["qa_correctness"] = qa_results
        
        # 3. Hallucination Evaluation
        logger.info("Running hallucination evaluation...")
        hallucination_results = llm_classify(
            dataframe=eval_df,
            model=eval_model,
            template=HALLUCINATION_PROMPT_TEMPLATE,
            rails=["factual", "hallucinated"],
            provide_explanation=True,
        )
        evaluations["hallucination"] = hallucination_results
        
        # 4. Toxicity Evaluation
        logger.info("Running toxicity evaluation...")
        toxicity_results = llm_classify(
            dataframe=eval_df,
            model=eval_model,
            template=TOXICITY_PROMPT_TEMPLATE,
            rails=["toxic", "non-toxic"],
            provide_explanation=True,
        )
        evaluations["toxicity"] = toxicity_results
        
        return evaluations
        
    except Exception as e:
        logger.error(f"Error during Phoenix evaluation: {e}")
        return {}


def display_evaluation_results(evaluations: Dict[str, Any], results: List[Dict[str, Any]]):
    """Display evaluation results in a user-friendly format."""
    print("\n" + "="*60)
    print("🏛️  LANDMARK SEARCH AGENT EVALUATION RESULTS")
    print("="*60)
    
    if not evaluations:
        print("❌ No evaluation results available")
        return
    
    # Only count results that have reference answers
    evaluated_results = [r for r in results if get_reference_answer(r["query"]) != "No reference answer available for this query."]
    total_evaluated = len(evaluated_results)
    
    if total_evaluated == 0:
        print("❌ No queries with reference answers found")
        return
    
    # Calculate metrics
    metrics = {}
    for eval_type, eval_results in evaluations.items():
        if eval_type == "relevance":
            relevant_count = sum(1 for r in eval_results if r.get("label") == "relevant")
            metrics["Relevance"] = (relevant_count / total_evaluated) * 100
        elif eval_type == "qa_correctness":
            correct_count = sum(1 for r in eval_results if r.get("label") == "correct")
            metrics["QA Correctness"] = (correct_count / total_evaluated) * 100
        elif eval_type == "hallucination":
            factual_count = sum(1 for r in eval_results if r.get("label") == "factual")
            metrics["Factual Accuracy"] = (factual_count / total_evaluated) * 100
        elif eval_type == "toxicity":
            non_toxic_count = sum(1 for r in eval_results if r.get("label") == "non-toxic")
            metrics["Non-toxic"] = (non_toxic_count / total_evaluated) * 100
    
    # Display metrics
    print(f"\n📊 EVALUATION METRICS ({total_evaluated} queries evaluated)")
    print("-" * 40)
    
    for metric, value in metrics.items():
        status = "✅" if value >= 80 else "⚠️" if value >= 60 else "❌"
        print(f"{status} {metric}: {value:.1f}%")
    
    # Display recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)
    
    if metrics.get("Relevance", 0) < 80:
        print("• Improve search relevance by refining embedding model or search parameters")
    
    if metrics.get("QA Correctness", 0) < 70:
        print("• Enhance response accuracy with better context retrieval")
    
    if metrics.get("Factual Accuracy", 0) < 80:
        print("• Reduce hallucinations by improving grounding to retrieved content")
    
    if metrics.get("Non-toxic", 0) < 95:
        print("• Review and filter any potentially harmful content")
    
    if all(v >= 80 for v in metrics.values()):
        print("• Great job! All metrics are performing well ✅")
    
    # Display sample results
    print(f"\n📝 SAMPLE EVALUATIONS")
    print("-" * 40)
    
    sample_results = evaluated_results[:3]
    for i, result in enumerate(sample_results):
        print(f"\nQuery {i+1}: {result['query']}")
        print(f"Response: {result['response'][:100]}...")
        
        # Show evaluation scores for this query
        for eval_type, eval_results in evaluations.items():
            if i < len(eval_results):
                label = eval_results[i].get("label", "unknown")
                explanation = eval_results[i].get("explanation", "")
                print(f"  {eval_type.title()}: {label} - {explanation[:50]}...")


def upload_to_arize(evaluations: Dict[str, Any], results: List[Dict[str, Any]]):
    """Upload evaluation results to Arize for monitoring."""
    try:
        from arize.pandas.logger import Client
        from arize.utils.types import Schema, Environments
        
        # Initialize Arize client
        api_key = os.getenv("ARIZE_API_KEY")
        if not api_key:
            logger.info("ARIZE_API_KEY not found, skipping Arize upload")
            return
        
        space_id = os.getenv("ARIZE_SPACE_ID", "default")
        client = Client(api_key=api_key)
        
        # Create dataset
        dataset_name = f"landmark-search-evaluation-{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare data for upload - only include results with reference answers
        upload_data = []
        for i, result in enumerate(results):
            reference = get_reference_answer(result["query"])
            if reference == "No reference answer available for this query.":
                continue
                
            row = {
                "query": result["query"],
                "response": result["response"],
                "response_time": result["response_time"],
                "reference": reference
            }
            
            # Add evaluation scores
            for eval_type, eval_results in evaluations.items():
                if i < len(eval_results):
                    row[f"{eval_type}_score"] = eval_results[i].get("label", "unknown")
                    row[f"{eval_type}_explanation"] = eval_results[i].get("explanation", "")
            
            upload_data.append(row)
        
        if not upload_data:
            logger.info("No data to upload to Arize")
            return
            
        df = pd.DataFrame(upload_data)
        
        # Upload to Arize
        response = client.log_validation_records(
            dataframe=df,
            schema=Schema(
                feature_column_names=["query", "response_time"],
                actual_label_column_name="reference",
                prediction_label_column_name="response",
            ),
            model_id="landmark-search-agent",
            model_version="1.0",
            environment=Environments.VALIDATION,
            dataset_name=dataset_name,
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully uploaded evaluation data to Arize: {dataset_name}")
        else:
            logger.error(f"Failed to upload to Arize: {response.status_code}")
            
    except ImportError:
        logger.info("Arize package not available, skipping upload")
    except Exception as e:
        logger.error(f"Error uploading to Arize: {e}")


def main():
    """Main evaluation function."""
    try:
        logger.info("Starting Landmark Search Agent Evaluation...")
        
        # Setup Phoenix session
        session = setup_phoenix_session()
        
        # Setup environment and agent
        setup_environment()
        
        # Import and setup agent
        from main import setup_landmark_agent
        agent, client = setup_landmark_agent()
        
        logger.info("Agent setup completed, starting evaluation...")
        
        # Get queries for evaluation
        queries = get_queries_for_evaluation(limit=10)
        logger.info(f"Using {len(queries)} queries for evaluation")
        
        # Run queries
        results = run_landmark_queries(agent, queries)
        
        # Evaluate with Phoenix
        evaluations = evaluate_with_phoenix(results)
        
        # Display results
        display_evaluation_results(evaluations, results)
        
        # Upload to Arize
        upload_to_arize(evaluations, results)
        
        logger.info("Evaluation completed successfully!")
        
        if session:
            print(f"\n🔍 View detailed traces at: {session.url}")
        
    except Exception as e:
        logger.error(f"Error in main evaluation: {e}")
        raise


if __name__ == "__main__":
    main() 