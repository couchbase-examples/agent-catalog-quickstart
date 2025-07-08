# #!/usr/bin/env python3
# """
# Arize AI Integration for Hotel Support Agent Evaluation

# This module provides comprehensive evaluation capabilities using Arize AI observability
# platform for the hotel support agent. It demonstrates how to:

# 1. Set up Arize observability for LangChain-based agents
# 2. Create and manage evaluation datasets for hotel search scenarios
# 3. Run automated evaluations with LLM-as-a-judge
# 4. Track performance metrics and traces for hotel recommendation systems
# 5. Monitor vector search effectiveness and tool usage

# The implementation integrates with the existing Agent Catalog infrastructure
# while extending it with Arize AI capabilities for production monitoring.
# """

# import json
# import os
# import pathlib
# import unittest.mock
# from typing import Dict, List
# from uuid import uuid4

# import agentc
# import pandas as pd

# # Configuration constants
# SPACE_ID = os.getenv("ARIZE_SPACE_ID", "your-space-id")
# API_KEY = os.getenv("ARIZE_API_KEY", "your-api-key")
# DEVELOPER_KEY = os.getenv("ARIZE_DEVELOPER_KEY", "your-developer-key")
# PROJECT_NAME = "hotel-support-agent-evaluation"

# # Import the hotel support agent components
# from main import setup_environment, setup_couchbase_connection, load_hotel_data, run_agent_chat

# # Try to import Arize dependencies with fallback
# try:
#     from arize.experimental.datasets import ArizeDatasetsClient
#     from arize.experimental.datasets.experiments.types import (
#         ExperimentTaskResultColumnNames,
#         EvaluationResultColumnNames,
#     )
#     from arize.experimental.datasets.utils.constants import GENERATIVE
#     from arize.otel import register
#     from openinference.instrumentation.langchain import LangChainInstrumentor
#     from openinference.instrumentation.openai import OpenAIInstrumentor
#     from phoenix.evals import (
#         QA_PROMPT_RAILS_MAP,
#         QA_PROMPT_TEMPLATE,
#         RAG_RELEVANCY_PROMPT_RAILS_MAP,
#         RAG_RELEVANCY_PROMPT_TEMPLATE,
#         OpenAIModel,
#         llm_classify,
#     )
#     ARIZE_AVAILABLE = True
# except ImportError as e:
#     print(f"⚠️ Arize dependencies not available: {e}")
#     print("   Running in local evaluation mode only...")
#     ARIZE_AVAILABLE = False


# class ArizeHotelSupportEvaluator:
#     """
#     Comprehensive evaluation system for hotel support agents using Arize AI.
    
#     This class provides:
#     - Hotel search performance evaluation with multiple metrics
#     - Vector search effectiveness monitoring
#     - Tool usage analysis and optimization
#     - Comparative analysis of search strategies
#     """

#     def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
#         """Initialize the Arize evaluator with Agent Catalog integration."""
#         self.catalog = catalog
#         self.span = span
#         self.arize_client = None
#         self.dataset_id = None
#         self.tracer_provider = None
        
#         # Initialize Arize observability if available
#         if ARIZE_AVAILABLE:
#             self._setup_arize_observability()
            
#             # Initialize evaluation models
#             self.evaluator_llm = OpenAIModel(model="gpt-4o")
            
#             # Define evaluation rails
#             self.relevance_rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())
#             self.qa_rails = list(QA_PROMPT_RAILS_MAP.values())
#         else:
#             print("⚠️ Arize not available - running basic evaluation only")

#     def _setup_arize_observability(self):
#         """Configure Arize observability with OpenTelemetry instrumentation."""
#         try:
#             # Setup tracer provider
#             self.tracer_provider = register(
#                 space_id=SPACE_ID,
#                 api_key=API_KEY,
#                 project_name=PROJECT_NAME,
#             )
            
#             # Instrument LangChain and OpenAI
#             LangChainInstrumentor().instrument(tracer_provider=self.tracer_provider)
#             OpenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
            
#             # Initialize Arize datasets client
#             if DEVELOPER_KEY != "your-developer-key":
#                 self.arize_client = ArizeDatasetsClient(
#                     developer_key=DEVELOPER_KEY,
#                     api_key=API_KEY
#                 )
            
#             print("✅ Arize observability configured successfully")
            
#         except Exception as e:
#             print(f"⚠️ Warning: Could not configure Arize observability: {e}")
#             print("   Continuing with local evaluation only...")

#     def run_hotel_search_evaluation(self, test_inputs: List[str]) -> pd.DataFrame:
#         """
#         Run the hotel support agent on test inputs and collect results.
        
#         Args:
#             test_inputs: List of hotel search queries to evaluate
            
#         Returns:
#             DataFrame with input queries and agent responses
#         """
#         results = []
        
#         with self.span.new("HotelSupportEvaluation") as eval_span:
#             # Setup the agent environment once
#             setup_environment()
#             cluster = setup_couchbase_connection()
#             load_hotel_data(cluster)
            
#             for i, query in enumerate(test_inputs):
#                 with eval_span.new(f"Query_{i}", query=query) as query_span:
#                     try:
#                         # Mock input to prevent interactive prompts
#                         with unittest.mock.patch("builtins.input", side_effect=[query, "quit"]), \
#                              unittest.mock.patch("builtins.print", lambda *args, **kwargs: None):
                            
#                             # Capture the agent response
#                             response_captured = []
                            
#                             # Mock the print function to capture agent responses
#                             def capture_print(*args, **kwargs):
#                                 response_captured.append(" ".join(str(arg) for arg in args))
                            
#                             with unittest.mock.patch("builtins.print", capture_print):
#                                 try:
#                                     run_agent_chat(cluster, self.catalog, eval_span)
#                                 except (EOFError, KeyboardInterrupt):
#                                     pass  # Expected when we quit the chat
                            
#                             # Extract the meaningful response (filter out system messages)
#                             response = ""
#                             for captured in response_captured:
#                                 if ("hotel" in captured.lower() or 
#                                     "search" in captured.lower() or 
#                                     "accommodation" in captured.lower() or
#                                     len(captured) > 50):  # Likely agent response
#                                     response = captured
#                                     break
                            
#                             if not response:
#                                 response = " ".join(response_captured) if response_captured else "No response"
                            
#                             # Analyze the response for hotel-related information
#                             has_hotel_info = any(keyword in response.lower() for keyword in 
#                                                ["hotel", "accommodation", "room", "amenities", "price", "booking"])
                            
#                             # Check if vector search was likely used
#                             uses_search = "search" in response.lower() or len(response) > 100
                            
#                             results.append({
#                                 "input": query,
#                                 "output": response,
#                                 "has_hotel_info": has_hotel_info,
#                                 "uses_search": uses_search,
#                                 "response_length": len(response),
#                                 "example_id": i
#                             })
                            
#                             query_span["response"] = response
#                             query_span["has_hotel_info"] = has_hotel_info
#                             query_span["uses_search"] = uses_search
                            
#                     except Exception as e:
#                         print(f"❌ Error evaluating query '{query}': {e}")
#                         results.append({
#                             "input": query,
#                             "output": f"Error: {str(e)}",
#                             "has_hotel_info": False,
#                             "uses_search": False,
#                             "response_length": 0,
#                             "example_id": i
#                         })
#                         query_span["error"] = str(e)
        
#         return pd.DataFrame(results)

#     def run_arize_evaluations(self, results_df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Run Arize LLM-based evaluations on agent responses.
        
#         Args:
#             results_df: DataFrame with agent inputs and outputs
            
#         Returns:
#             DataFrame with Arize evaluation scores added
#         """
#         if not ARIZE_AVAILABLE:
#             print("⚠️ Arize evaluations not available - dependencies missing")
#             return results_df
            
#         try:
#             # Relevance evaluation
#             print("🔍 Running Arize relevance evaluation...")
#             relevance_eval_df = llm_classify(
#                 dataframe=results_df,
#                 template=RAG_RELEVANCY_PROMPT_TEMPLATE,
#                 model=self.evaluator_llm,
#                 rails=self.relevance_rails,
#                 provide_explanation=True,
#                 include_prompt=True,
#                 concurrency=2,
#             )
            
#             # Correctness evaluation
#             print("🎯 Running Arize correctness evaluation...")
#             correctness_eval_df = llm_classify(
#                 dataframe=results_df,
#                 template=QA_PROMPT_TEMPLATE,
#                 model=self.evaluator_llm,
#                 rails=self.qa_rails,
#                 provide_explanation=True,
#                 include_prompt=True,
#                 concurrency=2,
#             )
            
#             # Merge evaluations
#             merged_df = results_df.copy()
#             merged_df["arize_relevance"] = relevance_eval_df.get("label", "unknown")
#             merged_df["arize_relevance_explanation"] = relevance_eval_df.get("explanation", "")
#             merged_df["arize_correctness"] = correctness_eval_df.get("label", "unknown")
#             merged_df["arize_correctness_explanation"] = correctness_eval_df.get("explanation", "")
            
#             return merged_df
            
#         except Exception as e:
#             print(f"❌ Error running Arize evaluations: {e}")
#             return results_df

#     def evaluate_hotel_search_scenarios(self) -> Dict[str, pd.DataFrame]:
#         """
#         Run comprehensive evaluation on different hotel search scenarios.
        
#         Returns:
#             Dictionary of scenario names to evaluation results
#         """
#         scenarios = {
#             "location_searches": [
#                 "Find hotels in New York City",
#                 "I need accommodation in San Francisco",
#                 "Show me hotels near LAX airport",
#                 "Hotels in downtown Chicago",
#             ],
#             "amenity_filtering": [
#                 "Hotels with swimming pool and gym",
#                 "I need a hotel with free WiFi and parking",
#                 "Find luxury hotels with spa services",
#                 "Budget hotels with continental breakfast",
#             ],
#             "budget_queries": [
#                 "Cheap hotels under $100 per night",
#                 "Luxury hotels with premium amenities",
#                 "Mid-range hotels for business travel",
#                 "Family-friendly hotels with good value",
#             ],
#             "specific_requirements": [
#                 "Pet-friendly hotels in Seattle",
#                 "Hotels with accessibility features",
#                 "Business hotels with conference rooms",
#                 "Boutique hotels with unique character",
#             ],
#             "travel_purpose": [
#                 "Hotels for romantic getaway",
#                 "Business travel accommodation",
#                 "Family vacation hotels with kids activities",
#                 "Solo traveler budget accommodation",
#             ],
#             "irrelevant_queries": [
#                 "What's the weather like today?",
#                 "How do I cook pasta?",
#                 "Tell me about stock market trends",
#                 "What's the capital of France?",
#             ],
#         }
        
#         results = {}
        
#         for scenario_name, queries in scenarios.items():
#             print(f"\n🚀 Evaluating scenario: {scenario_name}")
#             print(f"   Running {len(queries)} queries...")
            
#             # Run hotel search evaluation
#             agent_results = self.run_hotel_search_evaluation(queries)
            
#             # Run Arize evaluations if available
#             if ARIZE_AVAILABLE:
#                 evaluated_results = self.run_arize_evaluations(agent_results)
#             else:
#                 evaluated_results = agent_results
            
#             results[scenario_name] = evaluated_results
            
#             # Print summary
#             total_queries = len(evaluated_results)
#             hotel_info_count = sum(evaluated_results["has_hotel_info"])
#             search_usage_count = sum(evaluated_results["uses_search"])
            
#             print(f"   ✅ Completed: {hotel_info_count}/{total_queries} queries with hotel info")
#             print(f"   🔍 Search usage: {search_usage_count}/{total_queries} queries")
            
#             if ARIZE_AVAILABLE and "arize_relevance" in evaluated_results.columns:
#                 avg_relevance = self._calculate_average_score(evaluated_results["arize_relevance"])
#                 avg_correctness = self._calculate_average_score(evaluated_results["arize_correctness"])
#                 print(f"   📊 Avg Relevance: {avg_relevance:.2f}")
#                 print(f"   📊 Avg Correctness: {avg_correctness:.2f}")
        
#         return results

#     def _calculate_average_score(self, scores: List[str]) -> float:
#         """Calculate average score from evaluation labels."""
#         if not scores:
#             return 0.0
        
#         # Map evaluation labels to numeric scores
#         score_map = {
#             "correct": 1.0,
#             "incorrect": 0.0,
#             "relevant": 1.0,
#             "irrelevant": 0.0,
#             "unknown": 0.5,
#         }
        
#         numeric_scores = [score_map.get(str(score).lower(), 0.5) for score in scores]
#         return sum(numeric_scores) / len(numeric_scores)

#     def generate_evaluation_report(self, results: Dict[str, pd.DataFrame]) -> str:
#         """
#         Generate a comprehensive evaluation report.
        
#         Args:
#             results: Dictionary of scenario results
            
#         Returns:
#             Formatted evaluation report
#         """
#         report = []
#         report.append("# Hotel Support Agent Evaluation Report")
#         report.append("=" * 60)
#         report.append("")
        
#         total_queries = 0
#         total_with_hotel_info = 0
#         total_search_usage = 0
        
#         for scenario_name, df in results.items():
#             report.append(f"## {scenario_name.replace('_', ' ').title()}")
#             report.append(f"- Total Queries: {len(df)}")
#             report.append(f"- With Hotel Info: {sum(df['has_hotel_info'])}")
#             report.append(f"- Search Usage: {sum(df['uses_search'])}")
#             report.append(f"- Success Rate: {sum(df['has_hotel_info']) / len(df) * 100:.1f}%")
            
#             # Arize metrics
#             if ARIZE_AVAILABLE and "arize_relevance" in df.columns:
#                 avg_relevance = self._calculate_average_score(df["arize_relevance"])
#                 avg_correctness = self._calculate_average_score(df["arize_correctness"])
#                 report.append(f"- Avg Relevance: {avg_relevance:.2f}")
#                 report.append(f"- Avg Correctness: {avg_correctness:.2f}")
            
#             # Response quality metrics
#             avg_response_length = df["response_length"].mean()
#             report.append(f"- Avg Response Length: {avg_response_length:.0f} chars")
            
#             report.append("")
            
#             total_queries += len(df)
#             total_with_hotel_info += sum(df["has_hotel_info"])
#             total_search_usage += sum(df["uses_search"])
        
#         report.append("## Overall Summary")
#         report.append(f"- Total Queries Evaluated: {total_queries}")
#         report.append(f"- Total With Hotel Info: {total_with_hotel_info}")
#         report.append(f"- Overall Success Rate: {total_with_hotel_info / total_queries * 100:.1f}%")
#         report.append(f"- Search Tool Usage: {total_search_usage / total_queries * 100:.1f}%")
#         report.append("")
        
#         if ARIZE_AVAILABLE and self.arize_client:
#             report.append("## Arize AI Dashboard")
#             report.append("View detailed traces and metrics in your Arize workspace:")
#             report.append(f"- Project: {PROJECT_NAME}")
#             report.append(f"- Space ID: {SPACE_ID}")
#             report.append("")
        
#         return "\n".join(report)

#     def log_experiment_to_arize(self, results_df: pd.DataFrame, experiment_name: str) -> str:
#         """
#         Log experiment results to Arize for analysis and comparison.
        
#         Args:
#             results_df: DataFrame with evaluation results
#             experiment_name: Name for the experiment
            
#         Returns:
#             Experiment ID if successful, None otherwise
#         """
#         try:
#             if not self.arize_client or not ARIZE_AVAILABLE:
#                 print("⚠️ No Arize client available, skipping logging")
#                 return None
            
#             # Define column mappings
#             task_cols = ExperimentTaskResultColumnNames(
#                 example_id="example_id",
#                 result="output"
#             )
            
#             evaluator_columns = {}
            
#             if "arize_relevance" in results_df.columns:
#                 evaluator_columns["arize_relevance"] = EvaluationResultColumnNames(
#                     label="arize_relevance",
#                     explanation="arize_relevance_explanation",
#                 )
            
#             if "arize_correctness" in results_df.columns:
#                 evaluator_columns["arize_correctness"] = EvaluationResultColumnNames(
#                     label="arize_correctness",
#                     explanation="arize_correctness_explanation",
#                 )
            
#             # Log experiment to Arize
#             experiment_id = self.arize_client.log_experiment(
#                 space_id=SPACE_ID,
#                 experiment_name=f"{experiment_name}-{str(uuid4())[:4]}",
#                 experiment_df=results_df,
#                 task_columns=task_cols,
#                 evaluator_columns=evaluator_columns,
#                 dataset_name=f"hotel-support-eval-{str(uuid4())[:8]}",
#             )
            
#             print(f"✅ Logged experiment to Arize: {experiment_name}")
#             return experiment_id
            
#         except Exception as e:
#             print(f"❌ Error logging experiment to Arize: {e}")
#             return None


# def eval_hotel_search_basic():
#     """
#     Evaluate hotel search agent with basic queries.
    
#     This function tests the agent's ability to handle standard hotel search
#     requests and provide appropriate recommendations.
#     """
#     print("🔍 Running basic hotel search evaluation...")
    
#     # Initialize Agent Catalog
#     catalog = agentc.Catalog()
#     root_span = catalog.Span(name="HotelSupportEvalBasic")
    
#     # Initialize evaluator
#     evaluator = ArizeHotelSupportEvaluator(catalog=catalog, span=root_span)
    
#     # Basic test queries
#     test_inputs = [
#         "Find hotels in New York",
#         "I need a hotel with a pool",
#         "Show me luxury hotels",
#         "Budget accommodation options",
#     ]
    
#     # Run evaluation
#     results = evaluator.run_hotel_search_evaluation(test_inputs)
    
#     # Run Arize evaluations if available
#     if ARIZE_AVAILABLE:
#         evaluated_results = evaluator.run_arize_evaluations(results)
#     else:
#         evaluated_results = results
    
#     # Calculate metrics
#     total_queries = len(evaluated_results)
#     hotel_info_count = sum(evaluated_results["has_hotel_info"])
    
#     print(f"✅ Basic hotel search evaluation completed:")
#     print(f"   - Total queries: {total_queries}")
#     print(f"   - With hotel info: {hotel_info_count}")
#     print(f"   - Success rate: {hotel_info_count / total_queries * 100:.1f}%")
    
#     # Log results
#     root_span["total_queries"] = total_queries
#     root_span["hotel_info_count"] = hotel_info_count
#     root_span["success_rate"] = hotel_info_count / total_queries


# def eval_hotel_search_comprehensive():
#     """
#     Run comprehensive evaluation across all hotel search scenarios.
    
#     This function provides a complete evaluation suite that tests the
#     agent across different types of hotel search queries and use cases.
#     """
#     print("🚀 Starting comprehensive hotel support agent evaluation...")
    
#     # Initialize Agent Catalog
#     catalog = agentc.Catalog()
#     root_span = catalog.Span(name="HotelSupportComprehensiveEval")
    
#     # Initialize evaluator
#     evaluator = ArizeHotelSupportEvaluator(catalog=catalog, span=root_span)
    
#     # Run comprehensive evaluation
#     results = evaluator.evaluate_hotel_search_scenarios()
    
#     # Generate report
#     report = evaluator.generate_evaluation_report(results)
    
#     # Save report
#     report_file = pathlib.Path("evals") / "hotel_evaluation_report.md"
#     with report_file.open("w") as f:
#         f.write(report)
    
#     print(f"✅ Comprehensive evaluation completed!")
#     print(f"   📊 Report saved to: {report_file}")
#     print("\n" + "=" * 60)
#     print(report)


# if __name__ == "__main__":
#     import sys
    
#     # Check if specific evaluation is requested
#     if len(sys.argv) > 1:
#         if sys.argv[1] == "basic":
#             eval_hotel_search_basic()
#         elif sys.argv[1] == "comprehensive":
#             eval_hotel_search_comprehensive()
#         else:
#             print("Usage: python eval_arize.py [basic|comprehensive]")
#     else:
#         # Run comprehensive evaluation by default
#         print("🔍 Running comprehensive hotel support agent evaluation...")
#         eval_hotel_search_comprehensive()