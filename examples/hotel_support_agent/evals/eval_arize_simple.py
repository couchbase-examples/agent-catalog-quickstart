#!/usr/bin/env python3
"""
Hotel Support Agent Evaluation with Arize Phoenix
Real implementation using agentc and Couchbase vector store
"""

import os
import sys
import json
import base64
import warnings
from datetime import timedelta
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Environment setup
import dotenv
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Phoenix setup
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

# Agent Catalog imports
import agentc
import agentc_langchain

# Couchbase imports
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions

# LangChain imports
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseVectorStore

# Hotel data import
from data.hotel_data import get_hotel_texts

# Evaluation imports
import pandas as pd
import numpy as np
from phoenix.evals import (
    RelevanceEvaluator, 
    QAEvaluator,
    OpenAIModel,
    run_evals
)


class HotelAgentEvaluator:
    """Evaluates hotel search agent using real agentc and Couchbase setup"""
    
    def __init__(self):
        self.phoenix_session = None
        self.cluster = None
        self.agent_executor = None
        self.catalog = None
        self.application_span = None
        
        # Initialize Phoenix evaluators
        eval_model = OpenAIModel(model="gpt-4o")
        self.relevance_evaluator = RelevanceEvaluator(eval_model)
        self.qa_evaluator = QAEvaluator(eval_model)
        
    def setup_phoenix(self):
        """Setup Phoenix tracing"""
        try:
            # Launch Phoenix app
            self.phoenix_session = px.launch_app()
            print(f"Phoenix UI running at: {self.phoenix_session.url}")
            
            # Setup resource for traces
            resource = Resource.create({"service.name": "hotel-agent-evaluation"})
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Setup Phoenix instrumentation
            LangChainInstrumentor().instrument()
            OpenAIInstrumentor().instrument()
            
            print("✅ Phoenix setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Phoenix setup failed: {e}")
            return False
    
    def setup_environment(self):
        """Setup environment variables with robust .env file loading"""
        try:
            # Find and load .env files from multiple locations
            env_files_loaded = self._load_env_files()
            
            # Map environment variable names for compatibility
            self._map_environment_variables()
            
            # Set default values for missing variables
            self._set_default_values()
            
            # Setup Capella AI credentials if available
            self._setup_capella_credentials()
            
            print(f"✅ Environment setup complete (loaded {env_files_loaded} .env files)")
            return True
            
        except Exception as e:
            print(f"❌ Environment setup failed: {e}")
            return False
    
    def _load_env_files(self) -> int:
        """Load .env files from multiple locations"""
        env_files_loaded = 0
        
        # Current directory (evals)
        current_dir = os.getcwd()
        
        # Parent directory (hotel_support_agent)
        parent_dir = os.path.dirname(current_dir)
        
        # Project root (agent-catalog-quickstart)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(parent_dir)))
        
        # Search locations in order of priority
        search_paths = [
            current_dir,                    # ./evals/
            parent_dir,                     # ./hotel_support_agent/
            project_root,                   # ./agent-catalog-quickstart/
            os.path.join(project_root, "examples", "hotel_support_agent"),  # explicit path
        ]
        
        for search_path in search_paths:
            env_file_path = os.path.join(search_path, ".env")
            if os.path.exists(env_file_path):
                try:
                    load_dotenv(env_file_path, override=True)
                    print(f"📄 Loaded .env from: {env_file_path}")
                    env_files_loaded += 1
                except Exception as e:
                    print(f"⚠️ Failed to load .env from {env_file_path}: {e}")
        
        if env_files_loaded == 0:
            print("⚠️ No .env files found, using system environment variables")
        
        return env_files_loaded
    
    def _map_environment_variables(self):
        """Map different environment variable naming conventions"""
        # Map Couchbase variables to consistent naming
        var_mappings = {
            'CB_HOST': ['COUCHBASE_CONNECTION_STRING', 'CB_CONN_STRING', 'AGENT_CATALOG_CONN_STRING'],
            'CB_USERNAME': ['COUCHBASE_USERNAME', 'AGENT_CATALOG_USERNAME'],
            'CB_PASSWORD': ['COUCHBASE_PASSWORD', 'AGENT_CATALOG_PASSWORD'],
            'CB_BUCKET_NAME': ['AGENT_CATALOG_BUCKET'],
        }
        
        for target_var, source_vars in var_mappings.items():
            if not os.environ.get(target_var):
                for source_var in source_vars:
                    if os.environ.get(source_var):
                        os.environ[target_var] = os.environ[source_var]
                        print(f"📝 Mapped {source_var} -> {target_var}")
                        break
    
    def _set_default_values(self):
        """Set default values for missing environment variables"""
        defaults = {
            'CB_HOST': 'couchbase://localhost',
            'CB_USERNAME': 'Administrator',
            'CB_PASSWORD': 'password',
            'CB_BUCKET_NAME': 'vector-search-testing',
            'INDEX_NAME': 'vector_search_agentcatalog',
            'SCOPE_NAME': 'shared',
            'COLLECTION_NAME': 'agentcatalog'
        }
        
        for key, default_value in defaults.items():
            if not os.environ.get(key):
                os.environ[key] = default_value
                print(f"🔧 Set default {key} = {default_value}")
    
    def _setup_capella_credentials(self):
        """Setup Capella AI credentials if endpoint is available"""
        if os.environ.get('CAPELLA_API_ENDPOINT'):
            username = os.environ.get('CB_USERNAME')
            password = os.environ.get('CB_PASSWORD')
            
            if username and password:
                os.environ['CAPELLA_API_KEY'] = base64.b64encode(
                    f"{username}:{password}".encode("utf-8")
                ).decode("utf-8")
                
                # Ensure endpoint has /v1 suffix
                endpoint = os.environ['CAPELLA_API_ENDPOINT']
                if not endpoint.endswith('/v1'):
                    os.environ['CAPELLA_API_ENDPOINT'] = endpoint.rstrip('/') + '/v1'
                    print(f"🔧 Added /v1 suffix to Capella endpoint")
                
                print("🔑 Capella AI credentials configured")
    
    def setup_couchbase(self):
        """Setup Couchbase cluster and vector store"""
        try:
            # Setup cluster connection
            auth = PasswordAuthenticator(os.environ['CB_USERNAME'], os.environ['CB_PASSWORD'])
            options = ClusterOptions(auth)
            options.apply_profile("wan_development")
            
            self.cluster = Cluster(os.environ['CB_HOST'], options)
            self.cluster.wait_until_ready(timedelta(seconds=15))
            
            # Setup bucket and collection
            bucket_name = os.environ['CB_BUCKET_NAME']
            scope_name = os.environ['SCOPE_NAME']
            collection_name = os.environ['COLLECTION_NAME']
            
            # Create bucket if needed
            try:
                bucket = self.cluster.bucket(bucket_name)
                print(f"✅ Bucket '{bucket_name}' exists")
            except Exception:
                print(f"Creating bucket '{bucket_name}'...")
                bucket_settings = CreateBucketSettings(
                    name=bucket_name,
                    bucket_type='couchbase',
                    ram_quota_mb=1024,
                    flush_enabled=True,
                    num_replicas=0
                )
                self.cluster.buckets().create_bucket(bucket_settings)
                bucket = self.cluster.bucket(bucket_name)
            
            # Setup collection
            bucket_manager = bucket.collections()
            scopes = bucket_manager.get_all_scopes()
            
            if not any(scope.name == scope_name for scope in scopes) and scope_name != "_default":
                bucket_manager.create_scope(scope_name)
                print(f"✅ Scope '{scope_name}' created")
            
            collections = bucket_manager.get_all_scopes()
            collection_exists = any(
                scope.name == scope_name and collection_name in [col.name for col in scope.collections]
                for scope in collections
            )
            
            if not collection_exists:
                bucket_manager.create_collection(scope_name, collection_name)
                print(f"✅ Collection '{collection_name}' created")
            
            # Setup search index
            try:
                index_file_path = self._find_index_file()
                if index_file_path:
                    with open(index_file_path, 'r') as file:
                        index_definition = json.load(file)
                    
                    scope_index_manager = bucket.scope(scope_name).search_indexes()
                    existing_indexes = scope_index_manager.get_all_indexes()
                    index_name = index_definition["name"]
                    
                    if index_name not in [index.name for index in existing_indexes]:
                        print(f"Creating vector search index '{index_name}'...")
                        search_index = SearchIndex.from_json(index_definition)
                        scope_index_manager.upsert_index(search_index)
                        print(f"✅ Vector search index '{index_name}' created")
                    else:
                        print(f"✅ Vector search index '{index_name}' exists")
                else:
                    print("⚠️ agentcatalog_index.json not found, skipping index setup")
                    
            except Exception as e:
                print(f"⚠️ Index setup warning: {e}")
            
            print("✅ Couchbase setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Couchbase setup failed: {e}")
            return False
    
    def _find_index_file(self) -> str:
        """Find the agentcatalog_index.json file in various locations"""
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(parent_dir)))
        
        search_paths = [
            current_dir,
            parent_dir,
            project_root,
            os.path.join(project_root, "examples", "hotel_support_agent"),
        ]
        
        for search_path in search_paths:
            index_file_path = os.path.join(search_path, "agentcatalog_index.json")
            if os.path.exists(index_file_path):
                print(f"📄 Found agentcatalog_index.json at: {index_file_path}")
                return index_file_path
        
        return None
    
    def setup_vector_store(self):
        """Setup vector store with hotel data"""
        try:
            # Setup embeddings - try Capella AI first
            embeddings = None
            try:
                if os.environ.get('CAPELLA_API_KEY') and os.environ.get('CAPELLA_API_ENDPOINT'):
                    embeddings = OpenAIEmbeddings(
                        api_key=os.environ['CAPELLA_API_KEY'],
                        base_url=os.environ['CAPELLA_API_ENDPOINT'],
                        model=os.environ.get('CAPELLA_API_EMBEDDING_MODEL', 'text-embedding-3-large')
                    )
                    # Test embeddings
                    test_embedding = embeddings.embed_query("test")
                    print(f"✅ Using Capella AI embeddings (dimension: {len(test_embedding)})")
                else:
                    raise Exception("Capella credentials not available")
            except Exception as e:
                print(f"⚠️ Capella AI embeddings failed: {e}")
                print("🔄 Falling back to OpenAI embeddings...")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                print("✅ Using OpenAI embeddings")
            
            # Setup vector store
            vector_store = CouchbaseVectorStore(
                cluster=self.cluster,
                bucket_name=os.environ['CB_BUCKET_NAME'],
                scope_name=os.environ['SCOPE_NAME'],
                collection_name=os.environ['COLLECTION_NAME'],
                embedding=embeddings,
                index_name=os.environ['INDEX_NAME'],
            )
            
            # Load hotel data
            hotel_data = get_hotel_texts()
            if hotel_data:
                try:
                    vector_store.add_texts(texts=hotel_data, batch_size=10)
                    print(f"✅ Loaded {len(hotel_data)} hotel records into vector store")
                except Exception as e:
                    print(f"⚠️ Hotel data loading warning: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Vector store setup failed: {e}")
            return False
    
    def setup_agent(self):
        """Setup the hotel search agent using agentc"""
        try:
            # Initialize Agent Catalog
            self.catalog = agentc.Catalog()
            self.application_span = self.catalog.Span(name="Hotel Search Agent Evaluation")
            
            # Setup LLM
            llm = None
            try:
                if os.environ.get('CAPELLA_API_KEY') and os.environ.get('CAPELLA_API_ENDPOINT'):
                    llm = ChatOpenAI(
                        api_key=os.environ['CAPELLA_API_KEY'],
                        base_url=os.environ['CAPELLA_API_ENDPOINT'],
                        model=os.environ.get('CAPELLA_API_LLM_MODEL', 'gpt-4o'),
                        temperature=0,
                        callbacks=[agentc_langchain.chat.Callback(span=self.application_span)]
                    )
                    # Test LLM
                    llm.invoke("Hello")
                    print("✅ Using Capella AI LLM")
                else:
                    raise Exception("Capella credentials not available")
            except Exception as e:
                print(f"⚠️ Capella AI LLM failed: {e}")
                print("🔄 Falling back to OpenAI LLM...")
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0,
                    callbacks=[agentc_langchain.chat.Callback(span=self.application_span)]
                )
                print("✅ Using OpenAI LLM")
            
            # Load tools from Agent Catalog
            tool_search = self.catalog.find("tool", name="search_vector_database")
            tool_details = self.catalog.find("tool", name="get_hotel_details")
            
            if not tool_search or not tool_details:
                print("⚠️ Tools not found in catalog, checking if they need to be indexed...")
                # Run agentc index to make sure tools are available
                import subprocess
                subprocess.run(["agentc", "index", "tools/"], cwd="../", capture_output=True)
                subprocess.run(["agentc", "index", "prompts/"], cwd="../", capture_output=True)
                
                # Try loading tools again
                tool_search = self.catalog.find("tool", name="search_vector_database")
                tool_details = self.catalog.find("tool", name="get_hotel_details")
            
            if not tool_search or not tool_details:
                raise ValueError("Could not find required tools in catalog")
            
            # Create LangChain tools
            tools = [
                Tool(
                    name=tool_search.meta.name,
                    description=tool_search.meta.description,
                    func=tool_search.func
                ),
                Tool(
                    name=tool_details.meta.name,
                    description=tool_details.meta.description,
                    func=tool_details.func
                )
            ]
            
            # Load prompt from Agent Catalog
            hotel_prompt = self.catalog.find("prompt", name="hotel_search_assistant")
            if not hotel_prompt:
                raise ValueError("Could not find hotel_search_assistant prompt in catalog")
            
            # Create custom prompt
            prompt_content = hotel_prompt.content.strip()
            custom_prompt = PromptTemplate(
                template=prompt_content,
                input_variables=["input", "agent_scratchpad"],
                partial_variables={
                    "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    "tool_names": ", ".join([tool.name for tool in tools])
                }
            )
            
            # Create agent
            agent = create_react_agent(llm, tools, custom_prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=8,
                return_intermediate_steps=True,
                early_stopping_method="force",
                max_execution_time=60
            )
            
            print("✅ Agent setup complete")
            return True
            
        except Exception as e:
            print(f"❌ Agent setup failed: {e}")
            return False
    
    def get_evaluation_queries(self) -> List[Dict[str, Any]]:
        """Get evaluation queries for testing - using actual hotels from our data"""
        return [
            {
                "query": "Find me luxury hotels with spa and pool",
                "expected_type": "luxury_amenities",
                "reference_answer": "Luxury hotels with spa and pool amenities like Grand Palace Hotel, Ocean Breeze Resort, and Wellness Retreat"
            },
            {
                "query": "I need a budget hotel under $300 per night",
                "expected_type": "budget_search",
                "reference_answer": "Budget-friendly hotels under $300 like Mountain Lodge in Aspen"
            },
            {
                "query": "Get me details about Grand Palace Hotel",
                "expected_type": "hotel_details",
                "reference_answer": "Detailed information about Grand Palace Hotel including amenities, pricing, and contact details"
            },
            {
                "query": "Find family-friendly hotels with kids activities",
                "expected_type": "family_search",
                "reference_answer": "Family-friendly hotels with kids activities like Seaside Resort with Kids Club"
            },
            {
                "query": "Show me hotels with rooftop amenities",
                "expected_type": "amenity_search",
                "reference_answer": "Hotels with rooftop amenities like Grand Palace Hotel with rooftop pool and City Loft Hotel with rooftop bar"
            },
            {
                "query": "What is the cancellation policy for Ocean Breeze Resort?",
                "expected_type": "policy_inquiry",
                "reference_answer": "Ocean Breeze Resort offers free cancellation up to 72 hours before check-in"
            }
        ]
    
    def evaluate_response_quality(self, query: str, response: str, expected_type: str) -> float:
        """Evaluate response quality based on hotel search criteria"""
        try:
            score = 0.0
            response_lower = response.lower()
            
            # Base score for providing a response
            if len(response) > 50:
                score += 2.0
            
            # Content quality scoring
            if expected_type == "luxury_amenities":
                if any(term in response_lower for term in ["spa", "pool", "luxury", "premium", "amenities"]):
                    score += 3.0
                if any(term in response_lower for term in ["grand palace", "ocean breeze", "wellness retreat"]):
                    score += 2.0
                if any(term in response_lower for term in ["price", "cost", "rate"]):
                    score += 2.0
                if "hotel" in response_lower:
                    score += 1.0
                    
            elif expected_type == "budget_search":
                if any(term in response_lower for term in ["budget", "affordable", "cheap", "under", "$"]):
                    score += 3.0
                if any(term in response_lower for term in ["mountain lodge", "aspen", "colorado"]):
                    score += 2.0
                if any(term in response_lower for term in ["300", "price", "cost", "150"]):
                    score += 2.0
                if "hotel" in response_lower:
                    score += 1.0
                    
            elif expected_type == "hotel_details":
                if any(term in response_lower for term in ["grand hotel", "details", "amenities"]):
                    score += 3.0
                if any(term in response_lower for term in ["address", "contact", "phone"]):
                    score += 2.0
                if any(term in response_lower for term in ["price", "rate", "cost"]):
                    score += 2.0
                if any(term in response_lower for term in ["check-in", "check-out", "policy"]):
                    score += 1.0
                    
            elif expected_type == "family_search":
                if any(term in response_lower for term in ["family", "kids", "children", "activities"]):
                    score += 3.0
                if any(term in response_lower for term in ["seaside resort", "kids club", "miami"]):
                    score += 2.0
                if any(term in response_lower for term in ["playground", "pool", "entertainment"]):
                    score += 2.0
                if "hotel" in response_lower:
                    score += 1.0
                    
            elif expected_type == "amenity_search":
                if any(term in response_lower for term in ["rooftop", "pool", "bar", "amenities"]):
                    score += 3.0
                if any(term in response_lower for term in ["grand palace", "city loft", "hotel"]):
                    score += 2.0
                if any(term in response_lower for term in ["view", "dining", "facilities"]):
                    score += 2.0
                    
            elif expected_type == "policy_inquiry":
                if any(term in response_lower for term in ["cancellation", "policy", "cancel"]):
                    score += 3.0
                if any(term in response_lower for term in ["ocean view", "resort"]):
                    score += 2.0
                if any(term in response_lower for term in ["free", "charge", "fee", "refund"]):
                    score += 2.0
            
            # Bonus for professional response
            if any(term in response_lower for term in ["recommend", "suggest", "available", "assist"]):
                score += 1.0
            
            return min(score, 10.0)  # Cap at 10.0
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return 5.0  # Default score on error
    
    def run_evaluation(self):
        """Run the complete evaluation"""
        print("🚀 Starting Hotel Agent Evaluation with Real Implementation")
        print("=" * 70)
        
        # Setup phases
        setup_steps = [
            ("Phoenix", self.setup_phoenix),
            ("Environment", self.setup_environment),
            ("Couchbase", self.setup_couchbase),
            ("Vector Store", self.setup_vector_store),
            ("Agent", self.setup_agent)
        ]
        
        for step_name, setup_func in setup_steps:
            print(f"\n📋 Setting up {step_name}...")
            if not setup_func():
                print(f"❌ Failed to setup {step_name}. Aborting evaluation.")
                return
        
        # Run evaluation queries
        print("\n🔍 Running Evaluation Queries...")
        print("=" * 70)
        
        queries = self.get_evaluation_queries()
        results = []
        
        for i, query_data in enumerate(queries, 1):
            query = query_data["query"]
            expected_type = query_data["expected_type"]
            reference_answer = query_data["reference_answer"]
            
            print(f"\n📝 Query {i}/{len(queries)}: {query}")
            print("-" * 50)
            
            try:
                with self.application_span.new(f"Evaluation Query {i}") as query_span:
                    query_span["query"] = query
                    query_span["expected_type"] = expected_type
                    
                    # Execute query
                    response = self.agent_executor.invoke({"input": query})
                    output = response.get('output', 'No response generated')
                    
                    # Evaluate using Phoenix evaluators
                    relevance_record = {
                        "input": query,
                        "reference": reference_answer
                    }
                    relevance_result = self.relevance_evaluator.evaluate(relevance_record)
                    relevance_score = relevance_result[1] if relevance_result[1] is not None else 0.0
                    
                    qa_record = {
                        "input": query,
                        "reference": reference_answer,
                        "output": output
                    }
                    qa_result = self.qa_evaluator.evaluate(qa_record)
                    qa_score = qa_result[1] if qa_result[1] is not None else 0.0
                    
                    # Custom quality score for hotel-specific criteria
                    quality_score = self.evaluate_response_quality(query, output, expected_type)
                    
                    # Store results
                    result = {
                        'query': query,
                        'expected_type': expected_type,
                        'response': output,
                        'relevance_score': relevance_score,
                        'qa_score': qa_score,
                        'quality_score': quality_score,
                        'success': True
                    }
                    results.append(result)
                    
                    query_span["response"] = output
                    query_span["relevance_score"] = relevance_score
                    query_span["qa_score"] = qa_score
                    query_span["quality_score"] = quality_score
                    query_span["success"] = True
                    
                    print(f"✅ Response: {output[:100]}...")
                    print(f"📊 Relevance: {relevance_score:.1f}/10.0 | QA: {qa_score:.1f}/10.0 | Quality: {quality_score:.1f}/10.0")
                    
            except Exception as e:
                print(f"❌ Query failed: {e}")
                result = {
                    'query': query,
                    'expected_type': expected_type,
                    'response': f"Error: {str(e)}",
                    'relevance_score': 0.0,
                    'qa_score': 0.0,
                    'quality_score': 0.0,
                    'success': False
                }
                results.append(result)
        
        # Generate summary
        print("\n📊 Evaluation Summary")
        print("=" * 70)
        
        successful_queries = [r for r in results if r['success']]
        success_rate = len(successful_queries) / len(results) * 100
        
        if successful_queries:
            avg_relevance = sum(r['relevance_score'] for r in successful_queries) / len(successful_queries)
            avg_qa = sum(r['qa_score'] for r in successful_queries) / len(successful_queries)
            avg_quality = sum(r['quality_score'] for r in successful_queries) / len(successful_queries)
            
            print(f"✅ Success Rate: {success_rate:.1f}% ({len(successful_queries)}/{len(results)} queries)")
            print(f"📈 Average Scores:")
            print(f"   🎯 Relevance: {avg_relevance:.1f}/10.0")
            print(f"   💬 QA: {avg_qa:.1f}/10.0")
            print(f"   ⭐ Quality: {avg_quality:.1f}/10.0")
            
            # Individual scores
            print("\n📋 Individual Query Scores:")
            for i, result in enumerate(results, 1):
                status = "✅" if result['success'] else "❌"
                if result['success']:
                    print(f"  {i}. {status} R:{result['relevance_score']:.1f} Q:{result['qa_score']:.1f} H:{result['quality_score']:.1f} - {result['expected_type']}")
                else:
                    print(f"  {i}. {status} 0.0/10.0 - {result['expected_type']}")
        else:
            print("❌ No successful queries")
        
        # Export results
        try:
            df = pd.DataFrame(results)
            df.to_csv('hotel_agent_evaluation_results.csv', index=False)
            print(f"\n💾 Results exported to 'hotel_agent_evaluation_results.csv'")
        except Exception as e:
            print(f"⚠️ Failed to export results: {e}")
        
        print(f"\n🌐 Phoenix UI: {self.phoenix_session.url if self.phoenix_session else 'Not available'}")
        print("\n✅ Evaluation complete!")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.phoenix_session:
                print("🧹 Cleaning up Phoenix session...")
                # Phoenix sessions are automatically managed
        except Exception as e:
            print(f"Warning during cleanup: {e}")


def main():
    """Main evaluation function"""
    evaluator = HotelAgentEvaluator()
    
    try:
        evaluator.run_evaluation()
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    finally:
        evaluator.cleanup()


if __name__ == "__main__":
    main() 