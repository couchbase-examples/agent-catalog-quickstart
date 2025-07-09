import base64
import json
import os
import time
from datetime import timedelta
import requests

import agentc
import agentc_langchain
import dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from couchbase.options import ClusterOptions
from langchain.agents import AgentExecutor, create_react_agent
from langchain.hub import pull
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import hotel data from the data module
from data.hotel_data import get_hotel_texts

# Make sure you populate your .env file with the correct credentials!
dotenv.load_dotenv(override=True)

def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        import getpass
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")

def setup_environment():
    required_vars = ['CB_HOST', 'CB_USERNAME', 'CB_PASSWORD', 'CB_BUCKET_NAME']
    for var in required_vars:
        _set_if_undefined(var)
    
    # Optional Capella AI variables (fallback to OpenAI if not provided)
    optional_vars = ['CAPELLA_API_ENDPOINT', 'CAPELLA_API_EMBEDDING_MODEL', 'CAPELLA_API_LLM_MODEL']
    for var in optional_vars:
        if not os.environ.get(var):
            print(f"ℹ️ {var} not provided - will use OpenAI fallback")
    
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
            os.environ[key] = input(f"Enter {key} (default: {default_value}): ") or default_value
    
    # Generate Capella AI API key from username and password if endpoint is provided
    if os.environ.get('CAPELLA_API_ENDPOINT'):
        os.environ['CAPELLA_API_KEY'] = base64.b64encode(f"{os.environ['CB_USERNAME']}:{os.environ['CB_PASSWORD']}".encode("utf-8")).decode("utf-8")
        
        # Ensure endpoint has /v1 suffix for OpenAI compatibility
        if not os.environ['CAPELLA_API_ENDPOINT'].endswith('/v1'):
            os.environ['CAPELLA_API_ENDPOINT'] = os.environ['CAPELLA_API_ENDPOINT'].rstrip('/') + '/v1'
            print(f"Added /v1 suffix to endpoint: {os.environ['CAPELLA_API_ENDPOINT']}")

def setup_couchbase_connection():
    try:
        auth = PasswordAuthenticator(os.environ['CB_USERNAME'], os.environ['CB_PASSWORD'])
        options = ClusterOptions(auth)
        # Use WAN profile for better timeout handling with remote clusters  
        options.apply_profile("wan_development")
        cluster = Cluster(os.environ['CB_HOST'], options)
        cluster.wait_until_ready(timedelta(seconds=15))  # Increased wait time
        print("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    try:
        try:
            bucket = cluster.bucket(bucket_name)
            print(f"Bucket '{bucket_name}' exists")
        except Exception:
            print(f"Creating bucket '{bucket_name}'...")
            bucket_settings = CreateBucketSettings(
                name=bucket_name,
                bucket_type='couchbase',
                ram_quota_mb=1024,
                flush_enabled=True,
                num_replicas=0
            )
            cluster.buckets().create_bucket(bucket_settings)
            time.sleep(5)
            bucket = cluster.bucket(bucket_name)
            print(f"Bucket '{bucket_name}' created successfully")

        bucket_manager = bucket.collections()
        
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)
        
        if not scope_exists and scope_name != "_default":
            print(f"Creating scope '{scope_name}'...")
            bucket_manager.create_scope(scope_name)
            print(f"Scope '{scope_name}' created successfully")

        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            print(f"Creating collection '{collection_name}'...")
            bucket_manager.create_collection(scope_name, collection_name)
            print(f"Collection '{collection_name}' created successfully")

        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(3)

        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            print("Primary index created successfully")
        except Exception as e:
            print(f"Warning: Error creating primary index: {str(e)}")

        print("Collection setup complete. Using existing documents in the database.")
        
        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")

def setup_vector_search_index(cluster, index_definition):
    try:
        scope_index_manager = cluster.bucket(os.environ['CB_BUCKET_NAME']).scope(os.environ['SCOPE_NAME']).search_indexes()
        
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]

        if index_name not in [index.name for index in existing_indexes]:
            print(f"Creating vector search index '{index_name}'...")
            search_index = SearchIndex.from_json(index_definition)
            scope_index_manager.upsert_index(search_index)
            print(f"Vector search index '{index_name}' created successfully")
        else:
            print(f"Vector search index '{index_name}' already exists")
    except Exception as e:
        raise RuntimeError(f"Error setting up vector search index: {str(e)}")

def setup_vector_store(cluster):
    try:
        # Try Capella AI embeddings first
        try:
            embeddings = OpenAIEmbeddings(
                api_key=os.environ['CAPELLA_API_KEY'],
                base_url=os.environ['CAPELLA_API_ENDPOINT'],
                model=os.environ['CAPELLA_API_EMBEDDING_MODEL']
            )
            # Test the embeddings work
            test_embedding = embeddings.embed_query("test")
            print(f"✅ Using Capella AI embeddings (dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"⚠️ Capella AI embeddings failed: {str(e)}")
            print("❌ Cannot fall back to OpenAI embeddings - dimension mismatch!")
            raise Exception("Embedding dimension mismatch - cannot proceed with OpenAI fallback")
        
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ['CB_BUCKET_NAME'],
            scope_name=os.environ['SCOPE_NAME'],
            collection_name=os.environ['COLLECTION_NAME'],
            embedding=embeddings,
            index_name=os.environ['INDEX_NAME'],
        )
        
        # Clear existing data before loading fresh hotel data
        clear_collection_data(cluster)
        
        # Use hotel data from the data module
        hotel_data = get_hotel_texts()
        
        try:
            vector_store.add_texts(texts=hotel_data, batch_size=10)
            print("Hotel data loaded into vector store successfully")
        except Exception as e:
            print(f"Warning: Error loading hotel data: {str(e)}. Vector store created but data not loaded.")
        
        return vector_store
    except Exception as e:
        raise ValueError(f"Error setting up vector store: {str(e)}")

def clear_collection_data(cluster):
    """Clear all documents from the collection to start fresh."""
    try:
        bucket_name = os.environ['CB_BUCKET_NAME']
        scope_name = os.environ['SCOPE_NAME']
        collection_name = os.environ['COLLECTION_NAME']
        
        # Delete all documents in the collection
        delete_query = f"DELETE FROM `{bucket_name}`.`{scope_name}`.`{collection_name}`"
        result = cluster.query(delete_query)
        
        print(f"Cleared existing data from collection {scope_name}.{collection_name}")
        
    except Exception as e:
        print(f"Warning: Could not clear collection data: {str(e)}. Continuing with existing data...")

def test_capella_connectivity():
    """Test Capella API connectivity for both embeddings and LLM before running main demo."""
    print("🔍 Testing Capella API connectivity...")
    print(f"Endpoint: {os.environ['CAPELLA_API_ENDPOINT']}")
    print(f"Embedding Model: {os.environ['CAPELLA_API_EMBEDDING_MODEL']}")
    print(f"LLM Model: {os.environ['CAPELLA_API_LLM_MODEL']}")
    print(f"Username: {os.environ['CB_USERNAME']}")
    print(f"API Key (first 20 chars): {os.environ['CAPELLA_API_KEY'][:20]}...")
    
    # First test basic HTTP connectivity
    try:
        print("Testing basic HTTP connectivity...")
        response = requests.get(f"{os.environ['CAPELLA_API_ENDPOINT']}/models", 
                               headers={"Authorization": f"Bearer {os.environ['CAPELLA_API_KEY']}"}, 
                               timeout=10)
        print(f"HTTP response status: {response.status_code}")
        if response.status_code != 200:
            print(f"HTTP response: {response.text[:200]}...")
    except Exception as e:
        print(f"HTTP test failed: {str(e)}")
    
    # Test embedding model
    try:
        print("Testing embedding model...")
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['CAPELLA_API_KEY'],
            base_url=os.environ['CAPELLA_API_ENDPOINT'],
            model=os.environ['CAPELLA_API_EMBEDDING_MODEL']
        )
        test_embedding = embeddings.embed_query("test connectivity")
        print(f"✅ Embedding model working - dimension: {len(test_embedding)}")
    except Exception as e:
        print(f"❌ Embedding model failed: {str(e)}")
        return False
    
    # Test LLM model
    try:
        print("Testing LLM model...")
        llm = ChatOpenAI(
            api_key=os.environ['CAPELLA_API_KEY'],
            base_url=os.environ['CAPELLA_API_ENDPOINT'],
            model=os.environ['CAPELLA_API_LLM_MODEL'],
            temperature=0
        )
        response = llm.invoke("Say 'Hello' if you can hear me")
        print(f"✅ LLM model working - response: {response.content[:50]}...")
    except Exception as e:
        print(f"❌ LLM model failed: {str(e)}")
        return False
    
    print("✅ All Capella API tests passed!")
    return True

def main():
    try:
        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Hotel Search Agent")

        with application_span.new("Environment Setup"):
            setup_environment()
            
        with application_span.new("Capella API Test"):
            if os.environ.get('CAPELLA_API_ENDPOINT'):
                if not test_capella_connectivity():
                    print("❌ Capella API connectivity test failed. Will use OpenAI fallback.")
            else:
                print("ℹ️ Capella API not configured - will use OpenAI models")
        
        with application_span.new("Couchbase Connection"):
            cluster = setup_couchbase_connection()
        
        with application_span.new("Couchbase Collection Setup"):
            setup_collection(
                cluster, 
                os.environ['CB_BUCKET_NAME'], 
                os.environ['SCOPE_NAME'], 
                os.environ['COLLECTION_NAME']
            )
        
        with application_span.new("Vector Index Setup"):
            try:
                with open('agentcatalog_index.json', 'r') as file:
                    index_definition = json.load(file)
                print("Loaded vector search index definition from agentcatalog_index.json")
            except Exception as e:
                raise ValueError(f"Error loading index definition: {str(e)}")
            
            setup_vector_search_index(cluster, index_definition)
        
        with application_span.new("Vector Store Setup"):
            setup_vector_store(cluster)
        
        with application_span.new("LLM Setup"):
            # Setup LLM with Agent Catalog callback - try Capella AI first, fallback to OpenAI
            try:
                llm = ChatOpenAI(
                    api_key=os.environ['CAPELLA_API_KEY'],
                    base_url=os.environ['CAPELLA_API_ENDPOINT'],
                    model=os.environ['CAPELLA_API_LLM_MODEL'],
                    temperature=0,
                    callbacks=[agentc_langchain.chat.Callback(span=application_span)]
                )
                # Test the LLM works
                llm.invoke("Hello")
                print("✅ Using Capella AI LLM")
            except Exception as e:
                print(f"⚠️ Capella AI LLM failed: {str(e)}")
                print("🔄 Falling back to OpenAI LLM...")
                _set_if_undefined("OPENAI_API_KEY")
                llm = ChatOpenAI(
                    api_key=os.environ['OPENAI_API_KEY'],
                    model="gpt-4o",
                    temperature=0,
                    callbacks=[agentc_langchain.chat.Callback(span=application_span)]
                )
                print("✅ Using OpenAI LLM as fallback")
        
        with application_span.new("Tool Loading"):
            # Load tools from Agent Catalog - they are now properly decorated
            tool_search = catalog.find("tool", name="search_vector_database")
            tool_details = catalog.find("tool", name="get_hotel_details")
            
            if not tool_search:
                raise ValueError("Could not find search_vector_database tool. Make sure it's indexed with 'agentc index tools/'")
            if not tool_details:
                raise ValueError("Could not find get_hotel_details tool. Make sure it's indexed with 'agentc index tools/'")
            
            from langchain_core.tools import Tool
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
        
        with application_span.new("Agent Creation"):
            # Get prompt from Agent Catalog
            hotel_prompt = catalog.find("prompt", name="hotel_search_assistant")
            if not hotel_prompt:
                raise ValueError("Could not find hotel_search_assistant prompt in catalog. Make sure it's indexed with 'agentc index prompts/'")
            
            # Create a custom prompt using the catalog prompt content
            from langchain_core.prompts import PromptTemplate
            
            # The prompt content is already properly structured with ReAct format
            prompt_content = hotel_prompt.content.strip()
            
            custom_prompt = PromptTemplate(
                template=prompt_content,
                input_variables=["input", "agent_scratchpad"],
                partial_variables={
                    "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
                    "tool_names": ", ".join([tool.name for tool in tools])
                }
            )
            
            agent = create_react_agent(llm, tools, custom_prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                handle_parsing_errors=True,
                max_iterations=8,
                return_intermediate_steps=True,
                early_stopping_method="force",  # Changed from "generate" to "force"
                max_execution_time=60  # 60 second timeout to prevent hanging
            )
        
        # Test the agent with sample queries
        print("\nHotel Search Agent is ready!")
        print("Testing with sample queries...")
        
        test_queries = [
            "Find me a luxury hotel with a pool and spa",
            "I need a beach resort in Miami", 
            "Get me details about Ocean Breeze Resort"
        ]
        
        with application_span.new("Query Execution") as span:
            for query in test_queries:
                with span.new(f"Query: {query}") as query_span:
                    print(f"\n🔍 Query: {query}")
                    try:
                        response = agent_executor.invoke({"input": query})
                        query_span["response"] = response['output']
                        print(f"✅ Response: {response['output']}")
                        print("-" * 80)
                    except Exception as e:
                        query_span["error"] = str(e)
                        print(f"❌ Error: {e}")
                        print("-" * 80)
                
    except Exception as e:
        print(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
