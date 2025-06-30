import agentc
import agentc_langchain
import dotenv
import os
import json
import time
from datetime import timedelta

from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseVectorStore
from langchain.agents import create_react_agent, AgentExecutor
from langchain.hub import pull
from langchain_core.messages import HumanMessage

# Make sure you populate your .env file with the correct credentials!
dotenv.load_dotenv()

def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        import getpass
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")

def setup_environment():
    required_vars = ['OPENAI_API_KEY', 'CB_HOST', 'CB_USERNAME', 'CB_PASSWORD', 'CB_BUCKET_NAME']
    for var in required_vars:
        _set_if_undefined(var)
    
    defaults = {
        'CB_HOST': 'couchbase://localhost',
        'CB_USERNAME': 'Administrator', 
        'CB_PASSWORD': 'password',
        'CB_BUCKET_NAME': 'vector-search-testing',
        'INDEX_NAME': 'vector_search_deepseek',
        'SCOPE_NAME': 'shared',
        'COLLECTION_NAME': 'deepseek'
    }
    
    for key, default_value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = input(f"Enter {key} (default: {default_value}): ") or default_value

def setup_couchbase_connection():
    try:
        auth = PasswordAuthenticator(os.environ['CB_USERNAME'], os.environ['CB_PASSWORD'])
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ['CB_HOST'], options)
        cluster.wait_until_ready(timedelta(seconds=10))
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

        print(f"Collection setup complete. Using existing documents in the database.")
        
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

def load_hotel_data():
    try:
        hotels_data = [
            {
                "name": "Grand Palace Hotel",
                "location": "New York City",
                "description": "Luxury hotel in Manhattan with stunning city views, world-class amenities, and exceptional service.",
                "price_range": "$300-$500",
                "amenities": ["Pool", "Spa", "Gym", "Restaurant", "Room Service", "WiFi"],
                "rating": 4.8
            },
            {
                "name": "Seaside Resort",
                "location": "Miami Beach",
                "description": "Beautiful beachfront resort with private beach access, multiple pools, and ocean view rooms.",
                "price_range": "$200-$400",
                "amenities": ["Beach Access", "Pool", "Restaurant", "Bar", "Water Sports", "WiFi"],
                "rating": 4.6
            },
            {
                "name": "Mountain Lodge",
                "location": "Aspen",
                "description": "Cozy mountain lodge perfect for skiing and hiking, featuring rustic charm and mountain views.",
                "price_range": "$150-$300",
                "amenities": ["Ski Access", "Fireplace", "Restaurant", "Hot Tub", "Hiking Trails", "WiFi"],
                "rating": 4.5
            },
            {
                "name": "Business Center Hotel",
                "location": "Chicago",
                "description": "Modern business hotel in downtown Chicago with state-of-the-art conference facilities.",
                "price_range": "$180-$280",
                "amenities": ["Business Center", "Meeting Rooms", "Gym", "Restaurant", "WiFi", "Parking"],
                "rating": 4.3
            },
            {
                "name": "Boutique Inn",
                "location": "San Francisco",
                "description": "Charming boutique hotel in the heart of San Francisco with unique decor and personalized service.",
                "price_range": "$220-$350",
                "amenities": ["Concierge", "Restaurant", "Bar", "WiFi", "Pet Friendly", "Valet Parking"],
                "rating": 4.7
            }
        ]
        
        hotel_texts = []
        for hotel in hotels_data:
            text = f"{hotel['name']} in {hotel['location']}. {hotel['description']} Price range: {hotel['price_range']}. Rating: {hotel['rating']}/5. Amenities: {', '.join(hotel['amenities'])}"
            hotel_texts.append(text)
        
        return hotel_texts
    except Exception as e:
        raise ValueError(f"Error loading hotel data: {str(e)}")

def setup_vector_store(cluster):
    try:
        embeddings = OpenAIEmbeddings(
            api_key=os.environ['OPENAI_API_KEY'],
            model="text-embedding-3-small"
        )
        
        vector_store = CouchbaseVectorStore(
            cluster=cluster,
            bucket_name=os.environ['CB_BUCKET_NAME'],
            scope_name=os.environ['SCOPE_NAME'],
            collection_name=os.environ['COLLECTION_NAME'],
            embedding=embeddings,
            index_name=os.environ['INDEX_NAME'],
        )
        
        hotel_data = load_hotel_data()
        
        try:
            vector_store.add_texts(texts=hotel_data, batch_size=10)
            print("Hotel data loaded into vector store successfully")
        except Exception as e:
            print(f"Warning: Error loading hotel data: {str(e)}. Vector store created but data not loaded.")
        
        return vector_store
    except Exception as e:
        raise ValueError(f"Error setting up vector store: {str(e)}")

def main():
    try:
        setup_environment()
        
        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Hotel Search Agent")
        
        # Setup Couchbase infrastructure
        cluster = setup_couchbase_connection()
        
        setup_collection(
            cluster, 
            os.environ['CB_BUCKET_NAME'], 
            os.environ['SCOPE_NAME'], 
            os.environ['COLLECTION_NAME']
        )
        
        try:
            with open('deepseek_index.json', 'r') as file:
                index_definition = json.load(file)
            print("Loaded vector search index definition from deepseek_index.json")
        except Exception as e:
            raise ValueError(f"Error loading index definition: {str(e)}")
        
        setup_vector_search_index(cluster, index_definition)
        
        setup_vector_store(cluster)
        
        # Setup LLM with Agent Catalog callback
        llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-4o",
            temperature=0,
            callbacks=[agentc_langchain.chat.Callback(span=application_span)]
        )
        
        # Load tools from Agent Catalog and convert to LangChain tools
        tool_search = catalog.find("tool", name="search_vector_database")
        tool_details = catalog.find("tool", name="get_hotel_details")
        
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
        
        # Create a simple ReAct agent using LangChain
        react_prompt = pull("hwchase17/react")
        agent = create_react_agent(llm, tools, react_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Test the agent with sample queries
        print("\nHotel Search Agent is ready!")
        print("Testing with sample queries...")
        
        test_queries = [
            "Find me a luxury hotel with a pool",
            "I need a beach resort with spa services", 
            "Get details about Ocean Breeze Resort"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            try:
                response = agent_executor.invoke({"input": query})
                print(f"✅ Response: {response['output']}")
                print("-" * 80)
            except Exception as e:
                print(f"❌ Error: {e}")
                print("-" * 80)
                
    except Exception as e:
        print(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()