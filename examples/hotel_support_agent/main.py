import os
import getpass
import logging
import time
from datetime import timedelta

from dotenv import load_dotenv
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions
from couchbase.management.buckets import CreateBucketSettings
from couchbase.management.search import SearchIndex
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from agentc.catalog import Catalog
from agentc_langgraph.graph import Callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('httpx').setLevel(logging.CRITICAL)

def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")

def setup_environment():
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY', 'CB_HOST', 'CB_USERNAME', 'CB_PASSWORD', 'CB_BUCKET_NAME']
    for var in required_vars:
        _set_if_undefined(var)
    
    defaults = {
        'CB_HOST': 'couchbase://localhost',
        'CB_USERNAME': 'Administrator', 
        'CB_PASSWORD': 'password',
        'CB_BUCKET_NAME': 'hotel-search',
        'INDEX_NAME': 'hotel_vector_search',
        'SCOPE_NAME': 'shared',
        'COLLECTION_NAME': 'hotels'
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
        logging.info("Successfully connected to Couchbase")
        return cluster
    except Exception as e:
        raise ConnectionError(f"Failed to connect to Couchbase: {str(e)}")

def setup_collection(cluster, bucket_name, scope_name, collection_name):
    try:
        try:
            bucket = cluster.bucket(bucket_name)
            logging.info(f"Bucket '{bucket_name}' exists")
        except Exception:
            logging.info(f"Creating bucket '{bucket_name}'...")
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
            logging.info(f"Bucket '{bucket_name}' created successfully")

        bucket_manager = bucket.collections()
        
        scopes = bucket_manager.get_all_scopes()
        scope_exists = any(scope.name == scope_name for scope in scopes)
        
        if not scope_exists and scope_name != "_default":
            logging.info(f"Creating scope '{scope_name}'...")
            bucket_manager.create_scope(scope_name)
            logging.info(f"Scope '{scope_name}' created successfully")

        collections = bucket_manager.get_all_scopes()
        collection_exists = any(
            scope.name == scope_name and collection_name in [col.name for col in scope.collections]
            for scope in collections
        )

        if not collection_exists:
            logging.info(f"Creating collection '{collection_name}'...")
            bucket_manager.create_collection(scope_name, collection_name)
            logging.info(f"Collection '{collection_name}' created successfully")

        collection = bucket.scope(scope_name).collection(collection_name)
        time.sleep(3)

        try:
            cluster.query(f"CREATE PRIMARY INDEX IF NOT EXISTS ON `{bucket_name}`.`{scope_name}`.`{collection_name}`").execute()
            logging.info("Primary index created successfully")
        except Exception as e:
            logging.warning(f"Error creating primary index: {str(e)}")

        return collection
    except Exception as e:
        raise RuntimeError(f"Error setting up collection: {str(e)}")

def setup_vector_search_index(cluster, index_definition):
    try:
        scope_index_manager = cluster.bucket(os.environ['CB_BUCKET_NAME']).scope(os.environ['SCOPE_NAME']).search_indexes()
        
        existing_indexes = scope_index_manager.get_all_indexes()
        index_name = index_definition["name"]

        if index_name not in [index.name for index in existing_indexes]:
            logging.info(f"Creating vector search index '{index_name}'...")
            search_index = SearchIndex.from_json(index_definition)
            scope_index_manager.upsert_index(search_index)
            logging.info(f"Vector search index '{index_name}' created successfully")
        else:
            logging.info(f"Vector search index '{index_name}' already exists")
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
        
        vector_store = CouchbaseSearchVectorStore(
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
            logging.info("Hotel data loaded into vector store successfully")
        except Exception as e:
            logging.warning(f"Error loading hotel data: {str(e)}. Vector store created but data not loaded.")
        
        return vector_store
    except Exception as e:
        raise ValueError(f"Error setting up vector store: {str(e)}")

def main():
    try:
        setup_environment()
        
        catalog = Catalog()
        application_span = catalog.Span(name="Hotel Search Agent")
        
        cluster = setup_couchbase_connection()
        
        setup_collection(
            cluster, 
            os.environ['CB_BUCKET_NAME'], 
            os.environ['SCOPE_NAME'], 
            os.environ['COLLECTION_NAME']
        )
        
        index_definition = {
            "name": os.environ['INDEX_NAME'],
            "type": "fulltext-index",
            "params": {
                "doc_config": {
                    "docid_prefix_delim": "",
                    "docid_regexp": "",
                    "mode": "scope.collection.type_field",
                    "type_field": "type"
                },
                "mapping": {
                    "default_analyzer": "standard",
                    "default_datetime_parser": "dateTimeOptional",
                    "default_field": {
                        "dynamic": True,
                        "enabled": False
                    },
                    "default_mapping": {
                        "dynamic": True,
                        "enabled": False
                    },
                    "default_type": "_default",
                    "docvalues_dynamic": False,
                    "index_dynamic": True,
                    "store_dynamic": False,
                    "type_field": "_type",
                    "types": {
                        f"{os.environ['SCOPE_NAME']}.{os.environ['COLLECTION_NAME']}": {
                            "dynamic": False,
                            "enabled": True,
                            "properties": {
                                "embedding": {
                                    "dynamic": False,
                                    "enabled": True,
                                    "fields": [
                                        {
                                            "dims": 1536,
                                            "index": True,
                                            "name": "embedding",
                                            "similarity": "dot_product",
                                            "type": "vector",
                                            "vector_index_optimized_for": "recall"
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },
                "store": {
                    "indexType": "scorch",
                    "segmentVersion": 16
                }
            },
            "sourceType": "gocbcore",
            "sourceName": os.environ['CB_BUCKET_NAME'],
            "sourceParams": {},
            "planParams": {
                "maxPartitionsPerPIndex": 1024,
                "indexPartitions": 1,
                "numReplicas": 0
            }
        }
        
        setup_vector_search_index(cluster, index_definition)
        
        setup_vector_store(cluster)
        
        llm = ChatOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            model="gpt-4o",
            temperature=0,
            callbacks=[Callback(span=application_span)]
        )
        
        tool_result_search = catalog.find("tool", name="search_vector_database")
        tool_result_details = catalog.find("tool", name="get_hotel_details")
        tools = [tool_result_search.func, tool_result_details.func]
        
        system_message = SystemMessage(content="""You are a professional hotel search assistant. 
        Help users find the perfect hotel based on their requirements including location, budget, amenities, and preferences.
        Use the available tools to search for hotels and provide detailed information.
        Always be helpful, accurate, and professional in your responses.""")
        
        agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)
        
        print("Hotel Search Agent is ready! Type 'exit' to quit.")
        
        while (user_input := input("\n>> ")) != "exit":
            if not user_input.strip():
                continue
                
            try:
                events = agent_executor.stream(
                    {"messages": [("user", user_input)]},
                )
                
                for event in events:
                    if "event" not in event:
                        output = event.get("agent", {}).get("messages", [])
                        if len(output):
                            print(output[-1].content)
                        continue
                    
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = event["data"]["chunk"].content
                        if content:
                            print(content, end="", flush=True)
                    elif kind == "on_tool_end":
                        print(f"\nTool output: {event['data']['output']}")
                        
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()