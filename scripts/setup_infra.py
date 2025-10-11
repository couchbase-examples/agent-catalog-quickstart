import os
from dotenv import load_dotenv

# Load environment variables with override
load_dotenv(override=True)

# --- Credentials from Environment ---
MANAGEMENT_API_KEY = os.getenv("MANAGEMENT_API_KEY")
ORGANIZATION_ID = os.getenv("ORGANIZATION_ID")

# --- Configuration for this Tutorial ---
PROJECT_NAME = os.getenv("PROJECT_NAME", "Agent-Hub-Project")
CLUSTER_NAME = os.getenv("CLUSTER_NAME", "agent-hub-flight-cluster")
DB_USERNAME = os.getenv("DB_USERNAME", "agent_app_user")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Snowflake/snowflake-arctic-embed-l-v2.0")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

# Validate required environment variables
if not MANAGEMENT_API_KEY:
    raise ValueError("Missing required environment variable: MANAGEMENT_API_KEY")

# Allow auto-detection of organization ID if not provided
if not ORGANIZATION_ID:
    print("No ORGANIZATION_ID provided, will auto-detect from first organization...")


import httpx
import time

# Correct sandbox API URL - use cloudapi subdomain
API_BASE_URL = "https://cloudapi.sbx-30.sandbox.nonprod-project-avengers.com"
HEADERS = {
    "Authorization": f"Bearer {MANAGEMENT_API_KEY}",
    "Content-Type": "application/json"
}

# --- API Helper Functions ---

def get_current_ip():
    """Get the current public IP address."""
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get("https://api.ipify.org")
        if response.status_code == 200:
            return response.text.strip()
    except Exception:
        pass
    return "Unable to determine IP"

def get_organization_id():
    """Get organization ID - either from env var or auto-detect first organization."""
    if ORGANIZATION_ID:
        return ORGANIZATION_ID

    try:
        with httpx.Client(headers=HEADERS, timeout=10) as client:
            response = client.get(f"{API_BASE_URL}/v4/organizations")
        if response.status_code == 200:
            orgs = response.json().get("data", [])
            if orgs:
                auto_org_id = orgs[0]["id"]
                print(f"   Auto-detected Organization ID: {auto_org_id}")
                return auto_org_id
        raise Exception(f"Failed to get organizations. Status: {response.status_code}")
    except Exception as e:
        raise Exception(f"Failed to auto-detect organization ID: {e}")

def test_api_connection(org_id):
    """Test API connection and provide debugging info."""
    print("🔍 Testing API connection...")
    print(f"   Current IP: {get_current_ip()}")
    print(f"   API Base URL: {API_BASE_URL}")
    print(f"   Organization ID: {org_id}")

    # Test basic API connectivity
    try:
        with httpx.Client(headers=HEADERS, timeout=10) as client:
            response = client.get(f"{API_BASE_URL}/v4/organizations/{org_id}")
        print(f"   API Response Status: {response.status_code}")
        if response.status_code == 401:
            print("   ❌ Authentication failed - check API key and IP allowlist")
        elif response.status_code == 200:
            print("   ✅ Authentication successful")
        else:
            print(f"   ⚠️  Unexpected response: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")

def wait_for_resource_ready(check_url: str, resource_type: str, timeout_seconds: int = 900):
    """Polls a Capella endpoint until the resource is in a 'healthy' or 'ready' state."""
    start_time = time.time()
    print(f"   Waiting for {resource_type} to become ready... (this can take several minutes)")
    while time.time() - start_time < timeout_seconds:
        try:
            with httpx.Client(headers=HEADERS, timeout=30) as client:
                response = client.get(f"{API_BASE_URL}{check_url}")
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", {}).get("state", data.get("currentState", "")).lower()
                if status in ["healthy", "ready", "deployed", "running"]:
                    print(f"✅ {resource_type} is ready!")
                    return data
                print(f"   Current status: {status}")
            time.sleep(20)
        except Exception as e:
            print(f"   ... still waiting (error polling: {e})")
            time.sleep(20)
    raise Exception(f"Timeout: {resource_type} was not ready within {timeout_seconds} seconds.")

def get_or_create_project(org_id, project_name):
    """Finds a project by name or creates it if it doesn't exist."""
    list_endpoint = f"/v4/organizations/{org_id}/projects"
    
    print(f"   Searching for project named '{project_name}'...")
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        list_response = client.get(f"{API_BASE_URL}{list_endpoint}")
    
    if list_response.status_code == 200:
        for project in list_response.json().get('data', []):
            if project.get('name') == project_name:
                project_id = project.get('id')
                print(f"✅ Found existing project. Project ID: {project_id}")
                return project_id
    
    print(f"   Project not found. Creating a new project named '{project_name}'...")
    create_endpoint = f"/v4/organizations/{org_id}/projects"
    payload = {"name": project_name, "description": "Project for Agent Application Hub samples."}
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        create_response = client.post(f"{API_BASE_URL}{create_endpoint}", json=payload)
    
    if create_response.status_code == 201:
        project_id = create_response.json().get("id")
        print(f"✅ Successfully created new project. Project ID: {project_id}")
        return project_id
    else:
        raise Exception(f"Failed to create project. Status: {create_response.status_code}, Response: {create_response.text}")

def create_free_tier_cluster(org_id, proj_id, name):
    """Creates a new free tier cluster using the Management API."""
    endpoint = f"/v4/organizations/{org_id}/projects/{proj_id}/clusters/freeTier"
    payload = {
        "name": name,
        "cloudProvider": {
            "type": "aws",
            "region": "us-east-2",
            "cidr": "10.1.30.0/23"
        }
    }
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        response = client.post(f"{API_BASE_URL}{endpoint}", json=payload)
    if response.status_code == 202:
        cluster_id = response.json().get("id")
        print(f"   Cluster creation job submitted. Cluster ID: {cluster_id}")
        return cluster_id
    elif response.status_code == 422 and "limited to provisioning one cluster" in response.text:
        print("   A free tier cluster already exists. Attempting to find it...")
        clusters_endpoint = f"/v4/organizations/{org_id}/projects/{proj_id}/clusters"
        with httpx.Client(headers=HEADERS, timeout=30) as client:
            list_response = client.get(f"{API_BASE_URL}{clusters_endpoint}")
        if list_response.status_code == 200:
            for cluster in list_response.json().get('data', []):
                 if cluster.get('name') == name:
                    print(f"   Found existing cluster with name '{name}'. Using it.")
                    return cluster.get('id')
        raise Exception(f"Failed to create or find free tier cluster. Response: {response.text}")
    else:
        raise Exception(f"Failed to create cluster. Status: {response.status_code}, Response: {response.text}")

def load_travel_sample(org_id, proj_id, cluster_id):
    """Loads the travel-sample bucket into the specified cluster."""
    endpoint = f"/v4/organizations/{org_id}/projects/{proj_id}/clusters/{cluster_id}/sampleBuckets"
    payload = {
        "name": "travel-sample"
    }
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        response = client.post(f"{API_BASE_URL}{endpoint}", json=payload)
    if response.status_code in [201, 422]:
        print(f"✅ `travel-sample` bucket load command accepted.")
        bucket_check_url = f"/v4/organizations/{org_id}/projects/{proj_id}/clusters/{cluster_id}/buckets"
        start_time = time.time()
        while time.time() - start_time < 300:
             with httpx.Client(headers=HEADERS, timeout=30) as client:
                bucket_list_response = client.get(f"{API_BASE_URL}{bucket_check_url}")
             if any(b.get('name') == 'travel-sample' for b in bucket_list_response.json().get('data',[])):
                 print("✅ `travel-sample` bucket is ready.")
                 return
             time.sleep(10)
        raise Exception("Timeout waiting for travel-sample bucket to become available.")
    else:
        raise Exception(f"Failed to load travel-sample. Status: {response.status_code}, Response: {response.text}")

def create_db_user(org_id, proj_id, cluster_id, username):
    """Creates a database user with broad access for the tutorial."""
    endpoint = f"/v4/organizations/{org_id}/projects/{proj_id}/clusters/{cluster_id}/users"

    # First, check if user already exists
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        list_response = client.get(f"{API_BASE_URL}{endpoint}")

    if list_response.status_code == 200:
        existing_users = list_response.json().get('data', [])
        for user in existing_users:
            if user.get('name') == username:
                print(f"   Database user '{username}' already exists. Skipping creation.")
                # Return a placeholder password since we can't retrieve the existing one
                return "existing_user_password_not_retrievable"

    # Create new user if doesn't exist
    payload = {
        "name": username,
        "access": [{
            "privileges": ["data_reader", "data_writer"],
            "resources": {
                "buckets": [{
                    "name": "travel-sample",
                    "scopes": [{
                        "name": "*"
                    }]
                }]
            }
        }]
    }
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        response = client.post(f"{API_BASE_URL}{endpoint}", json=payload)
    if response.status_code == 201:
        data = response.json()
        print(f"   Database user '{username}' created successfully.")
        return data['password']
    else:
        raise Exception(f"Failed to create DB user. Status: {response.status_code}, Response: {response.text}")

def create_ai_model(org_id, model_name, deployment_name, model_type="embedding"):
    """Deploys a new AI model using the AI Services API v4."""
    endpoint = f"/v4/organizations/{org_id}/aiServices/models"

    # Build the payload according to v4 API spec
    payload = {
        "name": deployment_name,
        "catalogModelName": model_name,
        "cloudConfig": {
            "provider": "aws",
            "region": "us-east-1"
        }
    }
    
    # Add model-type specific configurations
    if model_type == "llm":
        payload["quantization"] = "fullPrecision"
        payload["optimization"] = "throughput"
    
    print(f"   Creating {model_type} model '{deployment_name}' with catalog model '{model_name}'...")
    
    with httpx.Client(headers=HEADERS, timeout=60) as client:
        response = client.post(f"{API_BASE_URL}{endpoint}", json=payload)
    
    if response.status_code == 202:
        model_id = response.json().get("id")
        print(f"   {model_type.title()} model '{deployment_name}' deployment job submitted. Model ID: {model_id}")
        return model_id
    elif response.status_code == 422:
        # Model may already exist
        error_text = response.text.lower()
        if "already exists" in error_text or "duplicate" in error_text:
            print(f"   Model '{deployment_name}' may already exist. Attempting to find it...")
            # List existing models
            with httpx.Client(headers=HEADERS, timeout=30) as client:
                list_response = client.get(f"{API_BASE_URL}{endpoint}")
            if list_response.status_code == 200:
                models = list_response.json().get('data', [])
                for model in models:
                    if model.get('name') == deployment_name:
                        model_id = model.get('id')
                        print(f"   Found existing model. Model ID: {model_id}")
                        return model_id
        raise Exception(f"Failed to create {model_type} model. Status: {response.status_code}, Response: {response.text}")
    else:
        raise Exception(f"Failed to create {model_type} model '{deployment_name}'. Status: {response.status_code}, Response: {response.text}")
    
    
print("--- 🚀 Starting Automated Capella Environment Setup ---")

try:
    # Get organization ID (from env var or auto-detect)
    organization_id = get_organization_id()

    # Test API connection first
    test_api_connection(organization_id)

    # 1. Get or Create Project
    print("\n[1/6] Finding or Creating Capella Project...")
    project_id = get_or_create_project(organization_id, PROJECT_NAME)

    # 2. Create and Wait for Cluster
    print("\n[2/6] Deploying Capella Free Tier Cluster...")
    cluster_id = create_free_tier_cluster(organization_id, project_id, CLUSTER_NAME)
    cluster_check_url = f"/v4/organizations/{organization_id}/projects/{project_id}/clusters/{cluster_id}"
    cluster_details = wait_for_resource_ready(cluster_check_url, "Cluster")
    cluster_conn_string = cluster_details.get("connectionString")

    # 3. Load Sample Data
    print("\n[3/6] Loading 'travel-sample' Dataset...")
    load_travel_sample(organization_id, project_id, cluster_id)

    # 4. Create Database User
    print("\n[4/6] Creating Database Credentials...")
    db_password = create_db_user(organization_id, project_id, cluster_id, DB_USERNAME)

    # 5. Deploy AI Models
    print("\n[5/6] Deploying AI Models...")

    # Deploy Embedding Model
    print("   Deploying embedding model...")
    embedding_model_id = create_ai_model(organization_id, EMBEDDING_MODEL_NAME, "agent-hub-embedding-model", "embedding")
    embedding_check_url = f"/v4/organizations/{organization_id}/aiServices/models/{embedding_model_id}"
    embedding_details = wait_for_resource_ready(embedding_check_url, "Embedding Model")
    embedding_endpoint = embedding_details.get("connectionString", "")

    # Deploy LLM Model
    print("   Deploying LLM model...")
    llm_model_id = create_ai_model(organization_id, LLM_MODEL_NAME, "agent-hub-llm-model", "llm")
    llm_check_url = f"/v4/organizations/{organization_id}/aiServices/models/{llm_model_id}"
    llm_details = wait_for_resource_ready(llm_check_url, "LLM Model")
    llm_endpoint = llm_details.get("connectionString", "")
    
    # 6. Set Environment Variables for the Notebook
    print("\n[6/6] Configuring Environment for this Notebook Session...")
    os.environ["CB_CONN_STRING"] = cluster_conn_string + "?tls_verify=none"
    os.environ["CB_USERNAME"] = DB_USERNAME
    os.environ["CB_PASSWORD"] = db_password
    os.environ["CB_BUCKET"] = "travel-sample"

    # Set AI model endpoints and credentials
    os.environ["CAPELLA_API_EMBEDDING_ENDPOINT"] = embedding_endpoint
    os.environ["CAPELLA_API_LLM_ENDPOINT"] = llm_endpoint
    os.environ["CAPELLA_API_EMBEDDINGS_KEY"] = MANAGEMENT_API_KEY
    os.environ["CAPELLA_API_LLM_KEY"] = MANAGEMENT_API_KEY
    os.environ["CAPELLA_API_EMBEDDING_MODEL"] = EMBEDDING_MODEL_NAME
    os.environ["CAPELLA_API_LLM_MODEL"] = LLM_MODEL_NAME
    
    print("\n--- ✅ SETUP COMPLETE! ---")
    print("All resources have been deployed and configured.")
    print("You can now proceed to run the rest of the cells in the notebook.")

    print("\n--- Environment Variables Set ---")
    print(f"CB_CONN_STRING: {os.environ['CB_CONN_STRING']}")
    print(f"CB_USERNAME: {os.environ['CB_USERNAME']}")
    print(f"CB_PASSWORD: {os.environ['CB_PASSWORD']}")
    print(f"CB_BUCKET: {os.environ['CB_BUCKET']}")
    print(f"CAPELLA_API_EMBEDDING_ENDPOINT: {os.environ['CAPELLA_API_EMBEDDING_ENDPOINT']}")
    print(f"CAPELLA_API_LLM_ENDPOINT: {os.environ['CAPELLA_API_LLM_ENDPOINT']}")
    print(f"CAPELLA_API_EMBEDDINGS_KEY: {os.environ['CAPELLA_API_EMBEDDINGS_KEY']}")
    print(f"CAPELLA_API_LLM_KEY: {os.environ['CAPELLA_API_LLM_KEY']}")
    print(f"CAPELLA_API_EMBEDDING_MODEL: {os.environ['CAPELLA_API_EMBEDDING_MODEL']}")
    print(f"CAPELLA_API_LLM_MODEL: {os.environ['CAPELLA_API_LLM_MODEL']}")

except Exception as e:
    print("\n--- ❌ SETUP FAILED ---")
    print(f"An error occurred during the automated setup: {e}")

    # Enhanced error handling for authentication issues
    if "401" in str(e) or "Unauthorized" in str(e):
        print("\n🔐 Authentication Error Detected:")
        print("1. Verify your API key is correct and not expired")
        print("2. Check if your current IP address is in the API key allowlist")
        print("3. Ensure the API key has sufficient permissions (Organization Admin role)")
        print("\n💡 To get your current IP address, run: curl -s https://api.ipify.org")
        print("   Then add it to your API key allowlist in the Couchbase Capella console.")

    print("\nPlease check your credentials and permissions, then try running this cell again.")