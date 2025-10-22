#!/usr/bin/env python3
"""
Setup script for Couchbase Capella infrastructure.

This script uses the couchbase-infrastructure package.
Install it with: pip install couchbase-infrastructure

Or run this script directly - it will use the package if installed.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

print("🚀 Couchbase Capella Infrastructure Setup")
print("=" * 60)

try:
    from couchbase_infrastructure import CapellaConfig, CapellaClient
    from couchbase_infrastructure.resources import (
        create_project,
        create_cluster,
        add_allowed_cidr,
        load_sample_data,
        create_database_user,
        deploy_ai_model,
        create_ai_api_key,
    )
except ImportError:
    print("\n❌ Package 'couchbase-infrastructure' is not installed.")
    print("\nInstall it with:")
    print("    pip install couchbase-infrastructure")
    print("\nOr use the CLI:")
    print("    couchbase-infra setup")
    sys.exit(1)

print("\n--- Starting Automated Capella Environment Setup ---\n")

try:
    # Load configuration
    config = CapellaConfig.from_env()
    config.validate()

    # Initialize client
    client = CapellaClient(config)
    org_id = client.get_organization_id()

    # Test API connection
    if not client.test_connection(org_id):
        print("\n❌ API connection test failed. Please check your credentials.")
        sys.exit(1)

    # 1. Get or Create Project
    print("\n[1/7] Finding or Creating Capella Project...")
    project_id = create_project(client, org_id, config.project_name)

    # 2. Create and Wait for Cluster
    print("\n[2/7] Deploying Capella Free Tier Cluster...")
    cluster_id = create_cluster(client, org_id, project_id, config.cluster_name, config)
    cluster_check_url = f"/v4/organizations/{org_id}/projects/{project_id}/clusters/{cluster_id}"
    cluster_details = client.wait_for_resource(cluster_check_url, "Cluster", None)
    cluster_conn_string = cluster_details.get("connectionString")

    # 3. Add allowed CIDR for cluster access
    print("\n[3/7] Configuring Cluster Network Access...")
    add_allowed_cidr(client, org_id, project_id, cluster_id, config.allowed_cidr)

    # 4. Load Sample Data
    print("\n[4/7] Loading 'travel-sample' Dataset...")
    load_sample_data(client, org_id, project_id, cluster_id, config.sample_bucket)

    # 5. Create Database User
    print("\n[5/7] Creating Database Credentials...")
    db_password = create_database_user(
        client, org_id, project_id, cluster_id, config.db_username, config.sample_bucket,
        recreate_if_exists=True  # Delete and recreate if exists to get fresh password
    )

    # 6. Deploy AI Models
    print("\n[6/7] Deploying AI Models...")

    # Deploy Embedding Model
    print("   Deploying embedding model...")
    embedding_model_id = deploy_ai_model(
        client,
        org_id,
        config.embedding_model_name,
        "agent-hub-embedding-model",
        "embedding",
        config,
    )
    embedding_check_url = f"/v4/organizations/{org_id}/aiServices/models/{embedding_model_id}"
    embedding_details = client.wait_for_resource(embedding_check_url, "Embedding Model", None)
    embedding_endpoint = embedding_details.get("connectionString", "")
    embedding_dimensions = embedding_details.get("model", {}).get("config", {}).get("dimensions")
    print(f"   Model dimensions: {embedding_dimensions}")

    # Deploy LLM Model
    print("   Deploying LLM model...")
    llm_model_id = deploy_ai_model(
        client,
        org_id,
        config.llm_model_name,
        "agent-hub-llm-model",
        "llm",
        config,
    )
    llm_check_url = f"/v4/organizations/{org_id}/aiServices/models/{llm_model_id}"
    llm_details = client.wait_for_resource(llm_check_url, "LLM Model", None)
    llm_endpoint = llm_details.get("connectionString", "")

    # 7. Create API Key for Models
    print("\n[7/7] Creating API Key for AI Models...")
    api_key = create_ai_api_key(client, org_id, config.ai_model_region)

    # Set Environment Variables
    print("\n✅ Configuring Environment Variables...")
    os.environ["CB_CONN_STRING"] = cluster_conn_string + "?tls_verify=none"
    os.environ["CB_USERNAME"] = config.db_username
    os.environ["CB_PASSWORD"] = db_password
    os.environ["CB_BUCKET"] = config.sample_bucket
    os.environ["CAPELLA_API_EMBEDDING_ENDPOINT"] = embedding_endpoint
    os.environ["CAPELLA_API_LLM_ENDPOINT"] = llm_endpoint
    os.environ["CAPELLA_API_EMBEDDINGS_KEY"] = api_key
    os.environ["CAPELLA_API_LLM_KEY"] = api_key
    os.environ["CAPELLA_API_EMBEDDING_MODEL"] = config.embedding_model_name
    os.environ["CAPELLA_API_LLM_MODEL"] = config.llm_model_name

    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nEnvironment Variables Set:")
    print(f"  CB_CONN_STRING: {os.environ['CB_CONN_STRING']}")
    print(f"  CB_USERNAME: {os.environ['CB_USERNAME']}")
    print(f"  CB_PASSWORD: {os.environ['CB_PASSWORD']}")
    print(f"  CB_BUCKET: {os.environ['CB_BUCKET']}")
    print(f"  CAPELLA_API_EMBEDDING_ENDPOINT: {os.environ['CAPELLA_API_EMBEDDING_ENDPOINT']}")
    print(f"  CAPELLA_API_LLM_ENDPOINT: {os.environ['CAPELLA_API_LLM_ENDPOINT']}")
    print(f"  CAPELLA_API_EMBEDDINGS_KEY: {os.environ['CAPELLA_API_EMBEDDINGS_KEY']}")
    print(f"  CAPELLA_API_LLM_KEY: {os.environ['CAPELLA_API_LLM_KEY']}")
    print(f"  CAPELLA_API_EMBEDDING_MODEL: {os.environ['CAPELLA_API_EMBEDDING_MODEL']}")
    print(f"  CAPELLA_API_LLM_MODEL: {os.environ['CAPELLA_API_LLM_MODEL']}")

except ValueError as e:
    print(f"\n❌ CONFIGURATION ERROR: {e}")
    print("\nPlease check your .env file and ensure MANAGEMENT_API_KEY is set.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ SETUP FAILED: {e}")
    if "401" in str(e) or "Unauthorized" in str(e):
        print("\n🔐 Authentication Error:")
        print("  1. Verify your API key is correct and not expired")
        print("  2. Check if your IP is in the API key allowlist")
        print("  3. Ensure the API key has Organization Admin permissions")
    sys.exit(1)
