# Converting Python Scripts to Jupyter/Colab Notebooks

This guide documents all the necessary steps, gotchas, and best practices for converting Python agent scripts (`.py` files) to Jupyter notebooks (`.ipynb` / `.json` format) suitable for Google Colab.

## Overview

The agent-catalog-quickstart project contains example agents in both Python script and Colab notebook formats:
- **Python scripts**: Local development, full control, Poetry environment
- **Colab notebooks**: Zero-setup, cloud-based, beginner-friendly

## Table of Contents

1. [Environment Variable Handling](#environment-variable-handling)
2. [Input/Prompts in Colab](#inputprompts-in-colab)
3. [Connection Strings](#connection-strings)
4. [Database User Passwords](#database-user-passwords)
5. [Root Certificate Handling](#root-certificate-handling)
6. [AI Model API Response Structure](#ai-model-api-response-structure)
7. [Environment File (.env) Writing](#environment-file-env-writing)
8. [Reference Answers for Evaluation](#reference-answers-for-evaluation)
9. [Variable Naming Consistency](#variable-naming-consistency)
10. [Import Statements](#import-statements)
11. [Framework-Specific Notes](#framework-specific-notes)
12. [Common Pitfalls](#common-pitfalls)

---

## 1. Environment Variable Handling

### Issue
Python scripts use `os.getenv()` which returns `None` if variable doesn't exist, but notebooks need safer defaults.

### Solution
Always use `.get()` with empty string defaults:

```python
# ❌ BAD (will cause KeyError if not set)
os.environ['CAPELLA_API_ENDPOINT']

# ✅ GOOD (safe with default)
os.environ.get('CAPELLA_API_ENDPOINT', '')
```

### Where This Matters
- All CAPELLA_API_* variables
- Optional configuration variables
- Any variable that might not be set during infrastructure provisioning

---

## 2. Input/Prompts in Colab

### Issue
`input()` function doesn't work well in Colab/Jupyter environments and can block execution.

### Solution
Use `getpass.getpass()` for sensitive data with try/except fallback:

```python
# ❌ BAD (blocks in Colab)
openai_api_key = input("OpenAI API Key: ")

# ✅ GOOD (works in Colab)
import getpass

try:
    openai_api_key = getpass.getpass("OpenAI API Key: ").strip()
except:
    # Fallback for environments where getpass doesn't work
    openai_api_key = ""
```

### Benefits
- Works in Google Colab
- Hides sensitive input
- Graceful fallback for non-interactive environments

---

## 3. Connection Strings

### Critical Distinction
There are TWO different connection strings with different requirements:

#### CB_CONN_STRING (Couchbase SDK Connection)
```python
# ✅ MUST include ?tls_verify=none for sandbox clusters
CB_CONN_STRING = "couchbases://cb.xyz.sandbox.nonprod-project-avengers.com?tls_verify=none"
```

#### AGENT_CATALOG_CONN_STRING (agentc CLI Connection)
```python
# ✅ MUST NOT include ?tls_verify=none
AGENT_CATALOG_CONN_STRING = "couchbases://cb.xyz.sandbox.nonprod-project-avengers.com"
```

### Implementation
```python
# Set CB_CONN_STRING with TLS parameter
cluster_conn_string = cluster_details.get("connectionString")
if not cluster_conn_string.startswith("couchbase://") and not cluster_conn_string.startswith("couchbases://"):
    cluster_conn_string = f"couchbases://{cluster_conn_string}?tls_verify=none"

os.environ["CB_CONN_STRING"] = cluster_conn_string

# Strip TLS parameters for Agent Catalog
agent_catalog_conn_string = os.environ["CB_CONN_STRING"].split("?")[0]
os.environ["AGENT_CATALOG_CONN_STRING"] = agent_catalog_conn_string
```

### Why This Matters
- **CB_CONN_STRING**: Couchbase Python SDK needs `?tls_verify=none` for non-production sandbox clusters
- **AGENT_CATALOG_CONN_STRING**: agentc CLI handles TLS differently and fails with query parameters

---

## 4. Database User Passwords

### Issue
When database user already exists, `create_database_user()` returns `"existing_user_password_not_retrievable"` instead of a usable password, causing authentication failures.

### Solution
Use `recreate_if_exists=True` parameter to force fresh password generation:

```python
# ❌ BAD (fails if user already exists)
db_password = create_database_user(
    client,
    org_id,
    project_id,
    cluster_id,
    config.db_username,
    config.sample_bucket,
)

# ✅ GOOD (always gets fresh password)
db_password = create_database_user(
    client,
    org_id,
    project_id,
    cluster_id,
    config.db_username,
    config.sample_bucket,
    recreate_if_exists=True,  # Delete and recreate if exists to get fresh password
)
```

### Why This Matters
- Notebooks are often re-run multiple times during development
- Without `recreate_if_exists=True`, subsequent runs fail with authentication errors
- This is THE most common cause of "existing_user_password_not_retrievable" errors

---

## 5. Root Certificate Handling

### Requirements
The root certificate is needed for agentc CLI to connect securely to Couchbase clusters.

### Dual Approach Needed

#### 1. Certificate Upload Section (Sets os.environ)
```python
# Handle root certificate (required for secure connections)
try:
    from google.colab import files
    uploaded = files.upload()

    if uploaded:
        cert_filename = list(uploaded.keys())[0]
        # Validate it's actually a certificate file
        if cert_filename.endswith(('.pem', '.crt', '.cer', '.txt')):
            os.environ["AGENT_CATALOG_CONN_ROOT_CERTIFICATE"] = cert_filename
        else:
            os.environ["AGENT_CATALOG_CONN_ROOT_CERTIFICATE"] = ""
except ImportError:
    # Not in Colab - ask for manual input
    cert_filename = input("Enter certificate filename (or press Enter to skip): ").strip()
    if cert_filename:
        os.environ["AGENT_CATALOG_CONN_ROOT_CERTIFICATE"] = cert_filename
    else:
        os.environ["AGENT_CATALOG_CONN_ROOT_CERTIFICATE"] = ""
```

#### 2. Write to .env File (If Certificate Exists)
```python
with open('.env', 'w') as f:
    # ... other environment variables ...

    # Write certificate if set
    cert = os.environ.get('AGENT_CATALOG_CONN_ROOT_CERTIFICATE', '').strip()
    if cert:
        f.write(f"AGENT_CATALOG_CONN_ROOT_CERTIFICATE={cert}\n")
```

### File Extension Validation
Accept common certificate formats: `.pem`, `.crt`, `.cer`, `.txt`

---

## 6. AI Model API Response Structure

### Critical Issue
The Capella AI API returns model details in a **nested structure**, not flat.

### API Response Structure
```json
{
  "model": {
    "connectionString": "https://xyz.ai.sandbox.nonprod-project-avengers.com",
    "name": "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "status": "ready"
  }
}
```

### Wrong Approach
```python
# ❌ WRONG - tries to get connectionString at top level
embedding_endpoint = embedding_details.get("connectionString", "")
# Result: Empty string → "Request URL is missing protocol" error
```

### Correct Approach
```python
# ✅ CORRECT - extract from nested 'model' object
model_info = embedding_details.get("model", {})
embedding_endpoint = model_info.get("connectionString", "")
```

### Apply to Both Models
```python
# Embedding model endpoint
embedding_check_url = f"/v4/organizations/{org_id}/aiServices/models/{embedding_model_id}"
embedding_details = client.wait_for_resource(embedding_check_url, "Embedding Model", None)
model_info = embedding_details.get("model", {})
embedding_endpoint = model_info.get("connectionString", "")

# LLM model endpoint
llm_check_url = f"/v4/organizations/{org_id}/aiServices/models/{llm_model_id}"
llm_details = client.wait_for_resource(llm_check_url, "LLM Model", None)
llm_model_info = llm_details.get("model", {})
llm_endpoint = llm_model_info.get("connectionString", "")
```

---

## 7. Environment File (.env) Writing

### Template Structure

The .env file should follow this exact structure for consistency:

```python
with open('.env', 'w') as f:
    # Couchbase-specific environment variables (for the travel-agent example tools)
    f.write(f"CB_CONN_STRING={os.environ['CB_CONN_STRING']}\n")
    f.write(f"CB_USERNAME={os.environ['CB_USERNAME']}\n")
    f.write(f"CB_PASSWORD={os.environ['CB_PASSWORD']}\n")
    f.write(f"CB_BUCKET={os.environ['CB_BUCKET']}\n")
    f.write(f"CB_SCOPE={os.environ.get('CB_SCOPE', 'agentc_data')}\n")
    f.write(f"CB_COLLECTION={os.environ.get('CB_COLLECTION', 'flight_data')}\n")  # or hotel_data, landmark_data
    f.write(f"CB_INDEX={os.environ.get('CB_INDEX', 'flight_data_index')}\n")
    f.write("\n")

    # Capella AI API variables
    f.write(f"CAPELLA_API_ENDPOINT={os.environ.get('CAPELLA_API_ENDPOINT', '')}\n")
    f.write(f"CAPELLA_API_EMBEDDING_MODEL={os.environ.get('CAPELLA_API_EMBEDDING_MODEL', '')}\n")
    f.write(f"CAPELLA_API_EMBEDDINGS_KEY={os.environ.get('CAPELLA_API_EMBEDDINGS_KEY', '')}\n")
    f.write(f"CAPELLA_API_LLM_MODEL={os.environ.get('CAPELLA_API_LLM_MODEL', '')}\n")
    f.write(f"CAPELLA_API_LLM_KEY={os.environ.get('CAPELLA_API_LLM_KEY', '')}\n")
    f.write("\n")

    # Agent Catalog Configuration
    f.write(f"AGENT_CATALOG_CONN_STRING={os.environ['AGENT_CATALOG_CONN_STRING']}\n")
    f.write(f"AGENT_CATALOG_USERNAME={os.environ['AGENT_CATALOG_USERNAME']}\n")
    f.write(f"AGENT_CATALOG_PASSWORD={os.environ['AGENT_CATALOG_PASSWORD']}\n")
    f.write(f"AGENT_CATALOG_BUCKET={os.environ['AGENT_CATALOG_BUCKET']}\n")

    # Write certificate if set
    cert = os.environ.get('AGENT_CATALOG_CONN_ROOT_CERTIFICATE', '').strip()
    if cert:
        f.write(f"AGENT_CATALOG_CONN_ROOT_CERTIFICATE={cert}\n")
```

### Variables to REMOVE (No Longer Needed)
- ❌ `CAPELLA_API_EMBEDDING_ENDPOINT` - Use CAPELLA_API_ENDPOINT instead
- ❌ `CAPELLA_API_LLM_ENDPOINT` - Use CAPELLA_API_ENDPOINT instead
- ❌ `CAPELLA_API_EMBEDDING_MAX_TOKENS` - Not used
- ❌ `NVIDIA_API_KEY` - Not used
- ❌ `TOKENIZERS_PARALLELISM` - Not needed in notebooks
- ❌ `CB_CERTIFICATE` - Use AGENT_CATALOG_CONN_ROOT_CERTIFICATE instead

---

## 8. Reference Answers for Evaluation

### Issue
Hardcoded dates, booking IDs, and timestamps in reference answers cause evaluation failures.

### Wrong Approach
```python
# ❌ BAD - hardcoded values that will never match
FLIGHT_REFERENCE_ANSWERS = [
    """Flight Booking Confirmed!

    Booking ID: FL08061563CACD  # ← Will never match dynamically generated ID
    Departure Date: 2025-08-06  # ← Will never match "tomorrow's date"
    """
]
```

### Correct Approach
Use dynamic placeholders that match the LENIENT evaluation template:

```python
# ✅ GOOD - dynamic placeholders
FLIGHT_REFERENCE_ANSWERS = [
    """Flight Booking Confirmed!

    Booking ID: [Dynamically Generated]
    Departure Date: [Tomorrow's Date - Dynamically Calculated]
    Route: LAX → JFK
    Passengers: 2
    Class: business
    Total Price: $1500.00
    """
]
```

### Placeholder Patterns
- `[Dynamically Generated]` - For auto-generated IDs
- `[Tomorrow's Date - Dynamically Calculated]` - For relative dates
- `[Current Date]` - For today's date
- `[Flight/Hotel/Landmark Booking ID]` - For typed IDs

### Why This Matters
- LENIENT evaluators use fuzzy matching
- Placeholders tell evaluator to ignore specific values
- Exact values cause 0% correctness scores

---

## 9. Variable Naming Consistency

### Issue
Inconsistent variable names across notebooks cause confusion and errors.

### Standard Names Across All Notebooks

```python
# ✅ Use these consistently
ARIZE_AVAILABLE = True  # Not PHOENIX_AVAILABLE
OPENAI_AVAILABLE = True
```

### Why This Matters
- flight_search uses `ARIZE_AVAILABLE`
- landmark_search originally used `PHOENIX_AVAILABLE` (wrong)
- hotel_search uses `ARIZE_AVAILABLE` (correct)
- Consistency prevents NameError exceptions

---

## 10. Import Statements

### Infrastructure Resource Imports

```python
from couchbase_infrastructure.resources import (
    create_project,
    create_developer_pro_cluster,  # ✅ NOT create_cluster
    add_allowed_cidr,
    load_sample_data,
    create_database_user,
    deploy_ai_model,
    create_ai_api_key,
)
```

### Common Mistake
```python
# ❌ WRONG - old function name
from couchbase_infrastructure.resources import create_cluster

# ✅ CORRECT - updated function name
from couchbase_infrastructure.resources import create_developer_pro_cluster
```

---

## 11. Framework-Specific Notes

### LangGraph (flight_search_agent)
- Uses `StateGraph` for agent workflow
- State management via typed dictionaries
- Handles tool calling and routing

### LangChain (hotel_search_agent)
- Uses `AgentExecutor` with tools
- Simpler setup than LangGraph
- Direct tool integration

### LlamaIndex (landmark_search_agent)
- Uses `ReActAgent` for reasoning
- Query engine integration
- Different tool wrapper format

### Common Across All Frameworks
- Environment variable setup (identical)
- Infrastructure provisioning (identical)
- Evaluation setup (similar patterns)

---

## 12. Common Pitfalls

### 1. Empty CAPELLA_API_ENDPOINT
**Symptom**: `httpcore.UnsupportedProtocol: Request URL is missing protocol`
**Cause**: Not extracting endpoint from nested `model` object
**Fix**: See [AI Model API Response Structure](#6-ai-model-api-response-structure)

### 2. Authentication Errors
**Symptom**: `AuthenticationException`, "existing_user_password_not_retrievable"
**Cause**: Missing `recreate_if_exists=True` parameter
**Fix**: See [Database User Passwords](#4-database-user-passwords)

### 3. TLS Verification Errors
**Symptom**: SSL/TLS connection errors
**Cause**: Wrong connection string format
**Fix**: See [Connection Strings](#3-connection-strings)

### 4. Input Blocking
**Symptom**: Notebook hangs at input prompt
**Cause**: Using `input()` instead of `getpass.getpass()`
**Fix**: See [Input/Prompts in Colab](#2-inputprompts-in-colab)

### 5. Low Evaluation Scores
**Symptom**: 0% QA correctness despite correct responses
**Cause**: Hardcoded values in reference answers
**Fix**: See [Reference Answers for Evaluation](#8-reference-answers-for-evaluation)

### 6. NameError for ARIZE_AVAILABLE
**Symptom**: `NameError: name 'ARIZE_AVAILABLE' is not defined`
**Cause**: Using `PHOENIX_AVAILABLE` or variable defined after use
**Fix**: See [Variable Naming Consistency](#9-variable-naming-consistency)

### 7. ImportError for create_cluster
**Symptom**: `cannot import name 'create_cluster'`
**Cause**: Using old function name
**Fix**: See [Import Statements](#10-import-statements)

---

## Conversion Checklist

When converting a Python script to a Colab notebook, ensure:

- [ ] Environment variables use `.get()` with defaults
- [ ] Input prompts use `getpass.getpass()` with try/except
- [ ] CB_CONN_STRING includes `?tls_verify=none`
- [ ] AGENT_CATALOG_CONN_STRING excludes `?tls_verify=none`
- [ ] create_database_user includes `recreate_if_exists=True`
- [ ] Certificate upload section present
- [ ] Certificate written to .env only if set
- [ ] AI endpoints extracted from nested `model` object
- [ ] .env file follows standard template structure
- [ ] Reference answers use dynamic placeholders
- [ ] Variable names match standards (ARIZE_AVAILABLE, etc.)
- [ ] Imports use `create_developer_pro_cluster`
- [ ] CB_SCOPE, CB_COLLECTION, CB_INDEX in .env
- [ ] No unnecessary variables in .env
- [ ] Removed all duplicate code cells
- [ ] Queries defined once (not duplicated)

---

## Testing Your Conversion

After conversion, test these scenarios:

1. **Fresh Run**: Run all cells from scratch
2. **Re-Run**: Run all cells again (tests password recreation)
3. **Environment Variables**: Check `.env` file contents
4. **Agent Queries**: Test all example queries
5. **Evaluation**: Verify evaluation scores >90%
6. **Certificate Upload**: Test both Colab upload and manual input paths

---

## Additional Resources

- [Couchbase Infrastructure Documentation](../couchbase_infrastructure/)
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

---

## AI Agent Prompt for Future Conversions

When using an AI agent to perform conversions, provide this prompt:

```
Convert the Python script {script_name}.py to a Colab notebook following these requirements:

1. Environment Variables:
   - Use os.environ.get() with empty string defaults for all optional variables
   - Handle CB_CONN_STRING with "couchbases://...?tls_verify=none"
   - Handle AGENT_CATALOG_CONN_STRING without TLS parameters (strip with .split("?")[0])

2. User Input:
   - Replace all input() calls with getpass.getpass()
   - Include try/except fallback for non-Colab environments

3. Database User:
   - Add recreate_if_exists=True to create_database_user() call

4. Root Certificate:
   - Include certificate upload section with file validation (.pem, .crt, .cer, .txt)
   - Write to .env only if certificate is set
   - Use Google Colab files.upload() with fallback to input()

5. AI Model Endpoints:
   - Extract connectionString from nested model object:
     model_info = response.get("model", {})
     endpoint = model_info.get("connectionString", "")

6. Environment File:
   - Include: CB_CONN_STRING, CB_USERNAME, CB_PASSWORD, CB_BUCKET, CB_SCOPE, CB_COLLECTION, CB_INDEX
   - Include: CAPELLA_API_ENDPOINT, CAPELLA_API_EMBEDDING_MODEL, CAPELLA_API_EMBEDDINGS_KEY, CAPELLA_API_LLM_MODEL, CAPELLA_API_LLM_KEY
   - Include: AGENT_CATALOG_CONN_STRING, AGENT_CATALOG_USERNAME, AGENT_CATALOG_PASSWORD, AGENT_CATALOG_BUCKET
   - Include: AGENT_CATALOG_CONN_ROOT_CERTIFICATE (only if set)
   - Exclude: CAPELLA_API_EMBEDDING_ENDPOINT, CAPELLA_API_LLM_ENDPOINT, NVIDIA_API_KEY, TOKENIZERS_PARALLELISM, etc.

7. Reference Answers:
   - Replace hardcoded dates with "[Tomorrow's Date - Dynamically Calculated]"
   - Replace hardcoded IDs with "[Dynamically Generated]"
   - Use placeholders that match LENIENT evaluation templates

8. Variable Names:
   - Use ARIZE_AVAILABLE (not PHOENIX_AVAILABLE)
   - Ensure consistency with other notebooks

9. Imports:
   - Import create_developer_pro_cluster (not create_cluster)
   - Include all required infrastructure imports

10. Validation:
    - Remove duplicate cells
    - Remove duplicate query definitions
    - Ensure variables defined before use
    - Test: fresh run, re-run, evaluation

Follow the exact patterns shown in docs/PYTHON_TO_NOTEBOOK_CONVERSION.md
```

---

## Maintenance Notes

This document should be updated when:
- New environment variables are added
- Infrastructure API changes
- New agent frameworks are added
- New evaluation patterns emerge
- Common errors are discovered

Last updated: 2025-10-24
