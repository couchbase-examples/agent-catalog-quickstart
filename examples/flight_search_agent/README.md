# Flight Search Agent

A professional flight search agent built with Agent Catalog and LangGraph, demonstrating seamless integration of SQL++, semantic search, and Python tools for comprehensive flight booking assistance.

## 🎯 What This Tutorial Demonstrates

- **Agent Catalog**: Tool and prompt management with automatic loading
- **LangGraph**: Conversation orchestration using ReActAgent patterns
- **Couchbase**: Real flight data from travel-sample database
- **Mixed Tool Formats**: SQL++, YAML, and Python tools working together
- **Professional Architecture**: Clean separation of concerns and proper patterns

## 🏗️ Architecture Overview

```
customer_support_agent_advanced/
├── customer_support_agent.py    # Main application (minimal, just invokes graph)
├── graph.py                     # LangGraph workflow definition
├── node.py                      # ReActAgent implementation
├── test_tool_output.py          # Comprehensive tool testing
├── prompts/
│   └── customer_support_assistant.yaml  # Agent prompt with tool specifications
├── tools/
│   ├── lookup_flight_info.sqlpp         # SQL++ database queries
│   ├── search_policies.yaml             # YAML semantic search
│   ├── search_knowledge_base.sqlpp      # SQL++ knowledge search
│   └── update_customer_context.py       # Python functions
└── .env                         # Environment configuration
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Ensure Couchbase is running locally with travel-sample data
# Install dependencies (from project root)
pip install -r requirements.txt
```

### 2. Environment Setup

Configure your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o"  # or gpt-4o-mini for cost savings

# Agent Catalog Configuration
AGENT_CATALOG_CONN_STRING=couchbase://127.0.0.1
AGENT_CATALOG_USERNAME=your-username
AGENT_CATALOG_PASSWORD=your-password
AGENT_CATALOG_BUCKET=travel-sample

# Couchbase Configuration (for tools)
CB_CONN_STRING=couchbase://127.0.0.1
CB_USERNAME=your-username
CB_PASSWORD=your-password
CB_CERTIFICATE=""

# Performance Optimization
TOKENIZERS_PARALLELISM=false
```

### 3. Initialize and Run

```bash
# Initialize Agent Catalog (one-time setup)
agentc init

# Index tools and prompts
agentc index .

# Publish to make available at runtime
agentc publish

# Run the tutorial
python customer_support_agent.py

# Test individual tools
python test_tool_output.py
```

## 🔧 Agent Catalog Commands Explained

### `agentc init`
**Purpose**: Creates the Agent Catalog infrastructure in Couchbase
**When to use**: First time setting up Agent Catalog in a new environment
**What it does**:
- Creates collections for tools, prompts, and metadata
- Sets up vector indexes for semantic search
- Creates analytics collections for usage tracking
- Establishes audit trails and logging infrastructure

```bash
agentc init
```

### `agentc index .`
**Purpose**: Scans local files and builds the catalog index
**When to use**: After adding/modifying tools or prompts
**What it does**:
- Scans directories for `.sqlpp`, `.yaml`, `.py` files
- Validates tool and prompt syntax
- Generates embeddings for semantic search
- Creates local catalog metadata

```bash
agentc index .
```

### `agentc publish`
**Purpose**: Uploads indexed catalog to Couchbase for runtime access
**When to use**: After indexing and when ready to deploy changes
**Requirements**: Clean git repository (all changes committed)
**What it does**:
- Uploads tools and prompts to Couchbase collections
- Associates catalog version with git commit
- Makes tools available for Agent runtime loading
- Enables version tracking and rollback

```bash
git add -A && git commit -m "Updated tools"
agentc index .
agentc publish
```

### `agentc clean`
**Purpose**: Removes local catalog cache and temporary files
**When to use**: When troubleshooting or starting fresh
**What it does**:
- Clears `.agent-catalog/` directory
- Removes cached embeddings and metadata
- Forces full re-indexing on next `agentc index`

```bash
agentc clean
```

### `agentc status`
**Purpose**: Shows current catalog state and Git status
**When to use**: To check what needs to be indexed/published

```bash
agentc status
```

## 🛠️ Customizing Tools

### SQL++ Tools (Database Queries)

Create `.sqlpp` files in the `tools/` directory:

```sql
-- tools/my_custom_query.sqlpp
/*
name: my_custom_query
description: Description of what this query does
input:
  type: object
  properties:
    param1:
      type: string
      description: "Parameter description"
  required: ["param1"]
secrets:
  - couchbase:
      conn_string: CB_CONN_STRING
      username: CB_USERNAME
      password: CB_PASSWORD
*/

SELECT field1, field2 
FROM `bucket`.scope.collection 
WHERE condition = $param1
LIMIT 10;
```

**Key Points**:
- Metadata in C-style comments `/* */`
- Use `$parameter` for named parameters
- Define input schema for validation
- Specify required secrets for database access

### YAML Tools (Semantic Search)

Create `.yaml` files for vector search capabilities:

```yaml
# tools/my_semantic_search.yaml
record_kind: semantic_search
name: my_semantic_search
description: Semantic search description

input:
  type: object
  properties:
    query:
      type: string
      description: "Search query text"
  required: ["query"]

secrets:
  - couchbase:
      conn_string: CB_CONN_STRING
      username: CB_USERNAME
      password: CB_PASSWORD

vector_search:
  bucket: travel-sample
  scope: inventory
  collection: airline
  index: airline_vector_index
  vector_field: name_embedding
  text_field: name
  
  embedding_model:
    name: sentence-transformers/all-MiniLM-L12-v2
    
  num_candidates: 5
```

**Key Points**:
- `record_kind: semantic_search` identifies YAML tools
- Define vector search configuration
- Specify embedding model for semantic matching
- Include secrets for database access

### Python Tools (Custom Functions)

Create `.py` files with decorated functions:

```python
# tools/my_python_tool.py
import agentc
from pydantic import BaseModel
from typing import Dict, Any

class MyInput(BaseModel):
    parameter: str
    optional_param: int = 10

@agentc.catalog.tool
def my_python_function(input: MyInput) -> Dict[str, Any]:
    """
    Description of what this function does.
    This will be used by the LLM to understand when to call this tool.
    """
    result = {
        "processed": input.parameter,
        "multiplied": input.optional_param * 2,
        "timestamp": "2024-01-01"
    }
    return result
```

**Key Points**:
- Use `@agentc.catalog.tool` decorator
- Define Pydantic models for input validation
- Include comprehensive docstrings
- Return structured data

## 📝 Customizing Prompts

Create `.yaml` files in the `prompts/` directory:

```yaml
# prompts/my_agent_prompt.yaml
record_kind: prompt
name: my_agent_prompt
description: Description of this prompt's purpose

# Output schema for structured responses
output:
  title: MyResponse
  description: Response format
  type: object
  properties:
    response:
      type: string
      description: "Agent's response text"
    confidence:
      type: number
      description: "Confidence score 0-1"
  required: ["response"]

# Specify which tools this prompt can use
tools:
  - name: "lookup_flight_info"
  - name: "search_policies"
  - query: "semantic search for knowledge"
    limit: 3

# Annotations for categorization
annotations:
  domain: "customer_support"
  framework: "langgraph"

# Main prompt content
content:
  agent_instructions: >
    You are a helpful assistant. Your role is to...
    
    Use the available tools to provide accurate information.
    When customers ask about flights, use lookup_flight_info.
    For policies, use search_policies.
    
  response_guidelines:
    - Be professional and helpful
    - Use tools to get accurate data
    - Provide comprehensive information
    
  output_format_instructions: >
    Structure your responses clearly with relevant details.
```

**Key Points**:
- `record_kind: prompt` identifies prompt files
- Define output schema for structured responses
- Specify tools by name or semantic query
- Include comprehensive instructions for the agent

## 🧪 Testing and Evaluation

### Individual Tool Testing

Use `test_tool_output.py` to test tools in isolation:

```bash
python test_tool_output.py
```

This will:
- Test each tool with sample inputs
- Show actual data returned from Couchbase
- Validate tool functionality across multiple scenarios
- Demonstrate all tool formats working together

### Integration Testing

Run the full customer support scenario:

```bash
python customer_support_agent.py
```

This demonstrates:
- Multiple conversation scenarios
- Different airport routes and real flight data
- Policy searches with semantic matching
- End-to-end Agent Catalog + LangGraph integration

### Creating Custom Evaluations

Create evaluation scripts in the `evals/` directory:

```python
# evals/my_evaluation.py
import agentc
from my_test_cases import test_scenarios

def evaluate_agent():
    catalog = agentc.Catalog()
    
    for scenario in test_scenarios:
        # Run agent with test scenario
        result = run_agent_scenario(scenario)
        
        # Evaluate result quality
        score = evaluate_response_quality(result)
        
        # Log results
        print(f"Scenario: {scenario['name']}, Score: {score}")

if __name__ == "__main__":
    evaluate_agent()
```

## 🔄 Development Workflow

### 1. Adding New Tools

```bash
# 1. Create tool file
vim tools/my_new_tool.sqlpp

# 2. Test tool independently
python test_tool_output.py

# 3. Update prompts to reference new tool
vim prompts/customer_support_assistant.yaml

# 4. Index and test
agentc index .
python customer_support_agent.py

# 5. Commit and publish
git add -A
git commit -m "Add new tool for X functionality"
agentc index .
agentc publish
```

### 2. Updating Prompts

```bash
# 1. Modify prompt
vim prompts/customer_support_assistant.yaml

# 2. Test changes
agentc index .
python customer_support_agent.py

# 3. Publish
git commit -am "Improved prompt instructions"
agentc index .
agentc publish
```

### 3. Debugging Issues

```bash
# Check catalog status
agentc status

# View available tools
agentc find --type tool

# View available prompts  
agentc find --type prompt

# Clean and rebuild if needed
agentc clean
agentc index .
```

## 🎛️ Configuration Options

### Model Configuration

Adjust the AI model used:

```bash
# In .env file
OPENAI_MODEL="gpt-4o"          # Best performance
OPENAI_MODEL="gpt-4o-mini"     # Cost-effective
OPENAI_MODEL="gpt-3.5-turbo"   # Budget option
```

### Tool Behavior

Modify tool limits in SQL++ files:

```sql
-- Change LIMIT to return more results
SELECT * FROM routes 
WHERE condition = $param
LIMIT 20;  -- Increased from 10
```

### Search Configuration

Adjust semantic search in YAML files:

```yaml
vector_search:
  num_candidates: 10  # More results
  similarity_threshold: 0.7  # Stricter matching
```

## 📊 Monitoring and Analytics

Agent Catalog automatically tracks:
- Tool usage frequency
- Response times
- Error rates
- Conversation flows

View analytics in Couchbase Analytics or create custom queries:

```sql
-- View tool usage
SELECT tool_name, COUNT(*) as usage_count
FROM `travel-sample`._default.agent_catalog_logs
WHERE event_type = 'tool_call'
GROUP BY tool_name;
```

## 🔧 Troubleshooting

### Common Issues

**"Tool not found"**
- Check tool is properly indexed: `agentc status`
- Verify tool name matches exactly in prompt
- Re-index: `agentc index .`

**"Vector embedding errors"**
- Ensure sentence-transformers model is installed
- Check TOKENIZERS_PARALLELISM=false in .env
- Clean and re-index: `agentc clean && agentc index .`

**"Cannot publish dirty catalog"**
- Commit all changes: `git add -A && git commit -m "Changes"`
- Ensure git status is clean
- Then: `agentc index . && agentc publish`

**"Couchbase connection failed"**
- Verify Couchbase is running: `curl http://localhost:8091`
- Check credentials in .env file
- Ensure travel-sample bucket is loaded

### Debug Mode

Enable verbose logging:

```bash
export AGENTC_DEBUG=true
python customer_support_agent.py
```

## 🚀 Production Deployment

### Environment Setup

```bash
# Production .env
OPENAI_MODEL="gpt-4o"
AGENT_CATALOG_CONN_STRING=couchbase://prod-cluster
AGENT_CATALOG_USERNAME=prod-user
AGENT_CATALOG_PASSWORD=secure-password
```

### CI/CD Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy Agent Catalog
on:
  push:
    branches: [main]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Setup Python
      uses: actions/setup-python@v2
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Index catalog
      run: agentc index .
    - name: Publish catalog
      run: agentc publish
```

## 📚 Additional Resources

- **Agent Catalog Documentation**: [docs.couchbase.com/agent-catalog](https://docs.couchbase.com/agent-catalog)
- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **Couchbase Developer Portal**: [developer.couchbase.com](https://developer.couchbase.com)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-tool`
3. Add your tool/prompt following the patterns above
4. Test thoroughly: `python test_tool_output.py`
5. Commit changes: `git commit -m "Add feature"`
6. Create pull request

## 📄 License

This tutorial is part of the Agent Catalog quickstart examples.