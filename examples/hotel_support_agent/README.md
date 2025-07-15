# Hotel Support Agent: Agent Catalog + LangGraph + Couchbase Tutorial

This comprehensive tutorial demonstrates the integration of **Agent Catalog**, **LangGraph**, and **Couchbase** to build a sophisticated hotel search and recommendation system with vector search capabilities and real-time hotel data.

## <� What This Tutorial Demonstrates

- **Agent Catalog**: Tool and prompt management with automatic loading
- **LangChain**: Conversation orchestration using ReActAgent patterns  
- **Couchbase Vector Search**: Semantic search with real hotel data from travel-sample
- **Travel-sample Integration**: Uses actual hotel data (917 documents) from travel-sample.inventory.hotel
- **Simplified Architecture**: Single vector search tool with real-world data
- **Professional Architecture**: Clean separation of concerns and proper patterns

## <� New Workflow

This refactored version uses:

1. **travel-sample bucket**: No longer creates custom buckets - uses the standard travel-sample bucket
2. **agentc_data scope**: Creates a new scope within travel-sample for agent data
3. **hotel_data collection**: Stores embeddings of real hotel data from travel-sample.inventory.hotel
4. **Single tool**: Only `search_vector_database` - removed the details tool for simplicity
5. **Real data**: 917 actual hotel documents with embeddings for semantic search

## <� Architecture Overview

```
hotel_support_agent/
   main.py                          # Main application with agent setup
   agentcatalog_index.json          # Vector search index configuration
   pyproject.toml                   # Dependencies and project config
   prompts/
      hotel_search_assistant.yaml  # Agent prompt for hotel search
   tools/
      search_vector_database.py    # Semantic hotel search (single tool)
   data/
      hotel_data.py                # Real hotel data from travel-sample
   evals/                           # Evaluation scripts
   .env                             # Environment configuration
```

## = Quick Start

### 1. Prerequisites

```bash
# Ensure Couchbase is running locally or in Capella
# Install dependencies (from project root)
poetry install
```

### 2. Environment Setup

Configure your `.env` file:

```bash
# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key"

# Couchbase Configuration (using travel-sample bucket)
CB_CONN_STRING="couchbase://localhost"  # or your Capella connection string
CB_USERNAME="Administrator"             # or your Capella username
CB_PASSWORD="password"                  # or your Capella password
CB_BUCKET="travel-sample"               # using travel-sample bucket
CB_SCOPE="agentc_data"                  # new scope for agent data
CB_COLLECTION="hotel_data"              # collection for hotel embeddings
CB_INDEX="hotel_data_index"             # vector search index

# Performance Optimization
TOKENIZERS_PARALLELISM=false
```

### 3. Initialize and Run

```bash
# Navigate to the hotel support agent directory
cd examples/hotel_support_agent

# Initialize Agent Catalog (one-time setup)
agentc init

# Index tools and prompts
agentc index .

# Publish to make available at runtime
agentc publish

# Run the hotel support agent (interactive mode)
python main.py

# Run test suite with sample queries
python main.py test

# Test data loading from travel-sample independently
python main.py test-data
```

## =' Agent Catalog Commands Explained

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
- Scans directories for `.py` files with @tool decorators
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

```bash
agentc clean
```

### `agentc status`
**Purpose**: Shows current catalog state and Git status
**When to use**: To check what needs to be indexed/published

```bash
agentc status
```

## <� Hotel Search Features

### Vector Semantic Search
- **Embedding Model**: OpenAI text-embedding-3-small (1536 dimensions)
- **Search Method**: Dot product similarity with recall optimization
- **Capabilities**: Natural language hotel queries with semantic matching

### Hotel Database
- **Sample Hotels**: 5 comprehensive hotel examples
- **Data Fields**: Name, location, description, amenities, pricing, ratings
- **Search Types**: Location-based, amenity-based, price-range, rating

### Interactive Chat Interface
- Professional hotel recommendations
- Detailed hotel information on request
- Natural language query processing
- Error handling and user guidance

## =� Customizing Tools

### Python Tools (Hotel Functions)

The hotel support agent uses Python-based tools with the Agent Catalog decorator:

```python
# tools/search_vector_database.py
import os
from agentc.core import tool
from langchain_couchbase.vectorstores import CouchbaseSearchVectorStore

@tool
def search_vector_database(query: str) -> str:
    """Searches the Couchbase vector database for hotels matching the user's query.
    
    Args:
        query: The search query describing hotel preferences, location, amenities, etc.
        
    Returns:
        A formatted string containing relevant hotel search results with details.
    """
    # Implementation using Couchbase vector search
    vector_store = CouchbaseSearchVectorStore(...)
    search_results = vector_store.similarity_search_with_score(query, k=5)
    return formatted_results
```

**Key Points**:
- Use `@tool` decorator from `agentc.core`
- Include comprehensive docstrings for LLM understanding
- Return structured, formatted data
- Handle errors gracefully without fallbacks

### Tool Configuration

Both tools connect to Couchbase with consistent configuration:

```python
# Couchbase connection setup
auth = PasswordAuthenticator(
    os.environ.get('CB_USERNAME', 'Administrator'), 
    os.environ.get('CB_PASSWORD', 'password')
)
cluster = Cluster(os.environ.get('CB_HOST', 'couchbase://localhost'), options)

# Vector store configuration  
vector_store = CouchbaseSearchVectorStore(
    cluster=cluster,
    bucket_name=os.environ.get('CB_BUCKET_NAME', 'vector-search-testing'),
    scope_name=os.environ.get('SCOPE_NAME', 'shared'),
    collection_name=os.environ.get('COLLECTION_NAME', 'deepseek'),
    embedding=embeddings,
    index_name=os.environ.get('INDEX_NAME', 'vector_search_deepseek'),
)
```

## =� Customizing Prompts

Create professional prompts using the Agent Catalog decorator:

```python
# prompts/prompts.py
from agentc.core import prompt

@prompt(
    name="hotel_search_system_prompt",
    template="""You are a professional hotel search assistant. Your role is to help users find the perfect hotel accommodation based on their specific needs and preferences.

Key responsibilities:
- Understand user requirements including location, budget, dates, amenities, and special needs
- Use the search_vector_database tool to find relevant hotels based on semantic similarity
- Use the get_hotel_details tool to provide comprehensive information about specific hotels
- Provide accurate, helpful, and professional recommendations

Guidelines:
- Always be professional, courteous, and helpful
- Provide detailed explanations of hotel amenities and features
- Include pricing information when available
- Focus on matching user needs with appropriate hotel options"""
)
def hotel_search_system_prompt() -> str:
    pass
```

## =� Vector Search Index Configuration

The application uses the existing `deepseek_index.json` configuration:

```json
{
  "type": "fulltext-index",
  "name": "vector_search_deepseek",
  "sourceName": "vector-search-testing",
  "params": {
    "mapping": {
      "types": {
        "shared.deepseek": {
          "properties": {
            "embedding": {
              "fields": [{
                "dims": 1536,
                "similarity": "dot_product",
                "type": "vector",
                "vector_index_optimized_for": "recall"
              }]
            }
          }
        }
      }
    }
  }
}
```

**Key Features**:
- **Vector Dimensions**: 1536 (matches OpenAI text-embedding-3-small)
- **Similarity Metric**: Dot product for fast, accurate matching
- **Optimization**: Recall-optimized for comprehensive results
- **Schema**: Proper mapping for embedding and text fields

## >� Testing and Development

### Running the Application

```bash
# Start the interactive hotel search agent
python main.py

# Example queries to try:
# "Find luxury hotels in New York with a spa"
# "I need a business hotel in Chicago with meeting rooms"
# "Show me beachfront resorts in Miami"
# "Get details for Grand Palace Hotel"
```

### Testing Individual Tools

```python
# Test vector search
from tools.search_vector_database import search_vector_database
result = search_vector_database("luxury hotel with pool")
print(result)

# Test hotel details
from tools.get_hotel_details import get_hotel_details
details = get_hotel_details("Grand Palace Hotel")
print(details)
```

## = Development Workflow

### 1. Adding New Hotels

```bash
# 1. Update hotel data in main.py load_hotel_data() function
vim main.py

# 2. Test the application
python main.py

# 3. Index and publish changes
agentc index .
agentc publish
```

### 2. Updating Search Logic

```bash
# 1. Modify search tools
vim tools/search_vector_database.py

# 2. Test changes
python main.py

# 3. Commit and publish
git add -A
git commit -m "Improved search algorithm"
agentc index .
agentc publish
```

### 3. Debugging Issues

```bash
# Check catalog status
agentc status

# View available tools
agentc find --type tool

# Clean and rebuild if needed
agentc clean
agentc index .
```

## <� Configuration Options

### Model Configuration

Adjust the AI model used:

```python
# In main.py
llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    model="gpt-4o",         # Best performance
    # model="gpt-4o-mini",  # Cost-effective
    temperature=0,
)
```

### Search Configuration

Modify search behavior:

```python
# In search_vector_database.py
search_results = vector_store.similarity_search_with_score(query, k=5)  # Number of results
```

### Hotel Data

Customize hotel information in `main.py`:

```python
def load_hotel_data():
    hotels_data = [
        {
            "name": "Your Hotel Name",
            "location": "City, State",
            "description": "Hotel description...",
            "price_range": "$200-$400",
            "amenities": ["WiFi", "Pool", "Gym"],
            "rating": 4.5
        }
        # Add more hotels...
    ]
```

## =' Troubleshooting

### Common Issues

**"Tool not found"**
- Check tool is properly indexed: `agentc status`
- Verify tool name matches exactly in code
- Re-index: `agentc index .`

**"Vector search errors"**
- Ensure OpenAI API key is set correctly
- Check Couchbase connection and index exists
- Verify vector dimensions match (1536)

**"Cannot publish dirty catalog"**
- Commit all changes: `git add -A && git commit -m "Changes"`
- Ensure git status is clean
- Then: `agentc index . && agentc publish`

**"Couchbase connection failed"**
- Verify Couchbase is running and accessible
- Check credentials in environment variables
- Ensure bucket and collections exist

### Debug Mode

Enable verbose logging:

```bash
export AGENTC_DEBUG=true
python main.py
```

## =� Production Deployment

### Environment Setup

```bash
# Production environment variables
OPENAI_API_KEY="your-production-api-key"
CB_HOST="couchbase://prod-cluster"
CB_USERNAME="prod-user"  
CB_PASSWORD="secure-password"
```

### Capella Configuration

For Couchbase Capella deployment:

```bash
CB_HOST="couchbases://cb.your-endpoint.cloud.couchbase.com"
CB_USERNAME="your-capella-username"
CB_PASSWORD="your-capella-password"
CB_BUCKET_NAME="your-production-bucket"
```

## =� Additional Resources

- **Agent Catalog Documentation**: [docs.couchbase.com/agent-catalog](https://docs.couchbase.com/agent-catalog)
- **LangGraph Documentation**: [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **Couchbase Vector Search**: [docs.couchbase.com/server/current/fts/fts-vector-search.html](https://docs.couchbase.com/server/current/fts/fts-vector-search.html)
- **OpenAI Embeddings**: [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)

## > Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-hotel-feature`
3. Add your enhancement following the patterns above
4. Test thoroughly with the hotel search agent
5. Commit changes: `git commit -m "Add feature"`
6. Create pull request

## =� License

This tutorial is part of the Agent Catalog quickstart examples.