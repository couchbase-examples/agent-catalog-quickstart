# Agent Catalog Quickstart

This repository provides a quickstart guide for using the Agent Catalog with Capella Model Services and Couchbase.

## Prerequisites

- Python 3.8+
- Poetry ([Installation Guide](https://python-poetry.org/docs/#installation))
- pip (Python package installer)
- Git (for repository management)
- An OpenAI API Key (or other LLM provider)
- Couchbase Capella account (or local Couchbase installation)

## Quick Setup (Automated)

The fastest way to get started is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/couchbaselabs/agent-catalog-quickstart.git
cd agent-catalog-quickstart

# Make the setup script executable
chmod +x scripts/setup.sh

# Run the automated setup script
bash scripts/setup.sh
```

This script will:
- Install all Agent Catalog libraries and dependencies
- Run `poetry install` at the root and in all example directories
- Install required LlamaIndex packages
- Create template environment files for all examples
- Set up git for clean repository state
- Verify the installation
- Provide next steps

## Manual Setup (Step-by-Step)

If you prefer to install manually or need to troubleshoot:

### 1. Install Agent Catalog Libraries

Install all libraries in the correct dependency order:

```bash
# Install core library first
pip install -e agent-catalog/libs/agentc_core

# Install integrations
pip install -e agent-catalog/libs/agentc_integrations/langchain \
            -e agent-catalog/libs/agentc_integrations/langgraph \
            -e agent-catalog/libs/agentc_integrations/llamaindex

# Install CLI and testing
pip install -e agent-catalog/libs/agentc_cli \
            -e agent-catalog/libs/agentc_testing

# Install main package
pip install -e agent-catalog/libs/agentc

# Install LlamaIndex dependencies
pip install llama-index llama-index-vector-stores-couchbase
```

### 2. Install Poetry Dependencies

**Important:** You must run `poetry install` in multiple locations:

```bash
# Install root dependencies
poetry install

# Install dependencies for each example agent
cd notebooks/route_planner_agent && poetry install && cd ../..
cd notebooks/flight_search_agent && poetry install && cd ../..
cd notebooks/hotel_support_agent && poetry install && cd ../..
```

### 3. Verify Installation

```bash
agentc --help
```

### 4. Environment Configuration

Create a `.env` file in each example directory with the following configuration:

#### For Couchbase Capella (Cloud):
```bash
# OpenAI API Configuration
OPENAI_API_KEY="your-openai-api-key"

# Couchbase Configuration
CB_CONN_STRING="couchbases://your-cluster.cloud.couchbase.com"
CB_USERNAME="your-username"
CB_PASSWORD="your-password"
CB_BUCKET="vector-search-testing"
CB_SCOPE="agentc_data"
CB_COLLECTION="route_data"
CB_INDEX="route_data_index"

# Capella API Configuration
CAPELLA_API_ENDPOINT="https://your-endpoint.ai.cloud.couchbase.com"
CAPELLA_API_EMBEDDING_MODEL="intfloat/e5-mistral-7b-instruct"
CAPELLA_API_LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Agent Catalog Configuration
AGENT_CATALOG_CONN_STRING="couchbase://127.0.0.1"
AGENT_CATALOG_BUCKET="vector-search-testing"
AGENT_CATALOG_USERNAME="your-username"
AGENT_CATALOG_PASSWORD="your-password"
AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""

# Environment variable to prevent tokenizer warnings
TOKENIZERS_PARALLELISM=false
```

#### For Local Couchbase:
```bash
# OpenAI API Configuration
OPENAI_API_KEY="your-openai-api-key"

# Couchbase Configuration
CB_CONN_STRING="couchbase://127.0.0.1"
CB_USERNAME="Administrator"
CB_PASSWORD="password"
CB_BUCKET="default"
CB_SCOPE="_default"
CB_COLLECTION="_default"
CB_INDEX="vector_index"

# Agent Catalog Configuration
AGENT_CATALOG_CONN_STRING="couchbase://127.0.0.1"
AGENT_CATALOG_BUCKET="default"
AGENT_CATALOG_USERNAME="Administrator"
AGENT_CATALOG_PASSWORD="password"
AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""

# Environment variable to prevent tokenizer warnings
TOKENIZERS_PARALLELISM=false
```

**Important:** Each example directory needs its own `.env` file:
- `notebooks/route_planner_agent/.env`
- `notebooks/flight_search_agent/.env`
- `notebooks/hotel_support_agent/.env`

## Usage

### Initialize Agent Catalog

Navigate to any example directory and initialize:

```bash
cd notebooks/route_planner_agent
agentc init
```

### Index Your Agent

```bash
agentc index .
```

### Publish Your Agent

**Important:** The git repository must be clean before publishing:

```bash
# Commit any changes first
git add .
git commit -m "Your commit message"

# Then publish
agentc publish
```

### Run Example Agents

```bash
# Run the route planner agent
python main.py

# Run with specific queries
python main.py "Find a route from San Francisco to Los Angeles"
```

## Available Examples

This quickstart includes several example agents:

- **Route Planner Agent** (`notebooks/route_planner_agent/`) - Plans routes between locations
- **Flight Search Agent** (`notebooks/flight_search_agent/`) - Searches and books flights
- **Hotel Support Agent** (`notebooks/hotel_support_agent/`) - Hotel search and support
- **Customer Support Agent** (`notebooks/customer_support_agent/`) - General customer support

Each example includes:
- Complete source code
- Configuration files
- Test cases
- Documentation
- **Own poetry dependencies** (requires `poetry install` in each directory)

## Agent Catalog CLI Commands

| Command | Description |
|---------|-------------|
| `agentc init` | Initialize agent catalog in current directory |
| `agentc index .` | Index the current agent directory |
| `agentc publish` | Publish agent to catalog (requires clean git status) |
| `agentc --help` | Show all available commands |
| `agentc env` | Show environment configuration |

## Troubleshooting

### Common Issues

1. **"command not found: agentc"**
   - Run the setup script: `bash scripts/setup.sh`
   - Or install manually following the manual setup steps

2. **"No module named 'llama_index.vector_stores'"**
   - Install LlamaIndex: `pip install llama-index llama-index-vector-stores-couchbase`
   - Run `poetry install` in the example directory

3. **"Could not find the environment variable $AGENT_CATALOG_CONN_STRING"**
   - Ensure each example directory has its own `.env` file
   - Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in the `.env` file

4. **"Cannot publish a dirty catalog to the DB"**
   - Commit all changes: `git add . && git commit -m "Your message"`
   - Ensure `git status` shows a clean repository before `agentc publish`

5. **Connection errors to Couchbase**
   - Verify your `.env` configuration in each example directory
   - Check that your Couchbase cluster is accessible
   - Ensure proper credentials and connection strings

6. **"Certificate error" when connecting**
   - For local installations, use `couchbase://127.0.0.1`
   - For Capella, ensure you're using the correct `couchbases://` connection string
   - Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in your `.env`

7. **Poetry dependency issues**
   - Run `poetry install` in the root directory
   - Run `poetry install` in each example directory separately
   - Each example has its own `pyproject.toml` and requires separate installation

8. **Tokenizer parallelism warnings**
   - Add `TOKENIZERS_PARALLELISM=false` to your `.env` files

### Setup Requirements Checklist

- [ ] Poetry installed
- [ ] Agent Catalog libraries installed (`pip install -e ...`)
- [ ] LlamaIndex installed (`pip install llama-index llama-index-vector-stores-couchbase`)
- [ ] Root poetry dependencies installed (`poetry install` in root)
- [ ] Example poetry dependencies installed (`poetry install` in each example directory)
- [ ] `.env` files created in each example directory
- [ ] Environment variables configured with actual credentials
- [ ] Git repository in clean state (for publishing)

### Getting Help

- Check the `docs/` directory for detailed guides
- Look at example implementations in `notebooks/`
- Review error messages for specific configuration issues
- Ensure you've run `poetry install` in all required directories

## Development

### Adding New Agents

1. Create a new directory under `notebooks/`
2. Add your agent code, prompts, and tools
3. Create appropriate configuration files (`pyproject.toml`, `.env`)
4. Run `poetry install` in the new directory
5. Run `agentc init` and `agentc index .`

### Running Tests

```bash
cd notebooks/route_planner_agent
python -m pytest tests/
```

### Evaluation

Run evaluations with Arize:

```bash
python run_evaluations.py
```

## Architecture

Each example agent follows this structure:
```
notebooks/agent_name/
├── main.py              # Main agent implementation
├── pyproject.toml       # Poetry dependencies (requires poetry install)
├── .env                 # Environment configuration
├── prompts/             # Agent prompts and templates
├── tools/               # Agent tools and functions
├── data/                # Data loading and processing
├── tests/               # Test cases
└── evals/               # Evaluation scripts
```

## Contributing

This is a quickstart repository. For contributing to the main Agent Catalog:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all poetry dependencies are installed
5. Commit changes (required for publishing)
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

