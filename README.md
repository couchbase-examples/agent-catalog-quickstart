# Agent Catalog Quickstart

This repository provides a quickstart guide for using the Agent Catalog with Capella Model Services and Couchbase.

## Prerequisites

- Python 3.12+
- Poetry ([Installation Guide](https://python-poetry.org/docs/#installation))
- Git (for repository management)
- An OpenAI API Key (or other LLM provider)
- Couchbase Capella account (or local Couchbase installation)

## 🚀 Quick Setup (Recommended)

**The simplest way to get started** with any individual agent:

```bash
# Choose an agent project
cd notebooks/hotel_search_agent_langchain
# OR cd notebooks/flight_search_agent_langraph  
# OR cd notebooks/landmark_search_agent_llamaindex

# Install all dependencies with a single command
poetry install --no-root

# Set up your environment (see Environment Configuration below)
cp .env.sample .env
# Edit .env with your credentials

# Run the agent
poetry run python main.py
```

That's it! 🎯 Poetry automatically handles all Agent Catalog dependencies and integrations.

## Alternative: Full Setup (All Agents + Global CLI)

If you want to set up **all agents at once** and install the global `agentc` CLI:

```bash
# Clone the repository
git clone https://github.com/couchbaselabs/agent-catalog-quickstart.git
cd agent-catalog-quickstart

# Run the automated setup script
bash scripts/setup.sh
```

This comprehensive script will:
- Install all Agent Catalog libraries globally
- Set up all three agent environments
- Install the global `agentc` CLI command
- Verify the installation

## Manual Setup (Step-by-Step)

### Option 1: Individual Agent Setup (Recommended)

For working with a single agent, this is the simplest approach:

```bash
# Navigate to any agent directory
cd notebooks/hotel_search_agent_langchain

# Install all dependencies automatically
poetry install --no-root

# Set up environment
cp .env.sample .env
# Edit .env with your credentials

# Test the installation
poetry run python -c "import agentc; print('✅ Agent Catalog ready!')"
```

### Option 2: Global Installation (For CLI Usage)

If you need the global `agentc` CLI or want to work with multiple agents:

```bash
# Install Agent Catalog libraries in dependency order
pip install -e agent-catalog/libs/agentc_core
pip install -e agent-catalog/libs/agentc_cli  
pip install -e agent-catalog/libs/agentc
pip install -e agent-catalog/libs/agentc_integrations/langchain
pip install -e agent-catalog/libs/agentc_integrations/langgraph
pip install -e agent-catalog/libs/agentc_integrations/llamaindex

# Install each agent's dependencies
cd notebooks/flight_search_agent_langraph && poetry install --no-root && cd ../..
cd notebooks/hotel_search_agent_langchain && poetry install --no-root && cd ../..
cd notebooks/landmark_search_agent_llamaindex && poetry install --no-root && cd ../..

# Verify global CLI installation
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
CB_COLLECTION="hotel_data"
CB_INDEX="hotel_data_index"

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
- `notebooks/flight_search_agent_langraph/.env`
- `notebooks/hotel_search_agent_langchain/.env`
- `notebooks/landmark_search_agent_llamaindex/.env`

## Usage

### Running Agents (Simple Approach)

If you used the recommended individual setup:

```bash
cd notebooks/hotel_search_agent_langchain

# Run the agent
poetry run python main.py

# Run with specific queries  
poetry run python main.py "Find hotels in Paris with free breakfast"

# Run evaluations
poetry run python evals/eval_arize.py
```

### Using Global CLI (After Full Setup)

If you installed the global CLI:

```bash
cd notebooks/hotel_search_agent_langchain

# Initialize Agent Catalog
agentc init

# Index your agent
agentc index .

# Publish your agent (requires clean git)
git add . && git commit -m "Your changes"
agentc publish

# Run the agent
python main.py "Find hotels in Paris with free breakfast"
```

## Available Examples

This quickstart includes three self-contained example agents:

### 🛩️ Flight Search Agent (`notebooks/flight_search_agent_langraph/`)
- **Framework**: LangGraph
- **Setup**: `poetry install --no-root`
- **Run**: `poetry run python main.py`

### 🏨 Hotel Search Agent (`notebooks/hotel_search_agent_langchain/`) 
- **Framework**: LangChain
- **Setup**: `poetry install --no-root`
- **Run**: `poetry run python main.py`

### 🗺️ Landmark Search Agent (`notebooks/landmark_search_agent_llamaindex/`)
- **Framework**: LlamaIndex  
- **Setup**: `poetry install --no-root`
- **Run**: `poetry run python main.py`

Each example is **completely independent** and includes:
- Complete source code with all dependencies
- Configuration files (`pyproject.toml`, `.env.sample`)
- Test cases and evaluation scripts
- Documentation and prompts
- **One-command setup** with Poetry

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

1. **"No module named 'agentc'"**
   - **Solution**: Run `poetry install --no-root` in the agent directory
   - Poetry automatically handles all Agent Catalog dependencies

2. **"command not found: agentc"** (when using CLI)
   - **Solution**: Install globally with `bash scripts/setup.sh`
   - Or follow the global installation steps in Manual Setup

3. **"No module named 'llama_index.vector_stores'"**
   - **Solution**: Run `poetry install --no-root` (should auto-install)
   - If still failing, check the `pyproject.toml` in that agent directory

4. **"Could not find the environment variable $AGENT_CATALOG_CONN_STRING"**
   - **Solution**: Copy `.env.sample` to `.env` and edit with your credentials
   - Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in the `.env` file

5. **Poetry installation fails**
   - **Solution**: Delete `poetry.lock` and run `poetry install --no-root` again
   - Check Poetry version: `poetry --version` (should be 1.5+)

6. **Connection errors to Couchbase**
   - **Solution**: Verify your `.env` configuration 
   - Check that your Couchbase cluster is accessible
   - Ensure proper credentials and connection strings

7. **"Certificate error" when connecting**
   - **Solution**: For local use `couchbase://127.0.0.1`
   - For Capella use `couchbases://` connection string
   - Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in your `.env`

8. **Tokenizer parallelism warnings**
   - **Solution**: Add `TOKENIZERS_PARALLELISM=false` to your `.env` files

### Setup Requirements Checklist

#### For Individual Agent (Recommended)
- [ ] Poetry installed (`poetry --version`)
- [ ] Agent dependencies installed (`poetry install --no-root`)
- [ ] `.env` file created (`cp .env.sample .env`)
- [ ] Environment variables configured with actual credentials

#### For Global CLI Usage (Optional)
- [ ] Agent Catalog libraries installed globally (`bash scripts/setup.sh`)
- [ ] Global `agentc` CLI available (`agentc --help`)
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
cd notebooks/hotel_search_agent_langchain
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

