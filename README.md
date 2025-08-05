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
# Install Agent Catalog packages
pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex

# Install Arize Phoenix and evaluation dependencies
pip3 install "arize-phoenix[evals]" arize arize-otel openinference-instrumentation-langchain openinference-instrumentation-openai openinference-instrumentation-llama-index

# Fix OpenTelemetry version conflicts (if needed)
pip3 install --upgrade opentelemetry-instrumentation-asgi opentelemetry-instrumentation-fastapi opentelemetry-util-http

# Install root dependencies
poetry install

# Choose an agent project and install its dependencies
cd notebooks/hotel_search_agent_langchain
# OR cd notebooks/flight_search_agent_langraph  
# OR cd notebooks/landmark_search_agent_llamaindex

# Install notebook-specific dependencies
poetry install --no-root

# Set up your environment (see Environment Configuration below)
cp .env.sample .env
# Edit .env with your credentials

# Run the agent
poetry run python main.py
```

That's it! 🎯 The `pip3 install` handles all Agent Catalog packages, and Poetry manages the project-specific dependencies.

## Alternative: Automated Setup Script (Fallback)

If you encounter issues with the pip installation or prefer an automated approach:

```bash
# Clone the repository
git clone https://github.com/couchbaselabs/agent-catalog-quickstart.git
cd agent-catalog-quickstart

# Run the automated setup script
bash scripts/setup.sh
```

This fallback script will:
- Install all Agent Catalog libraries from local source
- Set up all three agent environments
- Install the global `agentc` CLI command
- Verify the installation

## Manual Setup (Step-by-Step)

### Option 1: Simple Installation (Recommended)

The easiest approach using published packages:

```bash
# Install Agent Catalog packages from PyPI
pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex

# Install Arize Phoenix and evaluation dependencies
pip3 install "arize-phoenix[evals]" arize arize-otel openinference-instrumentation-langchain openinference-instrumentation-openai openinference-instrumentation-llama-index

# Fix OpenTelemetry version conflicts (if needed)
pip3 install --upgrade opentelemetry-instrumentation-asgi opentelemetry-instrumentation-fastapi opentelemetry-util-http

# Install root dependencies
poetry install

# Navigate to any agent directory
cd notebooks/hotel_search_agent_langchain

# Install notebook-specific dependencies
poetry install --no-root

# Set up environment
cp .env.sample .env
# Edit .env with your credentials

# Test the installation
poetry run python -c "import agentc; print('✅ Agent Catalog ready!')"
```

### Option 2: Local Development Installation

If you need to modify the Agent Catalog source code:

```bash
# Install Agent Catalog libraries from local source
pip install -e agent-catalog/libs/agentc_core
pip install -e agent-catalog/libs/agentc_cli  
pip install -e agent-catalog/libs/agentc
pip install -e agent-catalog/libs/agentc_integrations/langchain
pip install -e agent-catalog/libs/agentc_integrations/langgraph
pip install -e agent-catalog/libs/agentc_integrations/llamaindex

# Install root dependencies
poetry install

# Install each agent's dependencies
cd notebooks/flight_search_agent_langraph && poetry install --no-root && cd ../..
cd notebooks/hotel_search_agent_langchain && poetry install --no-root && cd ../..
cd notebooks/landmark_search_agent_llamaindex && poetry install --no-root && cd ../..

# Verify installation
agentc --help
```

## Environment Configuration

Each agent needs its own `.env` file with your credentials:

```bash
# Copy the sample file and edit it
cp .env.sample .env
# Edit .env with your actual credentials
```

**Required files:**
- `notebooks/flight_search_agent_langraph/.env`
- `notebooks/hotel_search_agent_langchain/.env`
- `notebooks/landmark_search_agent_llamaindex/.env`

For complete environment configuration examples (Capella vs Local), see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md#environment-configuration-examples)**.

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

Having issues? Check our comprehensive troubleshooting guide: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Quick Fixes

- **"No module named 'agentc'"**: Run `pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex`
- **"No module named 'phoenix'" or evaluation errors**: Run `pip3 install "arize-phoenix[evals]" arize arize-otel openinference-instrumentation-langchain openinference-instrumentation-openai openinference-instrumentation-llama-index`
- **OpenTelemetry version conflicts**: Run `pip3 install --upgrade opentelemetry-instrumentation-asgi opentelemetry-instrumentation-fastapi opentelemetry-util-http`
- **Poetry issues**: Delete `poetry.lock` and run `poetry install --no-root` again  
- **Environment errors**: Copy `.env.sample` to `.env` and edit with your credentials
- **CLI not found**: Try the fallback setup script: `bash scripts/setup.sh`

For detailed solutions, environment configuration examples, and debugging commands, see **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**.

## Development

### Adding New Agents

1. Create a new directory under `notebooks/`
2. Add your agent code, prompts, and tools
3. Create appropriate configuration files (`pyproject.toml`, `.env`)
4. Install Agent Catalog: `pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex`
5. Install root dependencies: `poetry install`
6. Run `poetry install --no-root` in the new directory
7. Run `agentc init` and `agentc index .`

### Evaluation

Run evaluations with Arize:

```bash
poetry run python evals/eval_arize.py
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

