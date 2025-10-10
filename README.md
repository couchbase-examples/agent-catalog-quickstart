# Agent Catalog Quickstart

This repository provides a quickstart guide for using the Agent Catalog with Capella Model Services and Couchbase.

## Prerequisites

- Python 3.12+
- Poetry ([Installation Guide](https://python-poetry.org/docs/#installation))
- Git (for repository management)
- An OpenAI API Key (or other LLM provider)
- Couchbase Capella account (or local Couchbase installation)

## Quick Start

Two ways to get running fast. Choose one.

### 1) Full repo setup (script)

```bash
git clone --recursive https://github.com/couchbaselabs/agent-catalog-quickstart.git
cd agent-catalog-quickstart
bash scripts/setup.sh --yes               # add --skip-testing to speed up

# pick an agent and configure env
cd notebooks/hotel_search_agent_langchain
cp .env.sample .env && $EDITOR .env

# run
poetry run python main.py
```

### 2) Per-agent setup (fastest)

```bash
# from repo root
poetry -C notebooks/flight_search_agent_langraph install --no-root
cp notebooks/flight_search_agent_langraph/.env.sample notebooks/flight_search_agent_langraph/.env
$EDITOR notebooks/flight_search_agent_langraph/.env
poetry -C notebooks/flight_search_agent_langraph run python main.py
```

### 3) Installation Methods & Package Management

#### **Recommended: pipx for CLI (Isolated)**

For the cleanest installation that avoids system conflicts:

```bash
# Install pipx if not available
brew install pipx  # macOS
# or: python3 -m pip install --user pipx

# Use the proper pipx setup script
bash scripts/setup_pipx.sh
```

**Benefits:**

- Isolated environments prevent conflicts
- Clean separation between CLI tools and project dependencies
- Works with externally managed Python environments (Homebrew, system Python)

#### **Alternative: Global pip installs (PyPI)**

⚠️ **Note**: This method may conflict with externally managed environments. Use only if pipx fails.

```bash
# Install Agent Catalog packages (add --break-system-packages if needed)
pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex

# Install Arize Phoenix and evaluation dependencies
pip3 install "arize-phoenix[evals]" arize arize-otel openinference-instrumentation-langchain openinference-instrumentation-openai openinference-instrumentation-llama-index

# Fix OpenTelemetry version conflicts (if needed)
pip3 install --upgrade opentelemetry-instrumentation-asgi opentelemetry-instrumentation-fastapi opentelemetry-util-http
```

Then run the per-agent commands under "2) Per-agent setup (fastest)" above.

#### **Development Note**

These packages are currently installed from source during development. Once agentc packages are available on PyPI, the installation will be simplified to standard pip/pipx commands.

### Working with Git Submodules

This repository uses git submodules to manage the Agent Catalog dependency. If you encounter issues:

```bash
# If you cloned without --recursive, initialize submodules manually:
git submodule update --init --recursive

# Update submodules to latest versions:
git submodule update --remote

# Verify submodules are properly initialized:
git submodule status
```

**Note**: The `agent-catalog` directory is managed as a submodule and should not be manually edited.

## Per-Agent Details

Each example is independent and includes code, prompts, tools, and evals.

### 🛩️ Flight Search Agent (`notebooks/flight_search_agent_langraph/`)

- Framework: LangGraph
- Install: `poetry -C notebooks/flight_search_agent_langraph install --no-root`
- Run: `poetry -C notebooks/flight_search_agent_langraph run python main.py`

### 🏨 Hotel Support Agent (`notebooks/hotel_search_agent_langchain/`)

- Framework: LangChain
- Install: `poetry -C notebooks/hotel_search_agent_langchain install --no-root`
- Run: `poetry -C notebooks/hotel_search_agent_langchain run python main.py`

### 🗺️ Landmark Search Agent (`notebooks/landmark_search_agent_llamaindex/`)

- Framework: LlamaIndex
- Install: `poetry -C notebooks/landmark_search_agent_llamaindex install --no-root`
- Run: `poetry -C notebooks/landmark_search_agent_llamaindex run python main.py`

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

For complete environment configuration examples (Capella vs Local), see **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#environment-configuration-examples)**.

## Usage

```bash
# run with a query
poetry -C notebooks/hotel_search_agent_langchain run python main.py "Find hotels in Paris with free breakfast"

# run evaluations (Arize)
poetry -C notebooks/hotel_search_agent_langchain run python evals/eval_arize.py
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

## Agent Catalog CLI Commands

| Command          | Description                                          |
| ---------------- | ---------------------------------------------------- |
| `agentc init`    | Initialize agent catalog in current directory        |
| `agentc index .` | Index the current agent directory                    |
| `agentc publish` | Publish agent to catalog (requires clean git status) |
| `agentc --help`  | Show all available commands                          |
| `agentc env`     | Show environment configuration                       |

## Troubleshooting

Having issues? Check our comprehensive troubleshooting guide: **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**

### Quick Fixes

- "No module named 'agentc'": ensure you ran `poetry -C notebooks/<agent> install --no-root` and are executing with Poetry (`poetry run ...`).
- Evaluation deps missing: run the agent-specific Poetry install again; eval deps are included per agent.
- Poetry issues: delete the agent’s `poetry.lock` and run `poetry -C notebooks/<agent> install --no-root`.
- Environment errors: copy `.env.sample` to `.env` in the agent folder and fill in credentials.
- CLI not found after script: restart your shell or run `export PATH="$PATH:$HOME/.local/bin"`; rerun `bash scripts/setup.sh --yes` if needed.
- Submodule issues: run `git submodule update --init --recursive` to initialize the agent-catalog dependency.

For detailed solutions, environment configuration examples, and debugging commands, see **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**.

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

## Repository Resources

### 📚 Documentation (`docs/`)

Comprehensive guides and references for working with the Agent Catalog ecosystem:

| Document | Purpose |
|---|---|
| **[PYPROJECT_GUIDE.md](docs/PYPROJECT_GUIDE.md)** | Complete guide to `pyproject.toml` configuration |
| **[AGENTC_GUIDE.md](docs/AGENTC_GUIDE.md)** | Agent Catalog CLI and usage documentation |
| **[PYTHON_LINUX.md](docs/PYTHON_LINUX.md)** | Linux Python setup and pip troubleshooting |
| **[PYTHON_MAC.md](docs/PYTHON_MAC.md)** | macOS Python environment setup guide |
| **[CAPELLA_MODELS.md](docs/CAPELLA_MODELS.md)** | Couchbase Capella model services integration |
| **[EVALUATION_FRAMEWORKS_COMPARISON.md](docs/EVALUATION_FRAMEWORKS_COMPARISON.md)** | Comparison of AI evaluation frameworks |

**Quick access:**
```bash
# View documentation
ls docs/                    # List all documentation files
cat docs/PYPROJECT_GUIDE.md # Read specific guide
```

### 🛠️ Templates (`templates/`)

Ready-to-use templates for creating Agent Catalog components:

| Template | Purpose | Usage |
|---|---|---|
| **`prompt_template.yaml`** | Agent prompt templates | Create new prompts with proper structure |
| **`python_function_template.py`** | Python tool functions | Build custom tools and utilities |
| **`semantic_search_template.yaml`** | Couchbase vector search | Set up semantic search functionality |
| **`sqlpp_query_template.sqlpp`** | Database queries | Create SQL++ queries for Couchbase |
| **`http_request_template.yaml`** | HTTP/API requests | Build HTTP request tools |
| **`agentc_command_notes.txt`** | CLI command reference | AgentC command examples |

**Using templates:**
```bash
# Copy template for new component
cp templates/prompt_template.yaml prompts/my_new_prompt.yaml
cp templates/python_function_template.py tools/my_new_tool.py

# Edit with your specific requirements
$EDITOR prompts/my_new_prompt.yaml
```

### 🔧 Shared Resources (`shared/`)

Common utilities and configurations used across all agents:

| File | Purpose |
|---|---|
| **`agent_setup.py`** | Common agent initialization and setup utilities |
| **`couchbase_client.py`** | Couchbase database connection and client management |
| **`capella_model_services_langchain.py`** | LangChain integration with Capella model services |
| **`capella_model_services_llamaindex.py`** | LlamaIndex integration with Capella model services |
| **`__init__.py`** | Package initialization for shared utilities |

**Using shared resources:**
```python
# Import shared utilities in your agent
from shared.couchbase_client import get_couchbase_client
from shared.agent_setup import initialize_agent
from shared.capella_model_services_langchain import get_langchain_llm
```

### 📋 Scripts (`scripts/`)

Automation and setup scripts for the repository:

| Script | Purpose |
|---|---|
| **`setup.sh`** | Full repository setup and installation |
| **`setup_pipx.sh`** | Clean pipx-based installation (recommended) |
| **`scope_copy.py`** | Utility for copying agent scopes |

**Using scripts:**
```bash
# Run setup scripts
bash scripts/setup.sh --yes           # Full setup
bash scripts/setup_pipx.sh           # Clean pipx setup
python scripts/scope_copy.py         # Utility script
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
