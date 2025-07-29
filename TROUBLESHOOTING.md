# Agent Catalog Quickstart - Troubleshooting Guide

This guide covers common issues and solutions when setting up and using the Agent Catalog quickstart.

## Common Issues

### 1. **"No module named 'agentc'"**
**Solution**: Run the Agent Catalog installation command:
```bash
pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex
```
This installs all necessary Agent Catalog packages.

### 2. **"command not found: agentc"** (when using CLI)
**Solution**: Ensure the pip installation above included `agentc-cli`
- If still not working, try the fallback setup script: `bash scripts/setup.sh`
- Make sure `~/.local/bin` is in your PATH

### 3. **Poetry installation fails**
**Solution**: Try these steps in order:
```bash
# Delete lock file and try again
rm poetry.lock
poetry install --no-root

# Check Poetry version (should be 1.5+)
poetry --version

# Clear Poetry cache if needed
poetry cache clear PyPI --all
```

### 4. **"Could not find the environment variable $AGENT_CATALOG_CONN_STRING"**
**Solution**: Set up your environment file:
```bash
cp .env.sample .env
# Edit .env with your credentials
```
- Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in the `.env` file
- Make sure the `.env` file is in the correct agent directory

### 5. **Connection errors to Couchbase**
**Solution**: Verify your `.env` configuration:
- Check that your Couchbase cluster is accessible
- Ensure proper credentials and connection strings
- Test connectivity outside of the agent first

### 6. **"Certificate error" when connecting**
**Solution**: Check your connection string format:
- For **local Couchbase**: use `couchbase://127.0.0.1`
- For **Capella**: use `couchbases://your-cluster.cloud.couchbase.com`
- Include `AGENT_CATALOG_CONN_ROOT_CERTIFICATE=""` in your `.env`

### 7. **Tokenizer parallelism warnings**
**Solution**: Add this to your `.env` files:
```bash
TOKENIZERS_PARALLELISM=false
```

### 8. **Import errors for specific frameworks**
**Symptoms**: 
- `No module named 'langchain'`
- `No module named 'langgraph'`
- `No module named 'llama_index'`

**Solution**: Ensure both installation steps completed:
```bash
# Step 1: Root dependencies
poetry install

# Step 2: Notebook dependencies  
cd notebooks/[your-agent]
poetry install --no-root
```

### 9. **Poetry environment conflicts**
**Symptoms**: Packages installed but still getting import errors

**Solution**: Reset the Poetry environment:
```bash
poetry env remove python
poetry install --no-root
```

### 10. **Git repository issues when publishing**
**Symptoms**: `"Cannot publish a dirty catalog to the DB"`

**Solution**: Clean your git status:
```bash
git add .
git commit -m "Your changes"
git status  # Should show clean working tree
agentc publish
```

## Setup Requirements Checklist

### For Individual Agent (Recommended)
- [ ] **Agent Catalog packages**: `pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex`
- [ ] **Poetry installed**: `poetry --version` (should be 1.5+)
- [ ] **Root dependencies**: `poetry install` (at project root)
- [ ] **Notebook dependencies**: `cd notebooks/[agent] && poetry install --no-root`
- [ ] **Environment file**: `cp .env.sample .env` (in agent directory)
- [ ] **Credentials configured**: Edit `.env` with your actual credentials

### For Development/Fallback (Optional)
- [ ] **Local source installation**: `bash scripts/setup.sh` or manual pip installs
- [ ] **Global CLI available**: `agentc --help`
- [ ] **Git clean**: Clean repository state (for publishing)

## Environment Configuration Examples

### For Couchbase Capella (Cloud)
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

### For Local Couchbase
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

## Required Environment Files

Each example directory needs its own `.env` file:
- `notebooks/flight_search_agent_langraph/.env`
- `notebooks/hotel_search_agent_langchain/.env`
- `notebooks/landmark_search_agent_llamaindex/.env`

## Getting Additional Help

- Check the `docs/` directory for detailed guides
- Look at example implementations in `notebooks/`
- Review error messages for specific configuration issues
- Ensure you've installed Agent Catalog packages: `pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex`
- Check the main [README.md](README.md) for setup instructions

## Debugging Commands

### Check installations:
```bash
# Check Agent Catalog packages
pip list | grep agentc

# Check Poetry environment
poetry env info

# Check what's installed in Poetry env
poetry run pip list

# Test Agent Catalog import
poetry run python -c "import agentc; print('✅ Agent Catalog ready!')"
```

### Reset if needed:
```bash
# Reset Poetry environment
poetry env remove python
poetry install --no-root

# Clear Poetry cache
poetry cache clear PyPI --all

# Reinstall Agent Catalog
pip3 uninstall agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex
pip3 install agentc agentc-core agentc-cli agentc-langchain agentc-langgraph agentc-llamaindex
```
