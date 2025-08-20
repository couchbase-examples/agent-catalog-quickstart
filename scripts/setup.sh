#!/bin/bash

# Agent Catalog Quickstart Setup Script
# This script installs Agent Catalog globally and sets up all example agents

set -e  # Exit on any error

echo "🚀 Agent Catalog Quickstart Setup"
echo ""
echo "ℹ️  For a single agent, you can use the simpler approach:"
echo "   cd notebooks/hotel_search_agent_langchain"
echo "   poetry install --no-root"
echo ""
echo "🔧 This script provides comprehensive setup for:"
echo "   • Global agentc CLI installation"
echo "   • All three example agents"
echo "   • Development tools and dependencies"
echo ""
# Options (env or flags)
ASSUME_YES="${AGENTC_SETUP_ASSUME_YES:-0}"
SKIP_TESTING="${AGENTC_SETUP_SKIP_TESTING:-0}"
for arg in "$@"; do
    case "$arg" in
        --yes|-y|--non-interactive)
            ASSUME_YES=1
            ;;
        --skip-testing)
            SKIP_TESTING=1
            ;;
    esac
done

if [[ "$ASSUME_YES" -ne 1 ]]; then
    read -p "Continue with full setup? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Use 'poetry install --no-root' for individual agents."
        exit 0
    fi
else
    echo "Non-interactive mode enabled; continuing without prompt."
fi

echo "🚀 Starting full Agent Catalog setup..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found"

# Check Python version >= 3.12
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 12) else 1)' >/dev/null 2>&1; then
    echo "❌ Python 3.12+ is required. Detected: $(python3 -V 2>&1)"
    exit 1
fi

echo "✅ Python version is >= 3.12"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is required but not installed. Please install Poetry first:"
    echo "   https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✅ Poetry found"

# Initialize git submodules if not already done
echo "🔄 Checking git submodules..."
if [[ ! -f "./agent-catalog/.git" ]] && [[ ! -d "./agent-catalog/.git" ]]; then
    echo "📦 Initializing agent-catalog submodule..."
    git submodule update --init --recursive
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to initialize git submodules. Please run:"
        echo "   git submodule update --init --recursive"
        exit 1
    fi
    echo "✅ Agent-catalog submodule initialized"
else
    echo "✅ Agent-catalog submodule already initialized"
fi

# Verify agent-catalog libs directory exists
if [[ ! -d "./agent-catalog/libs" ]]; then
    echo "❌ ./agent-catalog/libs not found. The agent-catalog submodule may not be properly initialized."
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Install Agent Catalog libraries in dependency order
echo "📦 Installing Agent Catalog libraries globally..."

echo "  Installing agentc_core..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc_core/

echo "  Installing agentc_cli..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc_cli/

echo "  Installing agentc main package..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc/

echo "  Installing langchain integration..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc_integrations/langchain/

echo "  Installing langgraph integration..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc_integrations/langgraph/

echo "  Installing llamaindex integration..."
python3 -m pip install --user -e ./agent-catalog/libs/agentc_integrations/llamaindex/

if [[ "$SKIP_TESTING" -eq 1 ]]; then
    echo "  Skipping agentc_testing installation (flag: --skip-testing)"
else
    echo "  Installing agentc_testing..."
    python3 -m pip install --user -e ./agent-catalog/libs/agentc_testing/
fi

# Install Poetry dependencies for each agent
echo "📦 Installing agent dependencies with Poetry..."

echo "  Setting up Flight Search Agent (LangGraph)..."
poetry -C notebooks/flight_search_agent_langraph install --no-root

echo "  Setting up Hotel Search Agent (LangChain)..."
poetry -C notebooks/hotel_search_agent_langchain install --no-root

echo "  Setting up Landmark Search Agent (LlamaIndex)..."
poetry -C notebooks/landmark_search_agent_llamaindex install --no-root

# Add local bin to PATH if not already there
echo "🔧 Configuring PATH..."
LOCAL_BIN="$HOME/.local/bin"
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo "Adding $LOCAL_BIN to PATH..."
    
    # Detect OS and configure shell accordingly
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS..."
        if [[ "$SHELL" == *"zsh"* ]]; then
            grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.zshrc || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
            echo "✅ Added to ~/.zshrc"
        else
            grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.bash_profile || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
            echo "✅ Added to ~/.bash_profile"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Linux
        echo "Detected Linux..."
        if [[ "$SHELL" == *"zsh"* ]]; then
            grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.zshrc || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
            echo "✅ Added to ~/.zshrc"
        else
            grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.bashrc || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
            echo "✅ Added to ~/.bashrc"
        fi
    else
        # Fallback for other systems
        echo "Unknown OS, adding to both ~/.bashrc and ~/.zshrc..."
        grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.bashrc || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
        grep -qxF 'export PATH="$PATH:$HOME/.local/bin"' ~/.zshrc 2>/dev/null || echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc 2>/dev/null || true
    fi
    
    export PATH="$PATH:$LOCAL_BIN"
    echo "✅ PATH updated (restart shell or run: export PATH=\"\$PATH:$LOCAL_BIN\")"
else
    echo "✅ PATH already includes $LOCAL_BIN"
fi

echo "✅ Agent Catalog global installation complete!"

# Verify installation
export PATH="$PATH:$LOCAL_BIN"
# Refresh hash table of commands for current shell session
hash -r 2>/dev/null || true
if command -v agentc &> /dev/null; then
    echo "✅ Global agentc CLI is available"
    echo ""
    echo "🎉 Full setup complete! You can now:"
    echo ""
    echo "📁 Work with any agent:"
    echo "   cd notebooks/hotel_search_agent_langchain"
    echo "   cp .env.sample .env  # Edit with your credentials"
    echo "   poetry run python main.py"
    echo ""
    echo "🔧 Use global CLI commands:"
    echo "   agentc init"
    echo "   agentc index ."
    echo "   agentc publish"
    echo ""
    echo "📖 See README.md for environment configuration details"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Set up .env files in each agent directory"
    echo "   2. Configure your Couchbase and OpenAI credentials"
    echo "   3. Run any agent with 'poetry run python main.py'"
else
    echo "❌ agentc command not found. Try restarting your shell or manually add $LOCAL_BIN to PATH"
    echo "Run: export PATH=\"\$PATH:$LOCAL_BIN\""
    exit 1
fi

