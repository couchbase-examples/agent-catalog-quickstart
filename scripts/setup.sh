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
read -p "Continue with full setup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled. Use 'poetry install --no-root' for individual agents."
    exit 0
fi

echo "🚀 Starting full Agent Catalog setup..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is required but not installed. Please install Poetry first:"
    echo "   https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "✅ Poetry found"

# Install Agent Catalog libraries in dependency order
echo "📦 Installing Agent Catalog libraries globally..."

echo "  Installing agentc_core..."
pip install -e ./agent-catalog/libs/agentc_core/

echo "  Installing agentc_cli..."
pip install -e ./agent-catalog/libs/agentc_cli/

echo "  Installing agentc main package..."
pip install -e ./agent-catalog/libs/agentc/

echo "  Installing langchain integration..."
pip install -e ./agent-catalog/libs/agentc_integrations/langchain/

echo "  Installing langgraph integration..."
pip install -e ./agent-catalog/libs/agentc_integrations/langgraph/

echo "  Installing llamaindex integration..."
pip install -e ./agent-catalog/libs/agentc_integrations/llamaindex/

echo "  Installing agentc_testing..."
pip install -e ./agent-catalog/libs/agentc_testing/

# Install Poetry dependencies for each agent
echo "📦 Installing agent dependencies with Poetry..."

echo "  Setting up Flight Search Agent (LangGraph)..."
cd notebooks/flight_search_agent_langraph && poetry install --no-root && cd ../..

echo "  Setting up Hotel Search Agent (LangChain)..."
cd notebooks/hotel_search_agent_langchain && poetry install --no-root && cd ../..

echo "  Setting up Landmark Search Agent (LlamaIndex)..."
cd notebooks/landmark_search_agent_llamaindex && poetry install --no-root && cd ../..

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
            echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
            echo "✅ Added to ~/.zshrc"
        else
            echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bash_profile
            echo "✅ Added to ~/.bash_profile"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Linux
        echo "Detected Linux..."
        if [[ "$SHELL" == *"zsh"* ]]; then
            echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc
            echo "✅ Added to ~/.zshrc"
        else
            echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
            echo "✅ Added to ~/.bashrc"
        fi
    else
        # Fallback for other systems
        echo "Unknown OS, adding to both ~/.bashrc and ~/.zshrc..."
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
        echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.zshrc 2>/dev/null || true
    fi
    
    export PATH="$PATH:$LOCAL_BIN"
    echo "✅ PATH updated (restart shell or run: export PATH=\"\$PATH:$LOCAL_BIN\")"
else
    echo "✅ PATH already includes $LOCAL_BIN"
fi

echo "✅ Agent Catalog global installation complete!"

# Verify installation
export PATH="$PATH:$LOCAL_BIN"
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

