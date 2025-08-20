#!/bin/bash

# Agent Catalog Quickstart Setup Script (Proper Python Package Management)
# This script installs Agent Catalog using pipx and virtual environments

set -e  # Exit on any error

echo "🚀 Agent Catalog Quickstart Setup (Proper Package Management)"
echo ""
echo "ℹ️  This script uses proper Python package management:"
echo "   • pipx for CLI tools (isolated environments)"
echo "   • Poetry for project dependencies"
echo "   • No system package breaking"
echo ""

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

# Check if pipx is available, install if not
if ! command -v pipx &> /dev/null; then
    echo "📦 Installing pipx..."
    if command -v brew &> /dev/null; then
        brew install pipx
    else
        python3 -m pip install --user pipx
        python3 -m pipx ensurepath
    fi
    echo "✅ pipx installed"
else
    echo "✅ pipx found"
fi

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

# Install Agent Catalog CLI tools with pipx
echo "📦 Installing Agent Catalog CLI tools with pipx..."

echo "  Installing agentc CLI (main application)..."
pipx install -e ./agent-catalog/libs/agentc_cli/ --force

echo "  Injecting core dependencies..."
pipx inject agentc-cli -e ./agent-catalog/libs/agentc_core/
pipx inject agentc-cli -e ./agent-catalog/libs/agentc/

echo "  Injecting integrations..."
pipx inject agentc-cli -e ./agent-catalog/libs/agentc_integrations/langchain/
pipx inject agentc-cli -e ./agent-catalog/libs/agentc_integrations/langgraph/
pipx inject agentc-cli -e ./agent-catalog/libs/agentc_integrations/llamaindex/

# Install Poetry dependencies for each agent
echo "📦 Installing agent dependencies with Poetry..."

echo "  Setting up Flight Search Agent (LangGraph)..."
poetry -C notebooks/flight_search_agent_langraph install --no-root

echo "  Setting up Hotel Search Agent (LangChain)..."
poetry -C notebooks/hotel_search_agent_langchain install --no-root

echo "  Setting up Landmark Search Agent (LlamaIndex)..."
poetry -C notebooks/landmark_search_agent_llamaindex install --no-root

echo "✅ Agent Catalog installation complete!"

# Verify installation
if command -v agentc &> /dev/null; then
    echo "✅ Global agentc CLI is available"
    echo ""
    echo "🎉 Setup complete! You can now:"
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
else
    echo "❌ agentc command not found. Try restarting your shell."
    exit 1
fi
