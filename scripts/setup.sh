#!/bin/bash

# Agent Catalog Quickstart Setup Script
# This script installs the Agent Catalog CLI and required dependencies

set -e  # Exit on any error

echo "🚀 Setting up Agent Catalog CLI..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

echo "✅ Python 3 found"

# Install Agent Catalog libraries in dependency order
echo "📦 Installing Agent Catalog libraries..."

echo "  Installing agentc_core..."
pip install ./agent-catalog/libs/agentc_core/

echo "  Installing langchain integration..."
pip install ./agent-catalog/libs/agentc_integrations/langchain/

echo "  Installing langgraph integration..."
pip install ./agent-catalog/libs/agentc_integrations/langgraph/

echo "  Installing llamaindex integration..."
pip install ./agent-catalog/libs/agentc_integrations/llamaindex/

echo "  Installing agentc_cli..."
pip install ./agent-catalog/libs/agentc_cli/

echo "  Installing agentc_testing..."
pip install ./agent-catalog/libs/agentc_testing/

echo "  Installing agentc main package..."
pip install ./agent-catalog/libs/agentc/

# Install LlamaIndex dependencies
echo "📦 Installing LlamaIndex dependencies..."
pip install llama-index-vector-stores-couchbase llama-index-embeddings-openai llama-index-llms-openai llama-index

echo "✅ Agent Catalog CLI installation complete!"

# Verify installation
if command -v agentc &> /dev/null; then
    echo "✅ agentc command is available"
    echo ""
    echo "🎉 Setup complete! Next steps:"
    echo "   1. Configure Poetry environments (see README.md)"
    echo "   2. Set up .env files (see README.md)"
    echo "   3. Run 'agentc --help' to see available commands"
else
    echo "❌ agentc command not found. Installation may have failed."
    exit 1
fi

