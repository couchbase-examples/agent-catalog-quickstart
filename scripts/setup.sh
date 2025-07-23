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

# Install additional dependencies for the quickstart
echo "📦 Installing additional dependencies..."
pip install python-dotenv langchain-openai langchain langgraph

# Install LlamaIndex dependencies
echo "📦 Installing LlamaIndex dependencies..."
pip install llama-index-vector-stores-couchbase llama-index-embeddings-openai llama-index-llms-openai llama-index

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

echo "✅ Agent Catalog CLI installation complete!"

# Verify installation
export PATH="$PATH:$LOCAL_BIN"
if command -v agentc &> /dev/null; then
    echo "✅ agentc command is available"
    echo ""
    echo "🎉 Setup complete! Next steps:"
    echo "   1. Restart your shell or run: export PATH=\"\$PATH:$HOME/.local/bin\""
    echo "   2. Configure Poetry environments (see README.md)"
    echo "   3. Set up .env files (see README.md)"
    echo "   4. Run 'agentc --help' to see available commands"
    echo "   5. Initialize a project with 'agentc init'"
else
    echo "❌ agentc command not found. Try restarting your shell or manually add $LOCAL_BIN to PATH"
    echo "Run: export PATH=\"\$PATH:$LOCAL_BIN\""
    exit 1
fi

