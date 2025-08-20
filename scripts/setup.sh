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
echo "🏗️  Available flags:"
echo "   --yes / -y                Non-interactive mode"
echo "   --skip-testing           Skip agentc_testing installation"
echo "   --break-system-packages  Force use of --break-system-packages"
echo ""
# Options (env or flags)
ASSUME_YES="${AGENTC_SETUP_ASSUME_YES:-0}"
SKIP_TESTING="${AGENTC_SETUP_SKIP_TESTING:-0}"
FORCE_BREAK_SYSTEM_PACKAGES="${AGENTC_SETUP_FORCE_BREAK_SYSTEM_PACKAGES:-0}"
for arg in "$@"; do
    case "$arg" in
        --yes|-y|--non-interactive)
            ASSUME_YES=1
            ;;
        --skip-testing)
            SKIP_TESTING=1
            ;;
        --break-system-packages)
            FORCE_BREAK_SYSTEM_PACKAGES=1
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

# Function to safely install pip packages with fallback to --break-system-packages
safe_pip_install() {
    local package_path="$1"
    local package_name="$2"
    
    echo "  Installing $package_name..."
    
    # Try normal installation first (unless forced to use --break-system-packages)
    if [[ "$FORCE_BREAK_SYSTEM_PACKAGES" -eq 1 ]]; then
        echo "    Using --break-system-packages (forced by flag)"
        python3 -m pip install --user --break-system-packages -e "$package_path"
        return $?
    fi
    
    # Attempt installation without --break-system-packages
    if python3 -m pip install --user -e "$package_path" 2>/dev/null; then
        echo "    ✅ Installed successfully"
        return 0
    fi
    
    # If installation failed, check if it's due to externally managed environment
    local error_output
    error_output=$(python3 -m pip install --user -e "$package_path" 2>&1)
    
    if echo "$error_output" | grep -q "externally-managed-environment"; then
        echo ""
        echo "⚠️  External environment management detected!"
        echo "    Your Python installation is managed by your system package manager."
        echo ""
        echo "💡 Options:"
        echo "    1. Use --break-system-packages (may affect system stability)"
        echo "    2. Use a virtual environment or pipx (recommended, but requires separate setup)"
        echo "    3. Exit and set up manually"
        echo ""
        
        if [[ "$ASSUME_YES" -eq 1 ]]; then
            echo "    Non-interactive mode: Using --break-system-packages"
            python3 -m pip install --user --break-system-packages -e "$package_path"
            return $?
        fi
        
        while true; do
            read -p "    Use --break-system-packages for $package_name? (y/n/exit): " -n 1 -r
            echo
            case $REPLY in
                [Yy]* )
                    echo "    Installing with --break-system-packages..."
                    python3 -m pip install --user --break-system-packages -e "$package_path"
                    return $?
                    ;;
                [Nn]* )
                    echo "    Skipping $package_name installation"
                    echo "    ❌ You'll need to install this manually or use a virtual environment"
                    return 1
                    ;;
                [Ee]* | exit )
                    echo "    Exiting setup..."
                    exit 1
                    ;;
                * )
                    echo "    Please answer y (yes), n (no), or 'exit'"
                    ;;
            esac
        done
    else
        # Different error - show it to user
        echo "    ❌ Installation failed with error:"
        echo "$error_output"
        return 1
    fi
}

# Install Agent Catalog libraries in dependency order
echo "📦 Installing Agent Catalog libraries globally..."

safe_pip_install "./agent-catalog/libs/agentc_core/" "agentc_core"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install agentc_core. Setup cannot continue."
    exit 1
fi

safe_pip_install "./agent-catalog/libs/agentc_cli/" "agentc_cli"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install agentc_cli. Setup cannot continue."
    exit 1
fi

safe_pip_install "./agent-catalog/libs/agentc/" "agentc main package"
if [[ $? -ne 0 ]]; then
    echo "❌ Failed to install agentc main package. Setup cannot continue."
    exit 1
fi

safe_pip_install "./agent-catalog/libs/agentc_integrations/langchain/" "langchain integration"
if [[ $? -ne 0 ]]; then
    echo "⚠️  Failed to install langchain integration. Continuing..."
fi

safe_pip_install "./agent-catalog/libs/agentc_integrations/langgraph/" "langgraph integration"
if [[ $? -ne 0 ]]; then
    echo "⚠️  Failed to install langgraph integration. Continuing..."
fi

safe_pip_install "./agent-catalog/libs/agentc_integrations/llamaindex/" "llamaindex integration"
if [[ $? -ne 0 ]]; then
    echo "⚠️  Failed to install llamaindex integration. Continuing..."
fi

if [[ "$SKIP_TESTING" -eq 1 ]]; then
    echo "  Skipping agentc_testing installation (flag: --skip-testing)"
else
    safe_pip_install "./agent-catalog/libs/agentc_testing/" "agentc_testing"
    if [[ $? -ne 0 ]]; then
        echo "⚠️  Failed to install agentc_testing. Continuing..."
    fi
fi

# Install Poetry dependencies for each agent
echo "📦 Installing agent dependencies with Poetry..."

echo "  Setting up Flight Search Agent (LangGraph)..."
poetry -C notebooks/flight_search_agent_langraph install --no-root

echo "  Setting up Hotel Search Agent (LangChain)..."
poetry -C notebooks/hotel_search_agent_langchain install --no-root

echo "  Setting up Landmark Search Agent (LlamaIndex)..."
poetry -C notebooks/landmark_search_agent_llamaindex install --no-root

# Add Python user binaries to PATH if not already there
echo "🔧 Configuring PATH..."

# Detect where Python user binaries are installed
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
LOCAL_BIN="$HOME/.local/bin"
PYTHON_USER_BIN="$HOME/Library/Python/$PYTHON_VERSION/bin"

# Check which directory has the agentc executable
AGENTC_PATH=""
if [[ -f "$LOCAL_BIN/agentc" ]]; then
    AGENTC_PATH="$LOCAL_BIN"
elif [[ -f "$PYTHON_USER_BIN/agentc" ]]; then
    AGENTC_PATH="$PYTHON_USER_BIN"
fi

# Configure PATH based on detected installation
if [[ -n "$AGENTC_PATH" ]]; then
    if [[ ":$PATH:" != *":$AGENTC_PATH:"* ]]; then
        echo "Adding $AGENTC_PATH to PATH..."
        
        # Detect OS and configure shell accordingly
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            echo "Detected macOS..."
            if [[ "$SHELL" == *"zsh"* ]]; then
                # Check if either path is already configured
                if ! grep -q "Library/Python.*bin.*PATH" ~/.zshrc 2>/dev/null && ! grep -q ".local/bin.*PATH" ~/.zshrc 2>/dev/null; then
                    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.zshrc
                    echo "export PATH=\"\$HOME/Library/Python/$PYTHON_VERSION/bin:\$PATH\"" >> ~/.zshrc
                    echo "✅ Added both Python paths to ~/.zshrc"
                else
                    echo "✅ Python paths already configured in ~/.zshrc"
                fi
            else
                if ! grep -q "Library/Python.*bin.*PATH" ~/.bash_profile 2>/dev/null && ! grep -q ".local/bin.*PATH" ~/.bash_profile 2>/dev/null; then
                    echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bash_profile
                    echo "export PATH=\"\$HOME/Library/Python/$PYTHON_VERSION/bin:\$PATH\"" >> ~/.bash_profile
                    echo "✅ Added both Python paths to ~/.bash_profile"
                else
                    echo "✅ Python paths already configured in ~/.bash_profile"
                fi
            fi
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Ubuntu/Linux
            echo "Detected Linux..."
            if [[ "$SHELL" == *"zsh"* ]]; then
                grep -qxF "export PATH=\"\$HOME/.local/bin:\$PATH\"" ~/.zshrc || echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.zshrc
                echo "✅ Added to ~/.zshrc"
            else
                grep -qxF "export PATH=\"\$HOME/.local/bin:\$PATH\"" ~/.bashrc || echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> ~/.bashrc
                echo "✅ Added to ~/.bashrc"
            fi
        fi
        
        export PATH="$AGENTC_PATH:$PATH"
        echo "✅ PATH updated for current session"
    else
        echo "✅ PATH already includes $AGENTC_PATH"
    fi
else
    echo "⚠️  agentc executable not found in expected locations. You may need to add the Python user bin directory to your PATH manually."
fi

echo "✅ Agent Catalog global installation complete!"

# Verify installation
if [[ -n "$AGENTC_PATH" ]]; then
    export PATH="$AGENTC_PATH:$PATH"
fi
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
    echo ""
    echo "🔄 If agentc is not available in new terminals:"
    echo "   1. Restart your shell"
    echo "   2. Or run: source ~/.zshrc (or ~/.bash_profile)"
else
    echo "❌ agentc command not found even after PATH configuration."
    if [[ -n "$AGENTC_PATH" ]]; then
        echo "   agentc is installed at: $AGENTC_PATH/agentc"
        echo "   Try restarting your shell or run: export PATH=\"$AGENTC_PATH:\$PATH\""
    else
        echo "   Installation may have failed. Please check the errors above."
    fi
    exit 1
fi

