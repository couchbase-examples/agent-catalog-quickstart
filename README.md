# agent-catalog-quickstart

This repository provides a quickstart guide for using the Agent Catalog with Capella Model Services.

## Prerequisites

- Python 3.12+
- Poetry
- An OpenAI API Key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/couchbaselabs/agent-catalog-quickstart.git
    cd agent-catalog-quickstart
    ```

2.  **Create a `.env` file and add your OpenAI API key:**
    Create a file named `.env` and add the following line, replacing `your_openai_api_key_here` with your actual key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

3.  **Install dependencies and set up the catalog:**
    You can run the setup script to automate this:
    ```bash
    bash scripts/setup.sh
    ```
    Alternatively, you can run the commands manually:
    ```bash
    poetry install
    poetry run agentc init --no-db --local
    poetry run agentc index . --prompts --tools
    ```

## Run

1.  **Run the main script:**
    ```bash
    poetry run python main.py
    ```

2.  **Run the examples:**
    ```bash
    # LangGraph
    poetry run python examples/langgraph_example.py

    # LangChain
    poetry run python examples/langchain_example.py

    # LlamaIndex
    poetry run python examples/llamaindex_example.py
    ```