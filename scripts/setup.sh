#!/bin/bash

# This script automates the setup process for the Agent Catalog quickstart.

# Install dependencies
poetry install

# Initialize the Agent Catalog
poetry run agentc init --no-db --local

# Index the tools and prompts
poetry run agentc index . --prompts --tools
