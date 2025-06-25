# Enhanced Customer Support Agent with Vector Search

This example demonstrates a sophisticated customer support system that combines Couchbase Vector Search capabilities with Agent Catalog tools and prompts, inspired by the LangGraph customer support tutorial but enhanced with Capella AI Services.

## 🎯 Key Features

- **Vector Search Integration**: Semantic search across knowledge base and policies using Couchbase Vector Search
- **Multi-Tool Architecture**: Coordinated use of SQL++, semantic search, and Python function tools
- **Agent Catalog Integration**: Centralized prompt and tool management with versioning
- **Capella AI Services**: Uses Capella-hosted Claude models for enhanced reasoning
- **Context Management**: Maintains customer context and interaction history
- **LangGraph Workflow**: Sophisticated conversation flow with conditional logic

## 🏗️ Architecture

The system uses a graph-based conversation flow with two main nodes:

1. **Assistant Node**: Processes customer input and decides on actions using Agent Catalog prompts
2. **Tools Node**: Executes relevant tools based on customer needs

### Available Tools

1. **`search_knowledge_base.yaml`** - Semantic search for support articles and FAQs
2. **`lookup_flight_info.sqlpp`** - SQL++ queries for flight data from travel-sample
3. **`search_policies.yaml`** - Vector search for airline policies and regulations  
4. **`update_customer_context.py`** - Python function for customer context management

### Prompts

- **`customer_support_assistant.yaml`** - Main assistant prompt with comprehensive instructions

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Poetry for dependency management
- Couchbase instance with travel-sample bucket
- Agent Catalog configured and indexed

### Setup

1. **Install dependencies:**
   ```bash
   cd examples/customer_support_agent
   poetry install
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your Couchbase and Capella credentials
   ```

3. **Initialize Agent Catalog:**
   ```bash
   agentc init
   agentc index .
   agentc publish
   ```

4. **Run the agent:**
   ```bash
   python customer_support_agent.py
   ```

## 🎭 Example Scenarios

The demo includes three realistic customer support scenarios:

### Scenario 1: Policy Inquiry
```
Customer: "Hi, I need help finding information about flight cancellation policies for my upcoming trip to Paris."

Agent: Uses search_policies tool to find relevant cancellation policies
      → Provides accurate, current policy information
      → Updates customer context with interaction
```

### Scenario 2: Flight Search
```
Customer: "I'm looking for flights from SFO to JFK next week. Can you help me find options and pricing?"

Agent: Uses lookup_flight_info tool to query flight data
      → Returns available routes with airlines and details
      → Offers additional assistance for booking
```

### Scenario 3: Issue Resolution
```
Customer: "My flight was delayed and I missed my connection. What are my options for rebooking?"

Agent: Uses search_knowledge_base for delay procedures
      → Uses search_policies for rebooking policies  
      → Provides step-by-step resolution options
```

## 🔧 Key Components

### Vector Search Implementation

The example uses Couchbase Vector Search with:
- **Embedding Model**: `sentence-transformers/all-MiniLM-L12-v2`
- **Vector Indices**: Pre-built on knowledge base and policy collections
- **Semantic Similarity**: Finds relevant content based on customer queries

### Agent Catalog Integration

- **Centralized Tool Management**: All tools versioned and managed through Agent Catalog
- **Prompt Engineering**: Structured prompts with clear instructions and output schemas
- **Environment Configuration**: Seamless integration with Couchbase and Capella services

### LangGraph Workflow

- **State Management**: Maintains conversation state with customer context
- **Conditional Logic**: Intelligent routing between assistant and tools
- **Tool Coordination**: Strategic use of multiple tools within single conversations

## 📊 Advantages Over Basic Examples

This enhanced example provides:

1. **Real Vector Search**: Actual semantic search capabilities vs. simple keyword matching
2. **Production Architecture**: Scalable, maintainable code structure
3. **Comprehensive Toolset**: Multiple tool types working together
4. **Customer Context**: Persistent context and personalization
5. **Policy Compliance**: Proper handling of customer data and privacy
6. **Realistic Scenarios**: Complex, multi-turn conversations

## 🔮 Extensions

This example can be extended with:

- **Real-time Integration**: Connect to live booking systems
- **Multi-language Support**: Internationalization for global customer base
- **Analytics Dashboard**: Monitor agent performance and customer satisfaction
- **Voice Integration**: Add speech-to-text and text-to-speech capabilities
- **Escalation Workflows**: Automated escalation to human agents when needed

## 📚 Learning Objectives

By studying this example, you'll learn:

- How to implement vector search in customer support scenarios
- Best practices for Agent Catalog tool and prompt design
- LangGraph patterns for complex conversation flows
- Integration techniques for Couchbase and Capella services
- Production-ready agent architecture patterns