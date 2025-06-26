"""
Customer Support Agent: Agent Catalog + LangGraph + Couchbase Tutorial

This tutorial demonstrates the proper integration of:
- Agent Catalog for tool and prompt management
- LangGraph for conversation orchestration  
- Couchbase for real data integration
- Mixed tool formats: SQL++, YAML, and Python
"""

if __name__ == "__main__":
    import agentc
    import graph
    import dotenv

    # Load environment variables
    dotenv.load_dotenv()

    print("🚀 Customer Support Agent: Agent Catalog + LangGraph + Couchbase Tutorial")
    print("=" * 70)
    print("This tutorial demonstrates:")
    print("- 📊 Real flight data from Couchbase travel-sample")
    print("- 🔧 Mixed tool formats: YAML (Semantic) + SQL++ (Database) + Python (Functions)")
    print("- 🤖 Agent Catalog prompt and tool management")
    print("- 🕸️  LangGraph conversation orchestration")
    print("=" * 70)

    # The Agent Catalog 'catalog' object serves versioned tools and prompts.
    # Parameters can be set with environment variables (e.g., bucket = $AGENT_CATALOG_BUCKET).
    catalog = agentc.Catalog()

    # Build starting state and run the customer support graph
    initial_state = graph.CustomerSupportGraph.build_starting_state(
        customer_id="DEMO_CUSTOMER",
        initial_message="Hi, I need help finding flights from SFO to LAX for next week."
    )
    
    # Start the customer support application
    graph.CustomerSupportGraph(catalog=catalog).invoke(input=initial_state)