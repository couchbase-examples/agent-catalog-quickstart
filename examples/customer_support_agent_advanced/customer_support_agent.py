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

    # Test scenarios to demonstrate comprehensive Agent Catalog functionality
    test_scenarios = [
        {
            "customer_id": "CUST_001",
            "message": "Hi, I need to find flights from ABE to ATL for next week. Please show me all available options with different airlines and give me at least 5 choices.",
            "description": "Flight search with multiple airlines - SQL++ tool demonstration"
        },
        {
            "customer_id": "CUST_002", 
            "message": "What are your cancellation and refund policies for airline tickets?",
            "description": "Policy search - YAML semantic search tool demonstration"
        },
        {
            "customer_id": "CUST_003",
            "message": "I'm looking for flights from ABI to DFW. What options do you have? Please show me multiple airlines.",
            "description": "Regional flight search - SQL++ tool with multiple results"
        },
        {
            "customer_id": "CUST_004",
            "message": "Can you help me find flights from ABE to ORD? I want to compare at least 5 different options.",
            "description": "Multi-option flight search - demonstrating comprehensive results"
        },
        {
            "customer_id": "CUST_005",
            "message": "I need flights from ABE to PHL. Please show me all available airlines and pricing options.",
            "description": "Short-haul flight search with multiple results"
        }
    ]
    
    print(f"\n🎯 Running {len(test_scenarios)} comprehensive test scenarios:")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['description']}")
        print("-" * 50)
        
        # Build and run each scenario
        initial_state = graph.CustomerSupportGraph.build_starting_state(
            customer_id=scenario["customer_id"],
            initial_message=scenario["message"]
        )
        
        result = graph.CustomerSupportGraph(catalog=catalog).invoke(input=initial_state)
        
        print(f"✅ Scenario {i} completed\n")
    
    print("🎉 COMPREHENSIVE TUTORIAL COMPLETED!")
    print("=" * 60)
    print("This demonstrated:")
    print("✅ Agent Catalog + LangGraph + Couchbase integration")
    print("✅ Mixed tool formats: SQL++, YAML, and Python tools")
    print("✅ Real flight data from multiple airport pairs") 
    print("✅ Multiple airline options and comprehensive results")
    print("✅ Policy search with semantic matching")
    print("✅ Professional customer support conversation flow")
    print("=" * 60)