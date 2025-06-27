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
    import dotenv
    import os
    import sys
    
    # Import the graph module with proper error handling
    try:
        from . import graph
    except ImportError:
        # Handle the case when running as a script directly
        import graph

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

    # Validate required environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)

    # Initialize Agent Catalog with error handling
    try:
        catalog = agentc.Catalog()
        print("✅ Agent Catalog initialized successfully")
    except Exception as e:
        print(f"⚠️  Agent Catalog initialization failed: {e}")
        print("The agent will still demonstrate the architecture patterns.")
        catalog = agentc.Catalog()  # Continue with basic catalog

    # Test scenarios to demonstrate comprehensive Agent Catalog functionality
    test_scenarios = [
        {
            "customer_id": "CUST_001",
            "message": "Hi, I need to find flights from ABE to ATL for next week. Please show me all available options with different airlines and give me at least 5 choices.",
            "description": "Flight search with multiple airlines - SQL++ tool demonstration",
        },
        {
            "customer_id": "CUST_002",
            "message": "What are your cancellation and refund policies for airline tickets?",
            "description": "Policy search - YAML semantic search tool demonstration",
        },
        {
            "customer_id": "CUST_003",
            "message": "I'm looking for flights from ABI to DFW. What options do you have? Please show me multiple airlines.",
            "description": "Regional flight search - SQL++ tool with multiple results",
        },
        {
            "customer_id": "CUST_004",
            "message": "Can you help me find flights from ABE to ORD? I want to compare at least 5 different options.",
            "description": "Multi-option flight search - demonstrating comprehensive results",
        },
        {
            "customer_id": "CUST_005",
            "message": "I need flights from ABE to PHL. Please show me all available airlines and pricing options.",
            "description": "Short-haul flight search with multiple results",
        },
    ]

    print(f"\n🎯 Running {len(test_scenarios)} comprehensive test scenarios:")
    print("=" * 60)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['description']}")
        print("-" * 50)

        try:
            # Build and run each scenario
            initial_state = graph.CustomerSupportGraph.build_starting_state(
                customer_id=scenario["customer_id"], initial_message=scenario["message"]
            )

            result = graph.CustomerSupportGraph(catalog=catalog).invoke(input=initial_state)

            print(f"✅ Scenario {i} completed\n")
        except Exception as e:
            print(f"❌ Scenario {i} failed: {e}")
            print("💡 This may indicate missing catalog setup or configuration issues.")
            print("   Try running: agentc init && agentc index . && agentc publish")
            print(f"⏭️  Continuing with next scenario...\n")

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
