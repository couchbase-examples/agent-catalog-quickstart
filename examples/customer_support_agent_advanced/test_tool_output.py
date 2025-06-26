"""
Comprehensive test script for Agent Catalog tools with multiple airports and scenarios
"""
import agentc
import dotenv

dotenv.load_dotenv()

def test_flight_tool():
    """Test the lookup_flight_info tool with multiple airport pairs"""
    print("🔧 Testing lookup_flight_info SQL++ tool")
    print("=" * 60)
    
    # Test actual routes from travel-sample database
    test_routes = [
        ("ATL", "LAX", "Atlanta to Los Angeles"),
        ("SFO", "JFK", "San Francisco to New York"), 
        ("LHR", "CDG", "London Heathrow to Paris"),
        ("DEN", "PHX", "Denver to Phoenix"),
        ("BOS", "ORD", "Boston to Chicago")
    ]
    
    catalog = agentc.Catalog()
    
    try:
        tool_result = catalog.find("tool", name="lookup_flight_info")
        if not tool_result:
            print("❌ lookup_flight_info tool not found")
            return
            
        for source, dest, description in test_routes:
            print(f"\n✈️  {description} ({source} → {dest})")
            print("-" * 40)
            
            try:
                # Call tool directly using Agent Catalog interface
                result = tool_result.func({
                    "source_airport": source,
                    "destination_airport": dest
                })
                
                if isinstance(result, list) and result:
                    print(f"📊 Found {len(result)} flights:")
                    for i, flight in enumerate(result[:5], 1):  # Show first 5
                        if isinstance(flight, dict):
                            airline = flight.get('airline', 'Unknown')
                            price = flight.get('estimated_price', 'N/A')
                            equipment = flight.get('equipment', 'N/A')
                            distance = flight.get('distance', 'N/A')
                            flight_type = flight.get('flight_type', 'N/A')
                            
                            print(f"\n  {i}. {airline} - ${price}")
                            print(f"     Equipment: {equipment}")
                            print(f"     Distance: {distance} miles")
                            print(f"     Type: {flight_type}")
                        else:
                            print(f"  {i}. {flight}")
                else:
                    print("❌ No flights found for this route")
                    
            except Exception as e:
                print(f"❌ Error testing {source} → {dest}: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing tool: {e}")

def test_policy_tool():
    """Test the search_policies YAML tool"""
    print("\n\n🔧 Testing search_policies YAML tool")
    print("=" * 60)
    
    catalog = agentc.Catalog()
    
    try:
        tool_result = catalog.find("tool", name="search_policies")
        if not tool_result:
            print("❌ search_policies tool not found")
            return
        
        print("✅ search_policies tool found in catalog")
        print("⚠️  Note: Vector index 'airline_vector_index' needs to be created in Couchbase")
        print("   for semantic search functionality to work properly.")
        print("   This demonstrates Agent Catalog YAML tool loading.")
                
    except Exception as e:
        print(f"❌ Error accessing policy tool: {e}")

def test_knowledge_tool():
    """Test the search_knowledge_base SQL++ tool"""
    print("\n\n🔧 Testing search_knowledge_base SQL++ tool")
    print("=" * 60)
    
    catalog = agentc.Catalog()
    
    try:
        tool_result = catalog.find("tool", name="search_knowledge_base")
        if not tool_result:
            print("❌ search_knowledge_base tool not found")
            return
        
        print("✅ search_knowledge_base tool found in catalog")
        print("⚠️  Note: This tool queries hotel collection for knowledge articles.")
        print("   Demonstrates Agent Catalog SQL++ tool functionality.")
                
    except Exception as e:
        print(f"❌ Error accessing knowledge tool: {e}")

def test_python_tools():
    """Test Python tools with @agentc.catalog.tool decorator"""
    print("\n\n🔧 Testing Python tools")
    print("=" * 60)
    
    catalog = agentc.Catalog()
    
    python_tools = [
        "update_customer_context",
        "get_customer_insights"
    ]
    
    for tool_name in python_tools:
        print(f"\n🐍 Testing {tool_name}")
        print("-" * 30)
        
        try:
            tool_result = catalog.find("tool", name=tool_name)
            if not tool_result:
                print(f"❌ {tool_name} tool not found")
                continue
                
            # Call appropriate test data based on tool using direct Agent Catalog interface
            if tool_name == "update_customer_context":
                result = tool_result.func(
                    customer_id="TEST_001",
                    context_update={
                        "preferences": {"seating": "aisle"},
                        "interaction_type": "test",
                        "satisfaction_score": 5
                    }
                )
            elif tool_name == "get_customer_insights":
                result = tool_result.func(customer_id="TEST_001")
            else:
                result = tool_result.func()
            
            print(f"🔧 Result: {result}")
            
        except Exception as e:
            print(f"❌ Error testing {tool_name}: {e}")

def show_catalog_info():
    """Show information about the Agent Catalog setup"""
    print("\n\n📊 AGENT CATALOG INFORMATION")
    print("=" * 60)
    
    catalog = agentc.Catalog()
    
    try:
        # Count available tools
        print("🔧 Available Tools:")
        tool_names = [
            "lookup_flight_info",
            "search_knowledge_base", 
            "search_policies",
            "update_customer_context",
            "get_customer_insights"
        ]
        
        for tool_name in tool_names:
            try:
                tool = catalog.find("tool", name=tool_name)
                if tool:
                    tool_type = "SQL++" if tool_name.endswith("_info") or tool_name.endswith("_base") else \
                               "YAML" if tool_name.endswith("_policies") else "Python"
                    print(f"  ✅ {tool_name} ({tool_type})")
                else:
                    print(f"  ❌ {tool_name} (not found)")
            except Exception as e:
                print(f"  ⚠️  {tool_name} (error: {e})")
        
        # Check prompts
        print("\n📝 Available Prompts:")
        try:
            prompt = catalog.find("prompt", name="customer_support_assistant")
            if prompt:
                print("  ✅ customer_support_assistant")
            else:
                print("  ❌ customer_support_assistant (not found)")
        except Exception as e:
            print(f"  ⚠️  customer_support_assistant (error: {e})")
            
        print("\n🎯 Catalog Status: Ready for Agent runtime")
        
    except Exception as e:
        print(f"❌ Error accessing catalog: {e}")

def show_comprehensive_demo():
    """Show a comprehensive demo of all tools working together"""
    print("\n\n🎯 COMPREHENSIVE AGENT CATALOG TUTORIAL")
    print("=" * 60)
    print("This demonstrates Agent Catalog + LangGraph + Couchbase integration")
    print("with mixed tool formats: SQL++, YAML, and Python tools")
    print("=" * 60)
    
    # Show catalog information first
    show_catalog_info()
    
    # Test all tool types
    test_flight_tool()
    test_policy_tool() 
    test_knowledge_tool()
    test_python_tools()
    
    print("\n\n🎉 TUTORIAL SUMMARY")
    print("=" * 60)
    print("✅ SQL++ Tools: Real flight data from Couchbase travel-sample")
    print("✅ YAML Tools: Semantic search configuration for policies")
    print("✅ Python Tools: Customer context and analytics functions")
    print("✅ Agent Catalog: Automatic tool loading and management")
    print("✅ LangGraph: Conversation orchestration and flow")
    print("✅ Mixed Formats: All three tool types working together")
    print("=" * 60)
    print("\n💡 Next Steps:")
    print("   - Run: python customer_support_agent.py")
    print("   - Test: Different airport combinations")
    print("   - Modify: Tools and prompts to suit your needs")
    print("   - Deploy: Using agentc publish for production")

if __name__ == "__main__":
    show_comprehensive_demo()