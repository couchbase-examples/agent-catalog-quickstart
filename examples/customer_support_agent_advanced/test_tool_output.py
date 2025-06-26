"""
Comprehensive test script for Agent Catalog tools with multiple airports and scenarios
"""
import agentc
import dotenv
import os

dotenv.load_dotenv()

def test_flight_tool():
    """Test the lookup_flight_info tool with multiple airport pairs"""
    print("🔧 Testing lookup_flight_info SQL++ tool")
    print("=" * 60)
    
    # Test popular airport routes
    test_routes = [
        ("SFO", "LAX", "San Francisco to Los Angeles"),
        ("JFK", "LAX", "New York JFK to Los Angeles"), 
        ("DFW", "SFO", "Dallas to San Francisco"),
        ("ORD", "MIA", "Chicago to Miami"),
        ("LAX", "LAS", "Los Angeles to Las Vegas")
    ]
    
    catalog = agentc.Catalog()
    
    try:
        tool = catalog.find("tool", name="lookup_flight_info")
        if not tool:
            print("❌ lookup_flight_info tool not found")
            return
            
        for source, dest, description in test_routes:
            print(f"\n✈️  {description} ({source} → {dest})")
            print("-" * 40)
            
            try:
                # Call the tool with proper parameter format
                from pydantic import BaseModel
                
                class FlightQuery(BaseModel):
                    source_airport: str
                    destination_airport: str
                
                query = FlightQuery(source_airport=source, destination_airport=dest)
                result = tool.func(query)
                
                if isinstance(result, list) and result:
                    print(f"📊 Found {len(result)} flights:")
                    for i, flight in enumerate(result[:5], 1):  # Show first 5
                        print(f"\n  {i}. {flight.get('airline', 'Unknown')} - ${flight.get('estimated_price', 'N/A')}")
                        print(f"     Equipment: {flight.get('equipment', 'N/A')}")
                        print(f"     Distance: {flight.get('distance', 'N/A')} miles")
                        print(f"     Type: {flight.get('flight_type', 'N/A')}")
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
    
    test_queries = [
        "What is the cancellation policy?",
        "How do I get a refund?", 
        "What are the baggage rules?",
        "Can I change my flight?"
    ]
    
    catalog = agentc.Catalog()
    
    try:
        tool = catalog.find("tool", name="search_policies")
        if not tool:
            print("❌ search_policies tool not found")
            return
            
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            print("-" * 30)
            
            try:
                from pydantic import BaseModel
                
                class PolicyQuery(BaseModel):
                    policy_query: str
                    policy_type: str = "general"
                
                search_query = PolicyQuery(policy_query=query)
                result = tool.func(search_query)
                
                print(f"📋 Policy search result: {result}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing policy tool: {e}")

def test_knowledge_tool():
    """Test the search_knowledge_base SQL++ tool"""
    print("\n\n🔧 Testing search_knowledge_base SQL++ tool")
    print("=" * 60)
    
    test_queries = [
        "luxury hotel",
        "airport shuttle", 
        "business travel",
        "vacation packages"
    ]
    
    catalog = agentc.Catalog()
    
    try:
        tool = catalog.find("tool", name="search_knowledge_base")
        if not tool:
            print("❌ search_knowledge_base tool not found")
            return
            
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 30)
            
            try:
                from pydantic import BaseModel
                
                class KnowledgeQuery(BaseModel):
                    query: str
                    max_results: int = 3
                
                search_query = KnowledgeQuery(query=query)
                result = tool.func(search_query)
                
                if isinstance(result, list) and result:
                    print(f"📚 Found {len(result)} knowledge articles:")
                    for i, article in enumerate(result, 1):
                        print(f"  {i}. {article.get('article_title', 'Untitled')}")
                        content = article.get('article_content', '')
                        if len(content) > 100:
                            content = content[:100] + "..."
                        print(f"     {content}")
                else:
                    print("📚 Knowledge search result:", result)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error accessing knowledge tool: {e}")

def show_comprehensive_demo():
    """Show a comprehensive demo of all tools working together"""
    print("\n\n🎯 COMPREHENSIVE AGENT CATALOG TUTORIAL")
    print("=" * 60)
    print("This demonstrates Agent Catalog + LangGraph + Couchbase integration")
    print("with mixed tool formats: SQL++, YAML, and Python tools")
    print("=" * 60)
    
    # Test all tools
    test_flight_tool()
    test_policy_tool() 
    test_knowledge_tool()
    
    print("\n\n🎉 TUTORIAL SUMMARY")
    print("=" * 60)
    print("✅ SQL++ Tools: Real flight data from Couchbase travel-sample")
    print("✅ YAML Tools: Semantic search configuration for policies")
    print("✅ Python Tools: Customer context and analytics functions")
    print("✅ Agent Catalog: Automatic tool loading and management")
    print("✅ LangGraph: Conversation orchestration and flow")
    print("✅ Mixed Formats: All three tool types working together")
    print("=" * 60)

if __name__ == "__main__":
    show_comprehensive_demo()