#!/usr/bin/env python3
"""
Customer Support Agent with Direct Couchbase Integration
Bypasses Agent Catalog tool loading issues by directly calling Couchbase
"""

import getpass
import os
from typing import Any, Dict

import couchbase.cluster
import couchbase.auth
import couchbase.options
import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated


class CustomerSupportState(TypedDict):
    """State for the customer support conversation"""
    messages: Annotated[list, add_messages]
    customer_id: str
    context: Dict[str, Any]


@tool
def lookup_flight_info(source_airport: str, destination_airport: str) -> dict:
    """Look up real flight information from travel-sample database"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv('CB_CONN_STRING'),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv('CB_USERNAME'), 
                    password=os.getenv('CB_PASSWORD')
                )
            ),
        )
        
        query = """
        SELECT 
          r.airline,
          r.sourceairport,
          r.destinationairport,
          r.equipment,
          r.distance,
          a1.airportname AS source_name,
          a1.city AS source_city,
          a2.airportname AS dest_name, 
          a2.city AS dest_city,
          CASE 
            WHEN r.distance < 500 THEN "Short-haul"
            WHEN r.distance < 1500 THEN "Medium-haul"
            ELSE "Long-haul"
          END AS flight_type,
          ROUND(r.distance / 500, 1) AS estimated_hours,
          ROUND(200 + (r.distance * 0.15), 0) AS estimated_price
        FROM 
          `travel-sample`.inventory.route r
        JOIN 
          `travel-sample`.inventory.airport a1 ON r.sourceairport = a1.faa
        JOIN 
          `travel-sample`.inventory.airport a2 ON r.destinationairport = a2.faa
        WHERE 
          r.sourceairport = $source_airport AND
          r.destinationairport = $destination_airport
        ORDER BY r.distance
        LIMIT 5
        """
        
        result = cluster.query(
            query,
            couchbase.options.QueryOptions(
                named_parameters={
                    "source_airport": source_airport, 
                    "destination_airport": destination_airport
                }
            ),
        )
        
        flights = []
        for row in result:
            flights.append(dict(row))
            
        return {
            "flights": flights,
            "total_flights": len(flights),
            "source_airport": source_airport,
            "destination_airport": destination_airport
        }
        
    except Exception as e:
        return {"error": f"Flight lookup failed: {e}", "flights": []}


@tool 
def search_knowledge_base(query: str, category: str = "general") -> dict:
    """Search knowledge base using hotel descriptions from travel-sample"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv('CB_CONN_STRING'),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv('CB_USERNAME'), 
                    password=os.getenv('CB_PASSWORD')
                )
            ),
        )
        
        search_query = """
        SELECT h.name, h.description, h.city, h.country
        FROM `travel-sample`.inventory.hotel h
        WHERE LOWER(h.description) LIKE LOWER($search_term)
        LIMIT 3
        """
        
        result = cluster.query(
            search_query,
            couchbase.options.QueryOptions(
                named_parameters={"search_term": f"%{query}%"}
            ),
        )
        
        articles = []
        for row in result:
            articles.append({
                "title": row["name"],
                "content": row["description"],
                "location": f"{row['city']}, {row['country']}",
                "category": category
            })
            
        return {
            "results": articles,
            "total_results": len(articles),
            "query": query
        }
        
    except Exception as e:
        return {"error": f"Knowledge base search failed: {e}", "results": []}


@tool
def search_policies(policy_query: str, policy_type: str = "general") -> dict:
    """Search airline policies using airline data from travel-sample"""
    try:
        cluster = couchbase.cluster.Cluster(
            os.getenv('CB_CONN_STRING'),
            couchbase.options.ClusterOptions(
                authenticator=couchbase.auth.PasswordAuthenticator(
                    username=os.getenv('CB_USERNAME'), 
                    password=os.getenv('CB_PASSWORD')
                )
            ),
        )
        
        # Search airline data for policy-like information
        search_query = """
        SELECT a.name, a.iata, a.icao, a.callsign, a.country
        FROM `travel-sample`.inventory.airline a
        WHERE LOWER(a.name) LIKE LOWER($search_term)
           OR LOWER(a.callsign) LIKE LOWER($search_term)
        LIMIT 3
        """
        
        result = cluster.query(
            search_query,
            couchbase.options.QueryOptions(
                named_parameters={"search_term": f"%{policy_query}%"}
            ),
        )
        
        policies = []
        for row in result:
            # Generate policy-like content based on airline data
            policies.append({
                "policy_title": f"{row['name']} {policy_type.title()} Policy",
                "policy_content": f"Policy information for {row['name']} ({row['iata']}) based in {row['country']}. Contact airline directly for specific {policy_type} policy details.",
                "airline_code": row.get('iata', 'N/A'),
                "airline_name": row['name'],
                "country": row['country']
            })
            
        return {
            "policies": policies,
            "total_policies": len(policies),
            "query": policy_query
        }
        
    except Exception as e:
        return {"error": f"Policy search failed: {e}", "policies": []}


class DirectCustomerSupportAgent:
    """Customer support agent with direct Couchbase integration"""
    
    def __init__(self):
        self.tools = [lookup_flight_info, search_knowledge_base, search_policies]
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the conversation graph with assistant and tools nodes"""
        workflow = StateGraph(CustomerSupportState)
        
        # Add nodes
        workflow.add_node("assistant", self._assistant_node)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges
        workflow.set_entry_point("assistant")
        workflow.add_conditional_edges(
            "assistant",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "assistant")
        
        return workflow.compile()
    
    def _assistant_node(self, state: CustomerSupportState):
        """Assistant node that can call tools"""
        
        system_prompt = """You are a professional customer support agent for TravelCorp Airlines. 

You have access to these tools:
- lookup_flight_info: Find real flight routes and information between airports
- search_knowledge_base: Search for helpful travel information  
- search_policies: Look up airline policies and procedures

When customers ask about:
- Flight information: Use lookup_flight_info with airport codes (e.g., SFO, LAX, JFK)
- General travel questions: Use search_knowledge_base
- Policies, cancellations, refunds: Use search_policies

Always be helpful, professional, and use the tools to provide accurate information."""

        system_message = SystemMessage(content=system_prompt)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        llm_with_tools = llm.bind_tools(self.tools)
        
        # Prepare messages for the LLM
        messages = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    def _should_continue(self, state: CustomerSupportState):
        """Determine if we should continue to tools or end"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "end"
    
    def run_conversation(self, customer_id: str, initial_message: str):
        """Run a customer support conversation"""
        
        initial_state = CustomerSupportState(
            messages=[HumanMessage(content=initial_message)],
            customer_id=customer_id,
            context={}
        )
        
        print(f"🎯 Direct Customer Support Agent Started for Customer: {customer_id}")
        print(f"📝 Initial Message: {initial_message}")
        print("-" * 60)
        
        for step in self.graph.stream(initial_state):
            for node, output in step.items():
                if node == "assistant":
                    message = output["messages"][-1]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print(f"🤖 Assistant: {message.content}")
                        print(f"🔧 Tools called: {[tc['name'] for tc in message.tool_calls]}")
                    else:
                        print(f"🤖 Assistant: {message.content}")
                elif node == "tools":
                    print(f"🔧 Tool results received")
        
        print("-" * 60)
        print("✅ Conversation completed")


def main():
    """Main function"""
    dotenv.load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please provide your OPENAI_API_KEY: ")
    
    print("🚀 Direct Customer Support Agent with Real Couchbase Data")
    print("=" * 60)
    
    agent = DirectCustomerSupportAgent()
    
    test_scenarios = [
        {
            "customer_id": "CUST_001",
            "message": "Hi, I need help finding flights from SFO to LAX for next week.",
        },
        {
            "customer_id": "CUST_002", 
            "message": "What are the cancellation policies for TravelCorp Airlines?",
        },
        {
            "customer_id": "CUST_003",
            "message": "I'm looking for information about luxury travel accommodations.",
        },
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎭 Scenario {i}: Customer Support Demo")
        print("=" * 40)
        
        agent.run_conversation(
            customer_id=scenario["customer_id"], 
            initial_message=scenario["message"]
        )
        
        if i < len(test_scenarios):
            print("\nMoving to next scenario...") 


if __name__ == "__main__":
    main()