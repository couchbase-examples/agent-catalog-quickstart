"""
Enhanced Customer Support Agent with Vector Search and Agent Catalog

This example demonstrates a sophisticated customer support system that combines:
- Vector search for knowledge base retrieval
- Multi-tool agent architecture 
- Agent Catalog prompts and tools
- Capella AI Services integration

Based on the LangGraph customer support tutorial but enhanced with Agent Catalog capabilities.
"""
import os
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import agentc
from agentc_langgraph import AgentCatalogToolNode, AgentCatalogAgent


class CustomerSupportState(TypedDict):
    """State for the customer support conversation"""
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    context: Dict[str, Any]
    current_intent: str
    resolved: bool


class CustomerSupportAgent:
    """
    Enhanced customer support agent with vector search capabilities.
    
    Features:
    - Knowledge base semantic search using Couchbase Vector Search
    - Policy lookup and flight information retrieval
    - Multi-step conversation handling
    - Context-aware responses using Agent Catalog prompts
    """
    
    def __init__(self, catalog: agentc.Catalog):
        self.catalog = catalog
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the conversation graph with assistant and tools nodes"""
        
        # Define the graph
        workflow = StateGraph(CustomerSupportState)
        
        # Create agent with Agent Catalog integration
        agent = AgentCatalogAgent(
            catalog=self.catalog,
            prompt_name="customer_support_assistant",
            model_name="capella-claude-sonnet",  # Using Capella model
            temperature=0.1
        )
        
        # Create tool node with Agent Catalog tools
        tool_node = AgentCatalogToolNode(
            catalog=self.catalog,
            tool_names=[
                "search_knowledge_base",
                "lookup_flight_info", 
                "search_policies",
                "update_customer_context"
            ]
        )
        
        # Add nodes
        workflow.add_node("assistant", self._assistant_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.set_entry_point("assistant")
        workflow.add_conditional_edges(
            "assistant",
            self._should_continue,
            {"continue": "tools", "end": END}
        )
        workflow.add_edge("tools", "assistant")
        
        return workflow.compile()
    
    def _assistant_node(self, state: CustomerSupportState):
        """Assistant node that processes user input and decides on actions"""
        
        # Get the customer support prompt from Agent Catalog
        prompt = self.catalog.find_prompt("customer_support_assistant")
        
        # Prepare context for the assistant
        context = {
            "customer_id": state.get("customer_id", ""),
            "conversation_history": state["messages"][-5:],  # Last 5 messages
            "current_context": state.get("context", {}),
            "resolved": state.get("resolved", False)
        }
        
        # Use Agent Catalog agent to generate response
        agent = AgentCatalogAgent(
            catalog=self.catalog,
            prompt_name="customer_support_assistant",
            model_name="capella-claude-sonnet"
        )
        
        response = agent.invoke({
            "messages": state["messages"],
            "context": context
        })
        
        return {"messages": [response]}
    
    def _should_continue(self, state: CustomerSupportState):
        """Determine if conversation should continue or end"""
        last_message = state["messages"][-1]
        
        # Check if assistant wants to use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Check if issue is resolved
        if state.get("resolved", False):
            return "end"
            
        return "end"
    
    def run_conversation(self, customer_id: str, initial_message: str):
        """Run a customer support conversation"""
        
        initial_state = CustomerSupportState(
            messages=[HumanMessage(content=initial_message)],
            customer_id=customer_id,
            context={},
            current_intent="",
            resolved=False
        )
        
        print(f"🎯 Customer Support Agent Started for Customer: {customer_id}")
        print(f"📝 Initial Message: {initial_message}")
        print("-" * 60)
        
        # Run the conversation
        for step in self.graph.stream(initial_state):
            for node, output in step.items():
                if node == "assistant":
                    message = output["messages"][-1]
                    print(f"🤖 Assistant: {message.content}")
                elif node == "tools":
                    print(f"🔧 Tools executed: {len(output.get('messages', []))} results")
                    
        print("-" * 60)
        print("✅ Conversation completed")


def create_sample_tools_and_prompts(catalog: agentc.Catalog):
    """Create sample tools and prompts for the customer support agent"""
    
    # This would typically be done through agentc CLI, but showing structure here
    print("📋 Setting up Agent Catalog tools and prompts...")
    
    # Sample tools that would be created:
    # 1. search_knowledge_base.yaml (semantic search)
    # 2. lookup_flight_info.sqlpp (SQL++ query) 
    # 3. search_policies.yaml (semantic search)
    # 4. update_customer_context.py (Python function)
    
    # Sample prompts:
    # 1. customer_support_assistant.yaml (main assistant prompt)
    
    print("🔧 Tools and prompts configured in Agent Catalog")


def main():
    """Main function to run the customer support agent demo"""
    
    print("🚀 Enhanced Customer Support Agent with Vector Search")
    print("=" * 60)
    
    # Initialize Agent Catalog with Capella connection
    catalog = agentc.Catalog(
        # Configuration would typically come from .env or config
        bucket=os.getenv("AGENT_CATALOG_BUCKET", "customer-support"),
        scope=os.getenv("AGENT_CATALOG_SCOPE", "support"), 
        collection=os.getenv("AGENT_CATALOG_COLLECTION", "agents")
    )
    
    # Set up sample tools and prompts
    create_sample_tools_and_prompts(catalog)
    
    # Create the customer support agent
    agent = CustomerSupportAgent(catalog)
    
    # Sample conversations
    test_scenarios = [
        {
            "customer_id": "CUST_001", 
            "message": "Hi, I need help finding information about flight cancellation policies for my upcoming trip to Paris."
        },
        {
            "customer_id": "CUST_002",
            "message": "I'm looking for flights from SFO to JFK next week. Can you help me find options and pricing?"
        },
        {
            "customer_id": "CUST_003", 
            "message": "My flight was delayed and I missed my connection. What are my options for rebooking?"
        }
    ]
    
    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎭 Scenario {i}: Customer Support Demo")
        print("=" * 40)
        
        agent.run_conversation(
            customer_id=scenario["customer_id"],
            initial_message=scenario["message"]
        )
        
        if i < len(test_scenarios):
            input("\nPress Enter to continue to next scenario...")


if __name__ == "__main__":
    main()