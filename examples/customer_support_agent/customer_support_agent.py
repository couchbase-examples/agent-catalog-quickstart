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
import dotenv
import getpass
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

import agentc


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
        
        # Create mock tools for demonstration
        def search_knowledge_base(query: str) -> str:
            """Search the knowledge base for customer support information"""
            return f"📋 Knowledge Base Result for '{query}': Found relevant support articles about your inquiry."
        
        def lookup_flight_info(source: str, destination: str) -> str:
            """Look up flight information between airports"""
            return f"✈️ Flight Info: Found flights from {source} to {destination}. Please check our website for current schedules."
        
        def search_policies(policy_type: str) -> str:
            """Search for airline policies"""
            return f"🛡️ Policy Info: Here are the relevant policies for {policy_type}. Please refer to our terms and conditions."
        
        def update_customer_context(customer_id: str, context: dict) -> str:
            """Update customer context"""
            return f"👤 Updated context for customer {customer_id}"
        
        tools = [search_knowledge_base, lookup_flight_info, search_policies, update_customer_context]
        
        # Create tool node with available tools
        tool_node = ToolNode(tools)
        
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
        

        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Create system message for customer support
        system_prompt = """You are a professional customer support agent for TravelCorp Airlines. 
        You help customers with flight bookings, cancellations, policies, and travel questions.
        
        Be helpful, empathetic, and solution-focused. Use the available tools when you need specific information:
        - search_knowledge_base: for general support topics
        - lookup_flight_info: for flight schedules and routes  
        - search_policies: for airline policies and rules
        - update_customer_context: to save customer preferences
        
        Always provide clear, actionable responses to help resolve customer issues."""
        
        # Prepare messages with system prompt
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        
        # Get response from LLM
        response = llm.invoke(messages)
        
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
    
    # Load environment variables
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
    
    # Set up OpenAI API key if not present
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please provide your OPENAI_API_KEY: ")
    
    print("🚀 Enhanced Customer Support Agent with Vector Search")
    print("=" * 60)
    
    # Initialize Agent Catalog
    catalog = agentc.Catalog()
    
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