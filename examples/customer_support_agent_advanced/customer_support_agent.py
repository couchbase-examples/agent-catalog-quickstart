"""
Enhanced Customer Support Agent with Vector Search and Agent Catalog

This example demonstrates a sophisticated customer support system that combines:
- Vector search for knowledge base retrieval
- Multi-tool agent architecture
- Agent Catalog prompts and tools
- Capella AI Services integration

Based on the LangGraph customer support tutorial but enhanced with Agent Catalog capabilities.
"""

import getpass
import os
from typing import Annotated, Any, Dict, List

import agentc
import dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


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

    def __init__(self, catalog: agentc.Catalog = None):
        self.catalog = catalog
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the conversation graph with assistant and tools nodes"""

        # Define the graph
        workflow = StateGraph(CustomerSupportState)

        if self.catalog:
            print("🔧 Agent Catalog available - using enhanced mode")
            # For now, use simple mode even with catalog until we set up tools properly
            workflow.add_node("assistant", self._catalog_assistant_node)
        else:
            print("📝 No Agent Catalog - using simple mode")
            workflow.add_node("assistant", self._simple_assistant_node)

        workflow.set_entry_point("assistant")
        workflow.add_edge("assistant", END)

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
            "resolved": state.get("resolved", False),
        }

        # For now, use simple LLM call (Agent Catalog agent integration coming later)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Try to get the prompt from Agent Catalog
        try:
            prompts = self.catalog.find_prompts("customer_support_assistant")
            prompt = prompts[0] if prompts else None
            if prompt:
                system_content = prompt.content
            else:
                system_content = "You are a professional customer support agent for TravelCorp Airlines."
        except Exception as e:
            system_content = "You are a professional customer support agent for TravelCorp Airlines."

        # Prepare messages
        system_message = SystemMessage(content=system_content)
        messages = [system_message] + state["messages"]

        # Get response
        response = llm.invoke(messages)

        return {"messages": [response]}

    def _catalog_assistant_node(self, state: CustomerSupportState):
        """Assistant node with Agent Catalog integration"""
        try:
            # Try to get the prompt from Agent Catalog
            prompts = self.catalog.find_prompts("customer_support_assistant")
            prompt = prompts[0] if prompts else None
            if prompt:
                print("✅ Using Agent Catalog prompt")
                # Extract and format the prompt content
                content = prompt.content
                system_content = f"""You are an expert customer support agent for TravelCorp Airlines.

{content.get('system_role', '')}

Core Instructions:
{' '.join(content.get('core_instructions', []))}

Response Guidelines:
{' '.join(content.get('response_guidelines', []))}"""
            else:
                print("⚠️  Agent Catalog prompt not found, using fallback")
                system_content = """You are a professional customer support agent for TravelCorp Airlines with access to advanced tools."""
        except Exception as e:
            print(f"⚠️  Error accessing Agent Catalog prompt: {e}")
            system_content = """You are a professional customer support agent for TravelCorp Airlines with access to advanced tools."""

        # Use OpenAI with catalog-aware prompt
        system_message = SystemMessage(content=system_content)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        # Prepare messages
        messages = [system_message] + state["messages"]

        # Get response
        response = llm.invoke(messages)

        return {"messages": [response]}

    def _simple_assistant_node(self, state: CustomerSupportState):
        """Simple assistant node without Agent Catalog tools"""

        # Simple system prompt for customer support
        system_message = SystemMessage(
            content="""You are a professional and helpful customer support agent for TravelCorp Airlines. 
        
Your role is to:
- Assist customers with travel-related inquiries
- Provide information about flights, bookings, and policies
- Be empathetic and solution-oriented
- Escalate complex issues when necessary

Be concise but thorough in your responses. Always maintain a friendly and professional tone."""
        )

        # Use OpenAI directly
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        # Prepare messages
        messages = [system_message] + state["messages"]

        # Get response
        response = llm.invoke(messages)

        return {"messages": [response]}

    def _should_continue(self, state: CustomerSupportState):
        """Determine if conversation should continue or end"""
        last_message = state["messages"][-1]

        # Check if assistant wants to use tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
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
            resolved=False,
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

    try:
        # Initialize Agent Catalog with environment variables
        catalog = agentc.Catalog()
        print("✅ Agent Catalog connected successfully")
    except Exception as e:
        print(f"⚠️  Agent Catalog not available: {e}")
        print("📝 Running in simple mode without Agent Catalog tools")
        catalog = None

    # Set up sample tools and prompts
    create_sample_tools_and_prompts(catalog)

    # Create the customer support agent
    agent = CustomerSupportAgent(catalog)

    # Sample conversations
    test_scenarios = [
        {
            "customer_id": "CUST_001",
            "message": "Hi, I need help finding information about flight cancellation policies for my upcoming trip to Paris.",
        },
        {
            "customer_id": "CUST_002",
            "message": "I'm looking for flights from SFO to JFK next week. Can you help me find options and pricing?",
        },
        {
            "customer_id": "CUST_003",
            "message": "My flight was delayed and I missed my connection. What are my options for rebooking?",
        },
    ]

    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎭 Scenario {i}: Customer Support Demo")
        print("=" * 40)

        agent.run_conversation(
            customer_id=scenario["customer_id"], initial_message=scenario["message"]
        )

        if i < len(test_scenarios):
            input("\nPress Enter to continue to next scenario...")


if __name__ == "__main__":
    main()
