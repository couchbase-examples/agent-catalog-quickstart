"""
Enhanced Customer Support Agent with Agent Catalog and LangGraph

This example demonstrates a sophisticated customer support system following the exact pattern
of the working agent-catalog/examples/with_langgraph example.
"""

import getpass
import os

import agentc
import agentc_langgraph
import dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


# Load environment variables
dotenv.load_dotenv()


class CustomerSupportState(agentc_langgraph.agent.State):
    """State for the customer support conversation"""
    customer_id: str
    resolved: bool


class CustomerSupportAgent(agentc_langgraph.agent.ReActAgent):
    """Customer support agent using Agent Catalog"""
    
    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        super().__init__(chat_model=chat_model, catalog=catalog, span=span, prompt_name="customer_support_assistant")
        
    def _invoke(self, span: agentc.Span, state: CustomerSupportState, config) -> CustomerSupportState:
        """Handle customer support conversation with tools"""
        
        # Create the ReAct agent with tools
        agent = self.create_react_agent(span)
        
        # Run the agent 
        response = agent.invoke(input=state, config=config)
        
        # Update state with response
        state["messages"].append(response["messages"][-1])
        
        # Check if resolved based on response
        if response.get("structured_response"):
            structured = response["structured_response"]
            state["resolved"] = structured.get("resolution_status") == "resolved"
        
        return state


class CustomerSupportGraph(agentc_langgraph.graph.GraphRunnable):
    """Customer support conversation graph"""
    
    @staticmethod
    def build_starting_state(customer_id: str, initial_message: str) -> CustomerSupportState:
        return CustomerSupportState(
            messages=[HumanMessage(content=initial_message)],
            customer_id=customer_id,
            resolved=False,
            previous_node=None
        )
    
    def compile(self):
        """Compile the graph"""
        import langgraph.graph
        
        # Create the customer support agent
        support_agent = CustomerSupportAgent(
            catalog=self.catalog,
            span=self.span
        )
        
        # Build the workflow
        workflow = langgraph.graph.StateGraph(CustomerSupportState)
        workflow.add_node("support_agent", support_agent)
        workflow.set_entry_point("support_agent")
        workflow.add_edge("support_agent", langgraph.graph.END)
        
        return workflow.compile()


def main():
    """Main function to run the customer support agent"""
    
    # Set up OpenAI API key if not present
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Please provide your OPENAI_API_KEY: ")

    print("🚀 Enhanced Customer Support Agent with Agent Catalog")
    print("=" * 60)

    try:
        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        print("✅ Agent Catalog connected successfully")
        
        # Check available tools
        print("🔧 Available tools:")
        for tool_name in ["lookup_flight_info", "search_knowledge_base", "search_policies", "update_customer_context", "get_customer_insights"]:
            try:
                tools = catalog.find_tools([tool_name])
                if tools:
                    print(f"  ✅ {tool_name}")
                else:
                    print(f"  ❌ {tool_name}: Not found")
            except Exception as e:
                print(f"  ⚠️  {tool_name}: Error - {e}")
        
        # Check prompt
        try:
            prompts = catalog.find_prompts("customer_support_assistant")
            print(f"📝 Found {len(prompts)} customer support prompts")
        except Exception as e:
            print(f"⚠️  Prompt error: {e}")
        
    except Exception as e:
        print(f"❌ Agent Catalog connection failed: {e}")
        return

    # Test scenarios
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

    # Run test scenarios
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🎭 Scenario {i}: Customer Support Demo")
        print("=" * 40)
        
        # Create the graph
        graph = CustomerSupportGraph(catalog=catalog)
        
        # Build starting state
        initial_state = CustomerSupportGraph.build_starting_state(
            customer_id=scenario["customer_id"],
            initial_message=scenario["message"]
        )
        
        print(f"🎯 Customer: {scenario['customer_id']}")
        print(f"📝 Message: {scenario['message']}")
        print("-" * 40)
        
        # Run the conversation
        try:
            result = graph.invoke(input=initial_state)
            
            # Display the conversation
            for msg in result["messages"]:
                if hasattr(msg, 'type'):
                    if msg.type == 'human':
                        print(f"👤 Customer: {msg.content}")
                    elif msg.type == 'ai':
                        print(f"🤖 Assistant: {msg.content}")
                        # Check for tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            print(f"🔧 Tools used: {[tc['name'] for tc in msg.tool_calls]}")
                    elif msg.type == 'tool':
                        print(f"🔧 Tool result: {msg.name}")
                        
        except Exception as e:
            print(f"❌ Error in conversation: {e}")
        
        print("-" * 40)
        print("✅ Scenario completed")
        
        if i < len(test_scenarios):
            print("\nMoving to next scenario...")


if __name__ == "__main__":
    main()