"""
Customer Support Graph Module

Defines the LangGraph workflow for customer support conversations
using Agent Catalog tools and prompts.
"""

import agentc_langgraph
import dotenv
import langgraph.graph

# Import with error handling for different execution contexts
try:
    from .node import CustomerSupportAgent, CustomerSupportState
except ImportError:
    from node import CustomerSupportAgent, CustomerSupportState

# Load environment variables
dotenv.load_dotenv()


class CustomerSupportGraph(agentc_langgraph.graph.GraphRunnable):
    """Customer support conversation graph using Agent Catalog."""

    @staticmethod
    def build_starting_state(customer_id: str, initial_message: str) -> CustomerSupportState:
        """Build the initial state for the conversation."""
        return CustomerSupportState(
            messages=[],
            customer_id=customer_id,
            initial_message=initial_message,
            resolved=False,
            tool_results=[],
            interaction_history=[],
            previous_node=None,
        )

    def compile(self) -> langgraph.graph.graph.CompiledGraph:
        """Compile the LangGraph workflow."""
        
        try:
            # Build the customer support agent with catalog integration
            support_agent = CustomerSupportAgent(catalog=self.catalog, span=self.span)

            # Create a simple workflow graph for customer support
            workflow = langgraph.graph.StateGraph(CustomerSupportState)

            # Add the customer support agent node
            workflow.add_node("customer_support", support_agent)

            # Set entry point and simple flow
            workflow.set_entry_point("customer_support")
            workflow.add_edge("customer_support", langgraph.graph.END)

            return workflow.compile()
            
        except Exception as e:
            print(f"❌ Error compiling customer support graph: {e}")
            print("💡 This may indicate missing dependencies or configuration issues")
            raise
