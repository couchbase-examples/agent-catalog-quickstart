#!/usr/bin/env python3
"""
Flight Search Agent - Agent Catalog + LangGraph Implementation

A streamlined flight search agent demonstrating Agent Catalog integration
with LangGraph for flight booking assistance.
"""

import typing
import os

import agentc
import agentc_langgraph.agent
import agentc_langgraph.graph
import dotenv
import langgraph.graph
import langchain_core.messages
import langchain_core.runnables
import langchain_openai.chat_models

# Load environment variables
dotenv.load_dotenv()


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations."""
    
    customer_id: str
    query: str
    resolved: bool
    search_results: typing.List[typing.Dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and prompts."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the flight search agent."""
        
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

        super().__init__(
            chat_model=chat_model,
            catalog=catalog,
            span=span,
            prompt_name="flight_search_assistant",
        )

    def _invoke(
        self,
        span: agentc.Span,
        state: FlightSearchState,
        config: langchain_core.runnables.RunnableConfig,
    ) -> FlightSearchState:
        """Handle flight search conversation with Agent Catalog tools."""

        # Initialize conversation if this is the first message
        if not state["messages"]:
            initial_msg = langchain_core.messages.HumanMessage(content=state["query"])
            state["messages"].append(initial_msg)
            print(f"🔍 Flight Query: {state['query']}")

        # Create the ReAct agent with tools from Agent Catalog
        agent = self.create_react_agent(span)

        # Run the agent with the current state
        response = agent.invoke(input=state, config=config)

        # Extract the assistant's response
        if response.get("messages"):
            assistant_message = response["messages"][-1]
            state["messages"].append(assistant_message)

            if hasattr(assistant_message, "content"):
                print(f"✈️ Response: {assistant_message.content}")

        # Check if search is complete
        if response.get("structured_response"):
            structured = response["structured_response"]
            state["resolved"] = structured.get("search_complete", True)
            
            # Store search results
            if "flight_results" in structured:
                state["search_results"] = structured["flight_results"]

        return state


class FlightSearchGraph(agentc_langgraph.graph.GraphRunnable):
    """Flight search conversation graph using Agent Catalog."""

    @staticmethod
    def build_starting_state(customer_id: str, query: str) -> FlightSearchState:
        """Build the initial state for the flight search."""
        return FlightSearchState(
            messages=[],
            customer_id=customer_id,
            query=query,
            resolved=False,
            search_results=[],
            previous_node=None,
        )

    def compile(self) -> langgraph.graph.graph.CompiledGraph:
        """Compile the LangGraph workflow."""
        
        # Build the flight search agent with catalog integration
        search_agent = FlightSearchAgent(catalog=self.catalog, span=self.span)

        # Create a simple workflow graph for flight search
        workflow = langgraph.graph.StateGraph(FlightSearchState)

        # Add the flight search agent node
        workflow.add_node("flight_search", search_agent)

        # Set entry point and simple flow
        workflow.set_entry_point("flight_search")
        workflow.add_edge("flight_search", langgraph.graph.END)

        return workflow.compile()


def run_flight_search_demo():
    """Run an interactive flight search demo."""
    
    print("\n🛫 Flight Search Agent - Agent Catalog Demo")
    print("=" * 50)
    
    try:
        # Initialize Agent Catalog
        catalog = agentc.Catalog()
        application_span = catalog.Span(name="Flight Search Agent")
        
        # Create the flight search graph
        flight_graph = FlightSearchGraph(catalog=catalog, span=application_span)
        
        # Compile the graph
        compiled_graph = flight_graph.compile()
        
        print("✅ Agent Catalog integration successful")
        
        # Interactive flight search loop
        while True:
            print("\n" + "─" * 40)
            query = input("🔍 Enter flight search query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("✈️ Thanks for using Flight Search Agent!")
                break
                
            if not query:
                continue
                
            try:
                # Build starting state
                state = FlightSearchGraph.build_starting_state(
                    customer_id="demo_user",
                    query=query
                )
                
                # Run the flight search
                result = compiled_graph.invoke(state)
                
                # Display results summary
                if result.get("search_results"):
                    print(f"\n📋 Found {len(result['search_results'])} flight options")
                
                print(f"✅ Search completed: {result.get('resolved', False)}")
                
            except Exception as e:
                print(f"❌ Search error: {e}")
                
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        print("💡 Ensure Agent Catalog is published: agentc index . && agentc publish")


if __name__ == "__main__":
    run_flight_search_demo()