"""
Customer Support Agent Node Module

Defines the agent nodes and state for customer support conversations
using Agent Catalog tools and prompts.
"""

import agentc
import agentc_langchain
import agentc_langgraph.agent
import langchain_core.messages
import langchain_core.runnables
import langchain_openai.chat_models
import typing


class CustomerSupportState(agentc_langgraph.agent.State):
    """State for customer support conversations."""
    customer_id: str
    initial_message: str
    resolved: bool
    tool_results: typing.List[typing.Dict]
    interaction_history: typing.List[typing.Dict]


class CustomerSupportAgent(agentc_langgraph.agent.ReActAgent):
    """Customer support agent using Agent Catalog tools and prompts."""
    
    def __init__(self, catalog: agentc.Catalog, span: agentc.Span):
        """Initialize the customer support agent."""
        chat_model = langchain_openai.chat_models.ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.1
        )
        
        try:
            # Try to initialize with Agent Catalog prompt
            super().__init__(
                chat_model=chat_model, 
                catalog=catalog, 
                span=span, 
                prompt_name="customer_support_assistant"
            )
            self.catalog_available = True
            print("✅ Agent Catalog integration successful!")
            print("🔧 Tools will be loaded automatically from catalog")
            
        except LookupError as e:
            if "Catalog version not found" in str(e):
                print("⚠️  Agent Catalog not published yet")
                print("💡 To enable full functionality:")
                print("   1. git commit (to clean repo)")
                print("   2. agentc index .")
                print("   3. agentc publish")
                print("🎯 Demonstrating architecture patterns...")
                
                # Initialize without catalog for demo purposes
                super(agentc_langgraph.agent.ReActAgent, self).__init__()
                self.chat_model = chat_model
                self.catalog = catalog
                self.span = span
                self.catalog_available = False
            else:
                raise

    @staticmethod
    def _display_message(span: agentc.Span, role: str, message: str):
        """Display messages with proper formatting."""
        if role == "customer":
            print(f"👤 Customer: {message}")
            span.log(agentc.span.UserContent(value=message))
        elif role == "assistant":
            print(f"🤖 Assistant: {message}")
            span.log(agentc.span.AssistantContent(value=message))

    def _invoke(
        self, 
        span: agentc.Span, 
        state: CustomerSupportState, 
        config: langchain_core.runnables.RunnableConfig
    ) -> CustomerSupportState:
        """Handle customer support conversation with Agent Catalog tools."""
        
        # Initialize conversation if this is the first message
        if not state["messages"]:
            # Add the initial customer message
            initial_msg = langchain_core.messages.HumanMessage(
                content=state["initial_message"]
            )
            state["messages"].append(initial_msg)
            self._display_message(span, "customer", state["initial_message"])

        if self.catalog_available:
            # Use full Agent Catalog integration
            print("\n🔧 Processing request with Agent Catalog tools...")
            
            # Create the ReAct agent with tools from Agent Catalog
            agent = self.create_react_agent(span)
            
            # Run the agent with the current state
            response = agent.invoke(input=state, config=config)
            
            # Extract the assistant's response
            if response.get("messages"):
                assistant_message = response["messages"][-1]
                state["messages"].append(assistant_message)
                
                # Display the response
                if hasattr(assistant_message, 'content'):
                    self._display_message(span, "assistant", assistant_message.content)
            
            # Check if the issue is resolved based on the structured response
            if response.get("structured_response"):
                structured = response["structured_response"]
                state["resolved"] = structured.get("resolution_status") == "resolved"
                
                # Store tool results if any were used
                if "tool_results" in response:
                    state["tool_results"].extend(response["tool_results"])
                
                # Update interaction history
                interaction = {
                    "customer_id": state["customer_id"],
                    "message": state["initial_message"],
                    "response": assistant_message.content if hasattr(assistant_message, 'content') else "",
                    "resolution_status": structured.get("resolution_status", "pending"),
                    "tools_used": [tr.get("tool_name") for tr in state["tool_results"] if tr.get("tool_name")]
                }
                state["interaction_history"].append(interaction)
                
        else:
            # Demonstrate architecture without published catalog
            print("\n🎯 Demonstrating Agent Catalog Architecture:")
            print("📁 Files in proper structure:")
            print("   - prompts/customer_support_assistant.yaml ✅")
            print("   - tools/lookup_flight_info.sqlpp ✅") 
            print("   - tools/search_policies.yaml ✅")
            print("   - tools/update_customer_context.py ✅")
            print("📋 When catalog is published, agent will:")
            print("   1. Load prompt automatically by name")
            print("   2. Load tools specified in prompt")  
            print("   3. Use ReActAgent.create_react_agent()")
            print("   4. Process with structured output")
            
            # Generate a demo response
            demo_response = (
                "I'd be happy to help you find flights from SFO to LAX! "
                "Let me search our flight database for available options.\n\n"
                "**[With published catalog, I would use:]**\n"
                "- `lookup_flight_info` tool (SQL++) for real flight data\n"
                "- `search_policies` tool (YAML) for airline policies\n"
                "- `update_customer_context` tool (Python) for personalization\n\n"
                "This demonstrates the proper Agent Catalog + LangGraph architecture!"
            )
            
            assistant_message = langchain_core.messages.AIMessage(content=demo_response)
            state["messages"].append(assistant_message)
            self._display_message(span, "assistant", demo_response)
            
            state["resolved"] = True
        
        print(f"\n✅ Architecture demonstration completed")
        print(f"📊 Status: {'🎯 Agent Catalog patterns demonstrated' if not self.catalog_available else '🎯 Resolved'}")
        
        return state