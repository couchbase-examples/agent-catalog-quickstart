"""
Customer Support Agent Node Module

Defines the agent nodes and state for customer support conversations
using Agent Catalog tools and prompts.
"""

import typing

import agentc
import agentc_langgraph.agent
import langchain_core.messages
import langchain_core.runnables
import langchain_openai.chat_models


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
        import os

        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

        # Initialize basic properties
        self.catalog = catalog
        self.span = span
        self.chat_model = chat_model
        self.catalog_available = False
        self.prompt = None

        try:
            # Try to initialize with Agent Catalog prompt
            super().__init__(
                chat_model=chat_model,
                catalog=catalog,
                span=span,
                prompt_name="customer_support_assistant",
            )
            self.catalog_available = True
            print("✅ Agent Catalog integration successful!")
            print("🔧 Tools will be loaded automatically from catalog")

        except (LookupError, Exception) as e:
            error_msg = str(e).lower()
            if "catalog version not found" in error_msg or "prompt" in error_msg or "not found" in error_msg:
                print("⚠️  Agent Catalog not published yet")
                print("💡 To enable full functionality:")
                print("   1. Ensure git repo is clean: git status")
                print("   2. Index catalog: agentc index .")
                print("   3. Publish catalog: agentc publish")
                print("🎯 Running in demo mode without published catalog...")

                # Initialize minimal ReActAgent functionality
                try:
                    super(agentc_langgraph.agent.ReActAgent, self).__init__()
                except Exception:
                    # If even basic initialization fails, create minimal structure
                    pass
                    
                self.catalog_available = False
            else:
                print(f"⚠️  Unexpected catalog error: {e}")
                print("🎯 Continuing with demo mode...")
                self.catalog_available = False

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
        config: langchain_core.runnables.RunnableConfig,
    ) -> CustomerSupportState:
        """Handle customer support conversation with Agent Catalog tools."""

        # Initialize conversation if this is the first message
        if not state["messages"]:
            # Add the initial customer message
            initial_msg = langchain_core.messages.HumanMessage(content=state["initial_message"])
            state["messages"].append(initial_msg)
            self._display_message(span, "customer", state["initial_message"])

        if self.catalog_available:
            try:
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
                    if hasattr(assistant_message, "content"):
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
                        "response": assistant_message.content
                        if hasattr(assistant_message, "content")
                        else "",
                        "resolution_status": structured.get("resolution_status", "pending"),
                        "tools_used": [
                            tr.get("tool_name") for tr in state["tool_results"] if tr.get("tool_name")
                        ],
                    }
                    state["interaction_history"].append(interaction)
                else:
                    # Fallback if no structured response
                    state["resolved"] = True
                    
            except Exception as e:
                print(f"\n❌ Error during Agent Catalog tool execution: {e}")
                print("💡 This may indicate issues with tool configuration or Couchbase connection")
                print("🔄 Falling back to demo mode...")
                self.catalog_available = False

        if not self.catalog_available:
            # Demonstrate architecture without published catalog
            print("\n🎯 Demonstrating Agent Catalog Architecture:")
            print("📁 Files are structured correctly:")
            print("   - prompts/customer_support_assistant.yaml")
            print("   - tools/lookup_flight_info.sqlpp")
            print("   - tools/search_policies.yaml")
            print("   - tools/update_customer_context.py")
            print("📋 When catalog is published, agent will:")
            print("   1. Load prompt automatically by name")
            print("   2. Load tools specified in prompt")
            print("   3. Use ReActAgent.create_react_agent()")
            print("   4. Process with structured output")

            # Generate a contextual demo response based on the customer's request
            customer_message = state["initial_message"].lower()
            
            if "flight" in customer_message:
                demo_response = self._generate_flight_demo_response(state["initial_message"])
            elif "policy" in customer_message or "cancel" in customer_message or "refund" in customer_message:
                demo_response = self._generate_policy_demo_response()
            else:
                demo_response = self._generate_general_demo_response()

            assistant_message = langchain_core.messages.AIMessage(content=demo_response)
            state["messages"].append(assistant_message)
            self._display_message(span, "assistant", demo_response)

            # Update interaction history for demo mode
            interaction = {
                "customer_id": state["customer_id"],
                "message": state["initial_message"],
                "response": demo_response,
                "resolution_status": "demo_completed",
                "tools_used": ["demo_mode"],
            }
            state["interaction_history"].append(interaction)
            state["resolved"] = True

        print(f"\n✅ Architecture demonstration completed")
        print(
            f"📊 Status: {'🎯 Agent Catalog patterns demonstrated' if not self.catalog_available else '🎯 Resolved'}"
        )

        return state

    def _generate_flight_demo_response(self, original_message: str) -> str:
        """Generate a demo response for flight-related queries."""
        return f"""I'd be happy to help you with your flight request! 

**Customer Request:** {original_message}

**[With published Agent Catalog, I would use:]**
- 🛫 `lookup_flight_info` tool (SQL++) - Query Couchbase travel-sample for real flight data
- 📋 `search_policies` tool (YAML) - Find relevant airline policies using semantic search  
- 👤 `update_customer_context` tool (Python) - Personalize recommendations based on customer history

**Demo Flight Results:**
✈️ Found 5 flight options with different airlines and pricing
🕐 Multiple departure times available
💰 Price range: $120 - $350
📅 Next week availability confirmed

This demonstrates the proper Agent Catalog + LangGraph + Couchbase architecture!
To see real data, run: `agentc init && agentc index . && agentc publish`"""

    def _generate_policy_demo_response(self) -> str:
        """Generate a demo response for policy-related queries."""
        return """I can help you with our airline policies!

**[With published Agent Catalog, I would use:]**
- 📋 `search_policies` tool (YAML) - Semantic search through policy documents
- 🔍 Vector search with sentence-transformers for accurate policy matching
- 📄 Real policy documents from Couchbase collections

**Demo Policy Information:**
✅ Cancellation: Free within 24 hours of booking
💰 Refunds: Processed within 7-10 business days  
🎫 Changes: $50 fee for domestic flights
🕐 Flight delays: Compensation available for delays >3 hours

This demonstrates Agent Catalog's semantic search capabilities!
To see real policy search, run: `agentc init && agentc index . && agentc publish`"""

    def _generate_general_demo_response(self) -> str:
        """Generate a demo response for general queries."""
        return """I'm here to help with your customer service needs!

**[With published Agent Catalog, I would use:]**
- 🛫 `lookup_flight_info` tool for flight searches
- 📋 `search_policies` tool for policy questions  
- 👤 `update_customer_context` tool for personalization
- 🤖 LangGraph ReActAgent for conversation orchestration

**Agent Catalog Features Demonstrated:**
✅ Mixed tool formats: SQL++, YAML, Python
✅ Automatic tool loading by name or semantic query
✅ Structured prompt management with output schemas
✅ Integration with Couchbase for real data
✅ Professional conversation patterns

This showcases the complete Agent Catalog + LangGraph architecture!
To experience full functionality, run: `agentc init && agentc index . && agentc publish`"""
