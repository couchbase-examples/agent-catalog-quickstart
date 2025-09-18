#!/usr/bin/env python3
"""
Flight Search Agent - Agent Catalog + LangGraph Implementation

A streamlined flight search agent demonstrating Agent Catalog integration
with LangGraph and Couchbase vector search for flight booking assistance.
"""

import json
import logging
import os
import sys
from datetime import timedelta

import agentc
import agentc_langgraph.agent
import agentc_langgraph.graph
import dotenv
import langchain_core.messages
import langchain_core.runnables
import langchain_openai.chat_models
import langgraph.graph
import langgraph.prebuilt
from couchbase.auth import PasswordAuthenticator
from couchbase.cluster import Cluster
from couchbase.exceptions import KeyspaceNotFoundException
from couchbase.options import ClusterOptions
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from pydantic import SecretStr


# Import shared modules using robust project root discovery
def find_project_root():
    """Find the project root by looking for the shared directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Look for the shared directory as the definitive marker
        shared_path = os.path.join(current, "shared")
        if os.path.exists(shared_path) and os.path.isdir(shared_path):
            return current
        current = os.path.dirname(current)
    return None


# Add project root to Python path
project_root = find_project_root()
if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

from shared.agent_setup import setup_ai_services, setup_environment, test_capella_connectivity
from shared.couchbase_client import create_couchbase_client

# Setup logging with essential level only
# Note: Agent Catalog does not integrate with Python logging
# The main application span will generate meaningful logs
# These logs can be queried from the Agent Catalog bucket (see query_agent_catalog_logs)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress verbose logging from external libraries
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("agentc_core").setLevel(logging.WARNING)

# Load environment variables
dotenv.load_dotenv(override=True)


class CapellaCompatibleToolNode(agentc_langgraph.tool.ToolNode):
    """Custom ToolNode that ensures Capella AI compatibility by handling tool arguments properly."""

    def __init__(self, span: agentc.Span, catalog: agentc.Catalog, *args, **kwargs):
        self.catalog = catalog

        # Get Agent Catalog tools and convert them with proper argument handling
        tool_names = [
            "lookup_flight_info",
            "save_flight_booking",
            "retrieve_flight_bookings",
            "search_airline_reviews",
        ]

        tools = []
        for tool_name in tool_names:
            try:
                catalog_tool = self.catalog.find("tool", name=tool_name)
                if catalog_tool:
                    # Create LangChain tool with Agent Catalog tool's metadata
                    wrapper_func = self._create_capella_compatible_wrapper(catalog_tool, tool_name)
                    # Set the function name and description for proper tool registration
                    wrapper_func.__name__ = tool_name
                    wrapper_func.__doc__ = getattr(catalog_tool, 'description', f"Tool for {tool_name.replace('_', ' ')}")

                    langchain_tool = langchain_core.tools.tool(wrapper_func)
                    tools.append(langchain_tool)
                    logger.info(f"✅ Added Capella-compatible tool: {tool_name}")
            except Exception as e:
                logger.error(f"❌ Failed to add tool {tool_name}: {e}")

        super().__init__(span, tools=tools, *args, **kwargs)

    def _create_capella_compatible_wrapper(self, catalog_tool, tool_name):
        """Create a wrapper that handles Capella AI argument parsing."""

        def wrapper_func(tool_input):
            """Wrapper that handles various input formats for Capella AI compatibility."""
            try:
                logger.info(f"🔧 Tool {tool_name} called with input: {repr(tool_input)}")

                # Handle different input types that Capella AI might send
                if isinstance(tool_input, dict):
                    # Direct dictionary input
                    input_str = str(tool_input)
                elif isinstance(tool_input, str):
                    input_str = tool_input.strip()
                else:
                    input_str = str(tool_input)

                # Clean up the input string
                clean_input = self._clean_tool_input(input_str)

                # Call the appropriate tool with proper parameters
                result = self._call_catalog_tool(catalog_tool, tool_name, clean_input)

                logger.info(f"✅ Tool {tool_name} executed successfully")
                return str(result) if result is not None else "No results found"

            except Exception as e:
                error_msg = f"Error in tool {tool_name}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                return error_msg

        return wrapper_func

    def _clean_tool_input(self, tool_input: str) -> str:
        """Clean and normalize tool input for consistent processing."""
        # Remove ReAct format artifacts
        artifacts_to_remove = [
            '\nObservation', 'Observation', '\nThought:', 'Thought:',
            '\nAction:', 'Action:', '\nAction Input:', 'Action Input:',
            '\nFinal Answer:', 'Final Answer:'
        ]

        clean_input = tool_input
        for artifact in artifacts_to_remove:
            if artifact in clean_input:
                clean_input = clean_input.split(artifact)[0]

        # Clean up quotes and whitespace
        clean_input = clean_input.strip().strip("\"'").strip()
        # Normalize whitespace
        clean_input = " ".join(clean_input.split())

        return clean_input

    def _call_catalog_tool(self, catalog_tool, tool_name: str, clean_input: str):
        """Call the Agent Catalog tool with appropriate parameter mapping."""

        if tool_name == "lookup_flight_info":
            return self._handle_lookup_flight_info(catalog_tool, clean_input)
        elif tool_name == "save_flight_booking":
            return catalog_tool.func(booking_input=clean_input)
        elif tool_name == "retrieve_flight_bookings":
            # Handle empty input for "all bookings"
            if not clean_input or clean_input.lower() in ["", "all", "none"]:
                return catalog_tool.func(booking_query="")
            else:
                return catalog_tool.func(booking_query=clean_input)
        elif tool_name == "search_airline_reviews":
            if not clean_input:
                return "Error: Please provide a search query for airline reviews"
            return catalog_tool.func(query=clean_input)
        else:
            # Generic fallback
            return catalog_tool.func(clean_input)

    def _handle_lookup_flight_info(self, catalog_tool, tool_input: str):
        """Handle lookup_flight_info with flexible airport code parsing."""
        import re

        source = None
        dest = None

        # 1) Support key=value style inputs (e.g., source_airport="JFK", destination_airport="LAX")
        try:
            m_src = re.search(r"source_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
            m_dst = re.search(r"destination_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
            if m_src and m_dst:
                source = m_src.group(1).upper()
                dest = m_dst.group(1).upper()
        except Exception:
            pass

        # 2) Fallback: comma separated codes (e.g., "JFK,LAX")
        if source is None or dest is None:
            if ',' in tool_input:
                parts = tool_input.split(',')
                if len(parts) >= 2:
                    source = parts[0].strip().upper()
                    dest = parts[1].strip().upper()

        # 3) Fallback: natural language (e.g., "JFK to LAX")
        if source is None or dest is None:
            words = tool_input.upper().split()
            airport_codes = [w for w in words if len(w) == 3 and w.isalpha()]
            if len(airport_codes) >= 2:
                source, dest = airport_codes[0], airport_codes[1]

        if not source or not dest:
            return "Error: Please provide source and destination airports (e.g., JFK,LAX or JFK to LAX)"

        return catalog_tool.func(source_airport=source, destination_airport=dest)


class CapellaCompatibleChatModel(langchain_core.runnables.Runnable):
    """Wrapper for chat models that disables function calling for Capella AI compatibility."""

    def __init__(self, chat_model):
        super().__init__()
        self.chat_model = chat_model

    def bind_tools(self, *args, **kwargs):
        """Disabled bind_tools to force traditional ReAct format."""
        return self

    def invoke(self, input, config=None, **kwargs):
        """Delegate invoke to the original model."""
        return self.chat_model.invoke(input, config, **kwargs)

    def generate(self, *args, **kwargs):
        """Delegate generate to the original model."""
        return self.chat_model.generate(*args, **kwargs)

    def __getattr__(self, name):
        """Delegate any missing attributes to the original model."""
        return getattr(self.chat_model, name)


class FlightSearchState(agentc_langgraph.agent.State):
    """State for flight search conversations - single user system."""

    query: str
    resolved: bool
    search_results: list[dict]


class FlightSearchAgent(agentc_langgraph.agent.ReActAgent):
    """Flight search agent using Agent Catalog tools and ReActAgent framework."""

    def __init__(self, catalog: agentc.Catalog, span: agentc.Span, chat_model=None):
        """Initialize the flight search agent."""

        if chat_model is None:
            # Fallback to OpenAI if no chat model provided
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            chat_model = langchain_openai.chat_models.ChatOpenAI(model=model_name, temperature=0.1)

        # Wrap the chat model to disable function calling for Capella AI compatibility
        chat_model = CapellaCompatibleChatModel(chat_model)
        logger.info("Wrapped chat model to disable function calling for Capella AI compatibility")

        super().__init__(
            chat_model=chat_model, catalog=catalog, span=span, prompt_name="flight_search_assistant"
        )

    def create_react_agent(self, span: agentc.Span, tools=None, **kwargs):
        """Override to use traditional ReAct format instead of function calling for Capella AI."""

        # For Capella AI compatibility, we'll use the traditional ReAct format
        # instead of modern function calling which Capella AI doesn't support well

        # Add a callback to our chat model for Agent Catalog integration
        from agentc_langchain.chat import Callback
        callback = Callback(span=span, tools=self.tools, output=self.output)
        self.chat_model.callbacks.append(callback)

        # Get tools from the Agent Catalog prompt (proper Agent Catalog way)
        simple_tools = []

        # Get the prompt and extract tools from it
        if self.prompt and hasattr(self.prompt, 'tools'):
            for tool_result in self.prompt.tools:
                try:
                    # Get tool name from the meta descriptor
                    tool_name = tool_result.meta.name

                    # Find the actual tool using the catalog
                    catalog_tool = self.catalog.find("tool", name=tool_name)
                    if not catalog_tool:
                        logger.error(f"❌ Tool not found in catalog: {tool_name}")
                        continue

                    # Create simple Tool with our wrapper
                    simple_tool = Tool(
                        name=tool_name,
                        description=getattr(catalog_tool, 'description', catalog_tool.meta.description if hasattr(catalog_tool, 'meta') else f"Tool for {tool_name.replace('_', ' ')}"),
                        func=self._create_capella_compatible_func(catalog_tool, tool_name)
                    )
                    simple_tools.append(simple_tool)
                    logger.info(f"✅ Added Capella-compatible tool from prompt: {tool_name}")
                except Exception as e:
                    tool_name = getattr(tool_result, 'meta', {}).get('name', 'unknown') if hasattr(tool_result, 'meta') else 'unknown'
                    logger.error(f"❌ Failed to add tool {tool_name}: {e}")
        else:
            logger.warning("No tools found in Agent Catalog prompt or prompt not loaded properly")

        # Use Agent Catalog prompt content directly - no fallbacks
        if self.prompt_content is not None:
            # Handle Agent Catalog prompt template variables
            import datetime
            current_date = datetime.date.today().strftime("%Y-%m-%d")

            # Extract the actual string content from Agent Catalog prompt object
            if hasattr(self.prompt_content, 'content'):
                prompt_str = str(self.prompt_content.content)
            else:
                prompt_str = str(self.prompt_content)

            # Replace Agent Catalog variables with actual content
            prompt_str = prompt_str.replace("{current_date}", current_date)

            # Create tool descriptions and names for the prompt
            tool_descriptions = []
            tool_names = []
            for tool in simple_tools:
                tool_descriptions.append(f"{tool.name}: {tool.description}")
                tool_names.append(tool.name)

            tools_str = "\n".join(tool_descriptions)
            tool_names_str = ", ".join(tool_names)

            # Replace tool placeholders before escaping other braces
            prompt_str = prompt_str.replace("{tools}", tools_str)
            prompt_str = prompt_str.replace("{tool_names}", tool_names_str)

            # Debug: Print the prompt to see what we have
            logger.info("🔍 Agent Catalog prompt after variable replacement:")
            logger.info(f"Length: {len(prompt_str)} characters")
            logger.info(f"Tools replacement: '{tools_str}' ({len(tools_str)} chars)")
            logger.info(f"Tool names replacement: '{tool_names_str}' ({len(tool_names_str)} chars)")
            if "{tools}" in prompt_str:
                logger.warning("⚠️ {tools} still found in prompt after replacement!")
            if "{tool_names}" in prompt_str:
                logger.warning("⚠️ {tool_names} still found in prompt after replacement!")
            # Show a snippet around the tools section
            tools_pos = prompt_str.find("You have access to the following tools:")
            if tools_pos >= 0:
                snippet = prompt_str[tools_pos:tools_pos+200]
                logger.info(f"Tools section: {repr(snippet)}")

            # Escape any remaining curly braces that aren't LangChain variables
            # This fixes the "Input to PromptTemplate is missing variables {''}" error
            import re
            # Find all {xxx} patterns that aren't input or agent_scratchpad
            def escape_braces(match):
                content = match.group(1)
                if content in ['input', 'agent_scratchpad']:
                    return match.group(0)  # Keep LangChain variables as-is
                else:
                    logger.info(f"🔧 Escaping placeholder: {{{content}}}")
                    return '{{' + content + '}}'  # Escape other braces

            prompt_str = re.sub(r'\{([^}]*)\}', escape_braces, prompt_str)

            # Ensure we have the required LangChain variables for ReAct format
            if "{input}" not in prompt_str:
                prompt_str = prompt_str + "\n\nQuestion: {input}\nThought:{agent_scratchpad}"

            # Debug: Check final prompt before creating PromptTemplate
            logger.info("🔍 Final prompt template content:")
            logger.info(f"Contains {{tools}}: {'{tools}' in prompt_str}")
            logger.info(f"Contains {{tool_names}}: {'{tool_names}' in prompt_str}")

            # Save the prompt to a file for inspection
            with open('/tmp/final_prompt.txt', 'w') as f:
                f.write(prompt_str)
            logger.info("💾 Saved final prompt to /tmp/final_prompt.txt for inspection")

            # Create PromptTemplate with Agent Catalog content
            # Since we've pre-filled {tools} and {tool_names}, we need to tell LangChain they're partial
            react_prompt = PromptTemplate(
                template=prompt_str,
                input_variables=["input", "agent_scratchpad"],
                partial_variables={"tools": tools_str, "tool_names": tool_names_str}
            )
        else:
            # Only if Agent Catalog prompt fails to load
            raise ValueError("Agent Catalog prompt not loaded - check prompt_name='flight_search_assistant'")

        # Create traditional ReAct agent with Agent Catalog prompt
        agent = create_react_agent(self.chat_model, simple_tools, react_prompt)

        # Return AgentExecutor with verbose logging to match original output
        return AgentExecutor(
            agent=agent,
            tools=simple_tools,
            verbose=True,  # Enable verbose output like the original
            handle_parsing_errors=True,
            max_iterations=2,  # Reduce to encourage single tool call + Final Answer
            return_intermediate_steps=True
        )

    def _create_capella_compatible_func(self, catalog_tool, tool_name):
        """Create a simple function wrapper for Capella AI compatibility."""

        def wrapper_func(tool_input: str) -> str:
            """Simple wrapper that handles Capella AI's text-based tool calling."""
            try:
                logger.info(f"🔧 Tool {tool_name} called with input: {repr(tool_input)}")

                # Clean the input
                clean_input = self._clean_tool_input(tool_input)

                # Call the catalog tool appropriately
                if tool_name == "lookup_flight_info":
                    result = self._handle_lookup_flight_info(catalog_tool, clean_input)
                elif tool_name == "save_flight_booking":
                    result = catalog_tool.func(booking_input=clean_input)
                elif tool_name == "retrieve_flight_bookings":
                    if not clean_input or clean_input.lower() in ["", "all", "none"]:
                        result = catalog_tool.func(booking_query="")
                    else:
                        result = catalog_tool.func(booking_query=clean_input)
                elif tool_name == "search_airline_reviews":
                    if not clean_input:
                        return "Error: Please provide a search query for airline reviews"
                    result = catalog_tool.func(query=clean_input)
                else:
                    result = catalog_tool.func(clean_input)

                logger.info(f"✅ Tool {tool_name} executed successfully")
                return str(result) if result is not None else "No results found"

            except Exception as e:
                error_msg = f"Error in tool {tool_name}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                return error_msg

        return wrapper_func

    def _clean_tool_input(self, tool_input: str) -> str:
        """Clean and normalize tool input for consistent processing."""
        if not isinstance(tool_input, str):
            tool_input = str(tool_input)

        # Remove ReAct format artifacts - handle all variations
        artifacts_to_remove = [
            '"\nObservation', '\nObservation', 'Observation',
            '\nThought:', 'Thought:', '\nAction:', 'Action:',
            '\nAction Input:', 'Action Input:', '\nFinal Answer:', 'Final Answer:'
        ]

        clean_input = tool_input
        for artifact in artifacts_to_remove:
            if artifact in clean_input:
                clean_input = clean_input.split(artifact)[0]

        # Clean up quotes and whitespace
        clean_input = clean_input.strip().strip("\"'").strip()
        clean_input = " ".join(clean_input.split())

        return clean_input

    def _handle_lookup_flight_info(self, catalog_tool, tool_input: str):
        """Handle lookup_flight_info with flexible airport code parsing."""
        import re

        source = None
        dest = None

        # Support key=value style inputs
        try:
            m_src = re.search(r"source_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
            m_dst = re.search(r"destination_airport\s*[:=]\s*\"?([A-Za-z]{3})\"?", tool_input, re.I)
            if m_src and m_dst:
                source = m_src.group(1).upper()
                dest = m_dst.group(1).upper()
        except Exception:
            pass

        # Fallback: comma separated codes
        if source is None or dest is None:
            if ',' in tool_input:
                parts = tool_input.split(',')
                if len(parts) >= 2:
                    source = parts[0].strip().upper()
                    dest = parts[1].strip().upper()

        # Fallback: natural language
        if source is None or dest is None:
            words = tool_input.upper().split()
            airport_codes = [w for w in words if len(w) == 3 and w.isalpha()]
            if len(airport_codes) >= 2:
                source, dest = airport_codes[0], airport_codes[1]

        if not source or not dest:
            return "Error: Please provide source and destination airports (e.g., JFK,LAX or JFK to LAX)"

        return catalog_tool.func(source_airport=source, destination_airport=dest)

    def _invoke(
        self,
        span: agentc.Span,
        state: FlightSearchState,
        config: langchain_core.runnables.RunnableConfig,
    ) -> FlightSearchState:
        """Handle flight search conversation using ReActAgent."""

        # Initialize conversation if this is the first message
        if not state["messages"]:
            initial_msg = langchain_core.messages.HumanMessage(content=state["query"])
            state["messages"].append(initial_msg)
            logger.info(f"Flight Query: {state['query']}")

        # Use the ReActAgent's built-in create_react_agent method
        # This automatically handles prompt loading, tool integration, and span management
        agent = self.create_react_agent(span)

        # Execute the agent with the correct input format for AgentExecutor
        response = agent.invoke({"input": state["query"]}, config=config)

        # Extract tool outputs from AgentExecutor intermediate_steps for search_results tracking
        if "intermediate_steps" in response and response["intermediate_steps"]:
            tool_outputs = []
            for step in response["intermediate_steps"]:
                if isinstance(step, tuple) and len(step) >= 2:
                    # step[0] is the action, step[1] is the tool output/observation
                    tool_output = str(step[1])
                    if tool_output and tool_output.strip():
                        tool_outputs.append(tool_output)
            state["search_results"] = tool_outputs

        # Update state with the final response from AgentExecutor
        if "output" in response:
            # Add the agent's final response as an AI message
            assistant_msg = langchain_core.messages.AIMessage(content=response["output"])
            state["messages"].append(assistant_msg)

        state["resolved"] = True
        return state


class FlightSearchGraph(agentc_langgraph.graph.GraphRunnable):
    """Flight search conversation graph using Agent Catalog."""

    def __init__(self, catalog, span, chat_model=None):
        """Initialize the flight search graph with optional chat model."""
        super().__init__(catalog=catalog, span=span)
        self.chat_model = chat_model

    @staticmethod
    def build_starting_state(query: str) -> FlightSearchState:
        """Build the initial state for the flight search - single user system."""
        return FlightSearchState(
            messages=[],
            query=query,
            resolved=False,
            search_results=[],
        )

    def compile(self):
        """Compile the LangGraph workflow."""

        # Build the flight search agent with catalog integration
        search_agent = FlightSearchAgent(
            catalog=self.catalog, span=self.span, chat_model=self.chat_model
        )

        # Create a wrapper function for the ReActAgent
        def flight_search_node(state: FlightSearchState) -> FlightSearchState:
            """Wrapper function for the flight search ReActAgent."""
            return search_agent._invoke(
                span=self.span,
                state=state,
                config={},  # Empty config for now
            )

        # Create a simple workflow graph for flight search
        workflow = langgraph.graph.StateGraph(FlightSearchState)

        # Add the flight search agent node using the wrapper function
        workflow.add_node("flight_search", flight_search_node)

        # Set entry point and simple flow
        workflow.set_entry_point("flight_search")
        workflow.add_edge("flight_search", langgraph.graph.END)

        return workflow.compile()


def clear_bookings_and_reviews():
    """Clear existing flight bookings to start fresh for demo."""
    try:
        client = create_couchbase_client()
        client.connect()

        # Clear bookings scope using environment variables
        bookings_scope = "agentc_bookings"
        client.clear_scope(bookings_scope)
        logger.info(
            f"✅ Cleared existing flight bookings for fresh test run: {os.environ['CB_BUCKET']}.{bookings_scope}"
        )

        # Check if airline reviews collection needs clearing by comparing expected vs actual document count
        try:
            # Import to get expected document count without loading all data
            from data.airline_reviews_data import _data_manager

            # Get expected document count (this uses cached data if available)
            expected_docs = _data_manager.process_to_texts()
            expected_count = len(expected_docs)

            # Check current document count in collection
            try:
                count_query = f"SELECT COUNT(*) as count FROM `{os.environ['CB_BUCKET']}`.`{os.environ['CB_SCOPE']}`.`{os.environ['CB_COLLECTION']}`"
                count_result = client.cluster.query(count_query)
                count_row = next(iter(count_result))
                existing_count = count_row["count"]

                logger.info(
                    f"📊 Airline reviews collection: {existing_count} existing, {expected_count} expected"
                )

                if existing_count == expected_count:
                    logger.info(
                        f"✅ Collection already has correct document count ({existing_count}), skipping clear"
                    )
                else:
                    logger.info(
                        f"🗑️  Clearing airline reviews collection: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                    )
                    client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                    logger.info(
                        f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                    )

            except KeyspaceNotFoundException:
                # Collection doesn't exist yet - this is expected for fresh setup
                logger.info(
                    f"📊 Collection doesn't exist yet, will create and load fresh data"
                )
            except Exception as count_error:
                # Other query errors - clear anyway to ensure fresh start
                logger.info(
                    f"📊 Collection query failed, will clear and reload: {count_error}"
                )
                client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
                logger.info(
                    f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
                )

        except Exception as e:
            logger.warning(f"⚠️  Could not check collection count, clearing anyway: {e}")
            client.clear_collection_data(os.environ["CB_SCOPE"], os.environ["CB_COLLECTION"])
            logger.info(
                f"✅ Cleared existing airline reviews for fresh data load: {os.environ['CB_BUCKET']}.{os.environ['CB_SCOPE']}.{os.environ['CB_COLLECTION']}"
            )

    except Exception as e:
        logger.warning(f"❌ Could not clear bookings: {e}")


def query_agent_catalog_logs():
    """Query and display Agent Catalog activity logs."""
    try:
        # Connect to Agent Catalog cluster
        auth = PasswordAuthenticator(
            os.environ["AGENT_CATALOG_USERNAME"], os.environ["AGENT_CATALOG_PASSWORD"]
        )
        options = ClusterOptions(auth)
        cluster = Cluster(os.environ["AGENT_CATALOG_CONN_STRING"], options)
        cluster.wait_until_ready(timedelta(seconds=10))

        bucket = os.environ["AGENT_CATALOG_BUCKET"]

        logger.info("Querying Agent Catalog activity logs...")
        logger.info("=" * 50)

        query = cluster.query(f"""
            FROM
                `{bucket}`.agent_activity.Sessions() s
            SELECT
                s.sid,
                s.cid,
                s.root,
                s.start_t,
                s.content,
                s.ann
            ORDER BY s.start_t DESC
            LIMIT 10;
        """)

        for result in query:
            print(f"Session ID: {result.get('sid', 'N/A')}")
            print(f"Content ID: {result.get('cid', 'N/A')}")
            print(f"Root: {result.get('root', 'N/A')}")
            print(f"Start Time: {result.get('start_t', 'N/A')}")
            print(f"Content: {result.get('content', 'N/A')}")
            print(f"Annotations: {result.get('ann', 'N/A')}")
            print("-" * 30)

    except Exception as e:
        logger.warning(f"Could not query Agent Catalog logs: {e}")


def setup_flight_search_agent():
    """Common setup function for flight search agent - returns all necessary components."""
    try:
        # Setup environment first
        setup_environment()

        # Initialize Agent Catalog
        catalog = agentc.Catalog(
            conn_string=os.environ["AGENT_CATALOG_CONN_STRING"],
            username=os.environ["AGENT_CATALOG_USERNAME"],
            password=SecretStr(os.environ["AGENT_CATALOG_PASSWORD"]),
            bucket=os.environ["AGENT_CATALOG_BUCKET"],
        )
        application_span = catalog.Span(name="Flight Search Agent", blacklist=set())

        # Test Capella AI connectivity
        if os.getenv("CAPELLA_API_ENDPOINT"):
            if not test_capella_connectivity():
                logger.warning("❌ Capella AI connectivity test failed. Will use OpenAI fallback.")
        else:
            logger.info("ℹ️ Capella API not configured - will use OpenAI models")

        # Create CouchbaseClient for all operations
        client = create_couchbase_client()

        # Setup everything in one call - bucket, scope, collection
        client.setup_collection(
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            clear_existing_data=False,  # Let data loader decide based on count check
        )

        # Setup vector search index
        try:
            with open("agentcatalog_index.json") as file:
                index_definition = json.load(file)
            logger.info("Loaded vector search index definition from agentcatalog_index.json")
            client.setup_vector_search_index(index_definition, os.environ["CB_SCOPE"])
        except Exception as e:
            logger.warning(f"Error loading index definition: {e!s}")
            logger.info("Continuing without vector search index...")

        # Setup embeddings using shared 4-case priority ladder
        embeddings, _ = setup_ai_services(framework="langgraph")

        # Import data loader function
        from data.airline_reviews_data import load_airline_reviews_to_couchbase

        # Setup vector store with airline reviews data
        vector_store = client.setup_vector_store_langchain(
            scope_name=os.environ["CB_SCOPE"],
            collection_name=os.environ["CB_COLLECTION"],
            index_name=os.environ["CB_INDEX"],
            embeddings=embeddings,
            data_loader_func=load_airline_reviews_to_couchbase,
        )

        # Setup LLM using shared 4-case priority ladder
        _, chat_model = setup_ai_services(framework="langgraph", temperature=0.1)

        # Create the flight search graph with the chat model
        flight_graph = FlightSearchGraph(
            catalog=catalog, span=application_span, chat_model=chat_model
        )
        # Compile the graph
        compiled_graph = flight_graph.compile()

        logger.info("Agent Catalog integration successful")

        return compiled_graph, application_span

    except Exception as e:
        logger.exception(f"Setup error: {e}")
        logger.info("Ensure Agent Catalog is published: agentc index . && agentc publish")
        raise


def run_interactive_demo():
    """Run an interactive flight search demo."""
    logger.info("Flight Search Agent - Interactive Demo")
    logger.info("=" * 50)

    try:
        compiled_graph, application_span = setup_flight_search_agent()

        # Interactive flight search loop
        logger.info("Available commands:")
        logger.info("- Enter flight search queries (e.g., 'Find flights from NYC to LAX')")
        logger.info("- 'logs' - View Agent Catalog activity logs")
        logger.info("- 'quit' - Exit the demo")
        logger.info(
            "Try asking: 'Find cheap flights to Miami' or 'Book a business class flight to Boston'"
        )
        logger.info("─" * 40)

        while True:
            query = input("🔍 Enter flight search query (or 'quit'/'logs'): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using Flight Search Agent!")
                break

            if query.lower() == "logs":
                logger.info("\n" + "=" * 50)
                logger.info("AGENT CATALOG ACTIVITY LOGS")
                logger.info("=" * 50)
                query_agent_catalog_logs()
                continue

            if not query:
                continue

            try:
                logger.info(f"Flight Query: {query}")

                # Build starting state - single user system
                state = FlightSearchGraph.build_starting_state(query=query)

                # Run the flight search
                result = compiled_graph.invoke(state)

                # Display results summary
                if result.get("search_results"):
                    logger.info(f"Found {len(result['search_results'])} flight options")

                logger.info(f"Search completed: {result.get('resolved', False)}")

            except Exception as e:
                logger.exception(f"Search error: {e}")

    except Exception as e:
        logger.exception(f"Demo initialization error: {e}")


def run_test():
    """Run comprehensive test of flight search agent with booking functionality."""
    logger.info("Flight Search Agent - Comprehensive Test Suite")
    logger.info("=" * 55)

    try:
        # Clear existing bookings first for a clean test run
        clear_bookings_and_reviews()

        compiled_graph, application_span = setup_flight_search_agent()

        # Comprehensive test scenarios covering all core functionality
        test_queries = [
            "Find flights from JFK to LAX for tomorrow",
            "Book a flight from LAX to JFK for tomorrow, 2 passengers, business class",
            "Book an economy flight from JFK to MIA for next week, 1 passenger",
            "Show me my current flight bookings",
            "What do passengers say about SpiceJet's service quality?",
        ]

        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test {i}: {query}")
            try:
                state = FlightSearchGraph.build_starting_state(query=query)
                result = compiled_graph.invoke(state)

                if result.get("search_results"):
                    logger.info(f"✅ Found {len(result['search_results'])} flight options")
                logger.info(f"✅ Test {i} completed: {result.get('resolved', False)}")

            except Exception as e:
                logger.exception(f"❌ Test {i} failed: {e}")

            logger.info("-" * 50)

        logger.info("All tests completed!")
        logger.info("💡 Run 'python main.py logs' to view Agent Catalog activity logs")

    except Exception as e:
        logger.exception(f"Test error: {e}")


def run_flight_search_demo():
    """Legacy function - redirects to interactive demo for compatibility."""
    run_interactive_demo()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            run_test()
        elif sys.argv[1] == "logs":
            setup_environment()
            query_agent_catalog_logs()
        else:
            print("Usage: python main.py [test|logs]")
            print("  test - Run comprehensive test suite")
            print("  logs - Query Agent Catalog activity logs")
            print("  (no args) - Run interactive demo")
            sys.exit(1)
    else:
        run_interactive_demo()

    # Uncomment the following lines to visualize the LangGraph workflow:
    # compiled_graph.get_graph().draw_mermaid_png(output_file_path="flight_search_graph.png")
    # compiled_graph.get_graph().draw_ascii()