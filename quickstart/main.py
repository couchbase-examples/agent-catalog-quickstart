#!/usr/bin/env python3
"""
Flight Search Agent Quickstart - Simplified Agent Catalog Implementation

A simplified flight search agent demonstrating Agent Catalog integration
with basic flight booking functionality.
"""

import getpass
import json
import logging
import os
import sys
from datetime import timedelta

import agentc
import dotenv
import langchain_openai.chat_models
from langchain_core.tools import Tool

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(override=True)


def _set_if_undefined(var: str):
    """Set environment variable if not defined."""
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}: ")


def setup_environment():
    """Setup required environment variables."""
    # Required for OpenAI
    _set_if_undefined("OPENAI_API_KEY")
    
    # Agent Catalog environment variables
    agent_catalog_vars = [
        "AGENT_CATALOG_CONN_STRING", 
        "AGENT_CATALOG_USERNAME", 
        "AGENT_CATALOG_PASSWORD", 
        "AGENT_CATALOG_BUCKET"
    ]
    
    for var in agent_catalog_vars:
        if not os.environ.get(var):
            # Set defaults for demo
            if var == "AGENT_CATALOG_CONN_STRING":
                os.environ[var] = "couchbase://localhost"
            elif var == "AGENT_CATALOG_USERNAME":
                os.environ[var] = "Administrator"  
            elif var == "AGENT_CATALOG_PASSWORD":
                os.environ[var] = "password"
            elif var == "AGENT_CATALOG_BUCKET":
                os.environ[var] = "agent-catalog"
    
    # Couchbase variables for flight data
    cb_vars = {
        "CB_CONN_STRING": "couchbase://localhost",
        "CB_USERNAME": "Administrator", 
        "CB_PASSWORD": "password",
        "CB_BUCKET": "vector-search-testing"
    }
    
    for var, default in cb_vars.items():
        if not os.environ.get(var):
            os.environ[var] = default


class SimpleFlightAgent:
    """Simplified flight search agent with direct tool integration."""
    
    def __init__(self, catalog: agentc.Catalog):
        self.catalog = catalog
        self.chat_model = langchain_openai.chat_models.ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), 
            temperature=0.1
        )
        self.tools = self._load_tools()
    
    def _load_tools(self):
        """Load tools from Agent Catalog with simplified parameter handling."""
        tools = []
        tool_configs = [
            ("lookup_flight_info", "Find flight routes between airports"),
            ("save_flight_booking", "Save a flight booking"),  
            ("retrieve_flight_bookings", "Retrieve existing flight bookings"),
            ("search_flight_policies", "Search flight policies")
        ]
        
        for tool_name, description in tool_configs:
            try:
                tool_obj = self.catalog.find("tool", name=tool_name)
                if tool_obj and hasattr(tool_obj, "func"):
                    # Create wrapper with simplified parameter handling
                    wrapped_func = self._create_tool_wrapper(tool_obj.func, tool_name)
                    lc_tool = Tool(
                        name=tool_obj.meta.name,
                        description=tool_obj.meta.description or description,
                        func=wrapped_func,
                    )
                    tools.append(lc_tool)
                    logger.info(f"Loaded tool: {tool_name}")
            except Exception as e:
                logger.warning(f"Could not load tool {tool_name}: {e}")
        
        return tools
    
    def _create_tool_wrapper(self, func, tool_name):
        """Create a wrapper function with simplified parameter mapping."""
        def wrapper(input_str: str) -> str:
            try:
                logger.info(f"Tool {tool_name} called with input: {input_str}")
                
                # Handle different input formats
                if tool_name == "retrieve_flight_bookings":
                    # This tool takes no parameters
                    return func()
                
                # Try to parse as JSON first
                try:
                    if input_str.strip() and input_str.strip() not in ["None", "null", ""]:
                        params = json.loads(input_str)
                    else:
                        params = {}
                except json.JSONDecodeError:
                    # If not JSON, treat as simple text input
                    params = {"query": input_str}
                
                # Map parameters based on tool
                mapped_params = self._map_parameters(tool_name, params, input_str)
                
                # Call the function with mapped parameters
                return func(**mapped_params)
                
            except Exception as e:
                logger.error(f"Tool {tool_name} execution error: {e}")
                return f"Error executing {tool_name}: {str(e)}"
        
        return wrapper
    
    def _map_parameters(self, tool_name: str, params: dict, original_input: str) -> dict:
        """Map parameters for each tool type."""
        
        if tool_name == "lookup_flight_info":
            # Map common parameter names
            mapped = {}
            if "from" in params:
                mapped["source_airport"] = params["from"]
            elif "source_airport" in params:
                mapped["source_airport"] = params["source_airport"]
            elif "origin" in params:
                mapped["source_airport"] = params["origin"]
            
            if "to" in params:
                mapped["destination_airport"] = params["to"]
            elif "destination_airport" in params:
                mapped["destination_airport"] = params["destination_airport"]
            elif "destination" in params:
                mapped["destination_airport"] = params["destination"]
            
            return mapped
            
        elif tool_name == "save_flight_booking":
            # Map booking parameters with defaults
            mapped = {}
            
            # Extract flight info from params or parse from text
            if "flight" in params:
                flight_info = params["flight"]
                # Parse flight string like "UA flight from SFO to LAX using 319"
                if "from" in flight_info and "to" in flight_info:
                    parts = flight_info.split()
                    try:
                        from_idx = parts.index("from")
                        to_idx = parts.index("to")
                        mapped["source_airport"] = parts[from_idx + 1]
                        mapped["destination_airport"] = parts[to_idx + 1]
                    except (ValueError, IndexError):
                        pass
            
            # Map other parameters
            if "from" in params:
                mapped["source_airport"] = params["from"]
            elif "source_airport" in params:
                mapped["source_airport"] = params["source_airport"]
                
            if "to" in params:
                mapped["destination_airport"] = params["to"]  
            elif "destination_airport" in params:
                mapped["destination_airport"] = params["destination_airport"]
            
            if "date" in params:
                mapped["departure_date"] = params["date"]
            elif "departure_date" in params:
                mapped["departure_date"] = params["departure_date"]
            else:
                mapped["departure_date"] = "tomorrow"  # Default
            
            # Set defaults for other required parameters
            mapped["passengers"] = params.get("passengers", 1)
            mapped["flight_class"] = params.get("flight_class", "economy")
            mapped["return_date"] = params.get("return_date", None)
            
            return mapped
            
        elif tool_name == "search_flight_policies":
            # Simple query mapping
            if "query" in params:
                return {"query": params["query"]}
            else:
                return {"query": original_input}
        
        return params
    
    def process_query(self, query: str) -> str:
        """Process a user query and return response."""
        try:
            # Determine which tool to use based on query intent
            query_lower = query.lower()
            
            if "book" in query_lower or "booking" in query_lower:
                if "show" in query_lower or "retrieve" in query_lower or "my" in query_lower:
                    # Show existing bookings
                    tool = next((t for t in self.tools if "retrieve" in t.name), None)
                    if tool:
                        result = tool.func("")
                        return f"Here are your current bookings:\n\n{result}"
                else:
                    # Book a new flight
                    return self._handle_booking(query)
            
            elif "flight" in query_lower and ("from" in query_lower or "to" in query_lower):
                # Search for flights
                return self._handle_flight_search(query)
            
            elif "policy" in query_lower or "rule" in query_lower:
                # Search policies
                tool = next((t for t in self.tools if "policy" in t.name), None)
                if tool:
                    result = tool.func(query)
                    return f"Flight Policy Information:\n\n{result}"
            
            else:
                return "I can help you with:\n- Searching for flights\n- Booking flights\n- Viewing your bookings\n- Flight policies\n\nPlease let me know what you'd like to do!"
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"I encountered an error processing your request: {str(e)}"
    
    def _handle_flight_search(self, query: str) -> str:
        """Handle flight search queries."""
        try:
            # Extract airports from query using simple parsing
            airports = self._extract_airports(query)
            
            tool = next((t for t in self.tools if "lookup" in t.name), None)
            if tool and airports["from"] and airports["to"]:
                params = json.dumps({
                    "from": airports["from"],
                    "to": airports["to"]
                })
                result = tool.func(params)
                
                if isinstance(result, list):
                    flights = "\n".join([f"- {flight}" for flight in result])
                    return f"Available flights from {airports['from']} to {airports['to']}:\n\n{flights}"
                else:
                    return str(result)
            else:
                return "Please specify both departure and destination airports (e.g., 'flights from SFO to LAX')"
                
        except Exception as e:
            logger.error(f"Error in flight search: {e}")
            return f"Error searching for flights: {str(e)}"
    
    def _handle_booking(self, query: str) -> str:
        """Handle booking requests."""
        try:
            # Extract airports and date from query
            airports = self._extract_airports(query)
            
            if not airports["from"] or not airports["to"]:
                return "Please specify both departure and destination airports for booking."
            
            # Determine date
            date = "tomorrow"  # Default
            if "today" in query.lower():
                date = "today"
            elif "tomorrow" in query.lower():
                date = "tomorrow"
            
            tool = next((t for t in self.tools if "save" in t.name), None)
            if tool:
                params = json.dumps({
                    "from": airports["from"],
                    "to": airports["to"], 
                    "date": date
                })
                result = tool.func(params)
                return result
            else:
                return "Booking service is currently unavailable."
                
        except Exception as e:
            logger.error(f"Error in booking: {e}")
            return f"Error processing booking: {str(e)}"
    
    def _extract_airports(self, query: str) -> dict:
        """Extract airport codes from query text."""
        # Simple extraction - look for common patterns
        words = query.upper().split()
        airports = {"from": None, "to": None}
        
        # Look for "from X to Y" pattern
        try:
            if "FROM" in words and "TO" in words:
                from_idx = words.index("FROM")
                to_idx = words.index("TO")
                if from_idx + 1 < len(words):
                    airports["from"] = words[from_idx + 1]
                if to_idx + 1 < len(words):
                    airports["to"] = words[to_idx + 1]
        except (ValueError, IndexError):
            pass
        
        # Look for common airport codes
        airport_codes = ["SFO", "LAX", "JFK", "LGA", "ORD", "DFW", "ATL", "DEN", "SEA", "BOS"]
        found_codes = [word for word in words if word in airport_codes]
        
        if len(found_codes) >= 2:
            airports["from"] = found_codes[0]
            airports["to"] = found_codes[1]
        elif len(found_codes) == 1 and not airports["from"] and not airports["to"]:
            # If only one airport found, ask for clarification
            pass
        
        return airports


def run_flight_search_demo():
    """Run the simplified flight search demo."""
    
    logger.info("Flight Search Agent Quickstart")
    logger.info("=" * 40)
    
    try:
        # Setup environment
        setup_environment()
        
        # Initialize Agent Catalog
        catalog = agentc.Catalog(
            conn_string=os.environ["AGENT_CATALOG_CONN_STRING"],
            username=os.environ["AGENT_CATALOG_USERNAME"], 
            password=os.environ["AGENT_CATALOG_PASSWORD"],
            bucket=os.environ["AGENT_CATALOG_BUCKET"],
        )
        
        # Create simplified agent
        agent = SimpleFlightAgent(catalog)
        
        if not agent.tools:
            logger.error("No tools loaded from catalog")
            logger.info("Make sure Agent Catalog is published: agentc index . && agentc publish")
            return
        
        logger.info(f"Agent initialized with {len(agent.tools)} tools")
        
        # Interactive loop
        while True:
            logger.info("-" * 40)
            query = input("🔍 Enter your request (or 'quit' to exit): ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using Flight Search Agent!")
                break
            
            if not query:
                continue
            
            try:
                response = agent.process_query(query)
                print(f"\n✈️  {response}\n")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print(f"\n❌ Error: {str(e)}\n")
    
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        logger.info("Make sure Agent Catalog is published and Couchbase is running")


if __name__ == "__main__":
    run_flight_search_demo() 