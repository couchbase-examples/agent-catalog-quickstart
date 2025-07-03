#!/usr/bin/env python3
"""
Route Planner Agent - Simplified Implementation

A basic route planning agent demonstrating Agent Catalog tools
for intelligent travel planning.
"""

import logging
import sys

import agentc
import dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv(override=True)


class SimpleRoutePlanner:
    """Simplified route planner using Agent Catalog tools."""

    def __init__(self):
        """Initialize the route planner."""
        self.catalog = None
        self.setup()

    def setup(self):
        """Setup the route planner."""
        try:
            # Load Agent Catalog
            self.catalog = agentc.Catalog()
            logger.info("Route planner setup complete")

        except Exception as e:
            logger.error(f"Error setting up route planner: {e}")

    def plan_route(self, query: str) -> str:
        """Plan a route based on user query."""
        try:
            if not self.catalog:
                return "Route planner not properly initialized. Please check your configuration."

            # Try to find and use the search_routes tool
            try:
                tool_obj = self.catalog.find("tool", name="search_routes")
                if tool_obj and hasattr(tool_obj, "func"):
                    result = tool_obj.func(query=query)
                    return str(result)
                else:
                    available_tools = list(self.catalog.find_tools())
                    tool_names = [t.meta.name if hasattr(t, "meta") else str(t) for t in available_tools]
                    return f"search_routes tool not found. Available tools: {tool_names}"
            except Exception as e:
                return f"Error searching routes: {e!s}. Try a simpler query like 'route from New York to Boston'."

        except Exception as e:
            return f"Error planning route: {e!s}"

    def calculate_distance(self, origin: str, destination: str) -> str:
        """Calculate distance between two locations."""
        try:
            if not self.catalog:
                return "Route planner not properly initialized."

            # Try to find and use the calculate_distance tool
            try:
                tool_obj = self.catalog.find("tool", name="calculate_distance")
                if tool_obj and hasattr(tool_obj, "func"):
                    result = tool_obj.func(origin=origin, destination=destination)
                    return str(result)
                else:
                    available_tools = list(self.catalog.find_tools())
                    tool_names = [t.meta.name if hasattr(t, "meta") else str(t) for t in available_tools]
                    return f"calculate_distance tool not found. Available tools: {tool_names}"
            except Exception as e:
                return f"Error calculating distance: {e!s}"

        except Exception as e:
            return f"Error calculating distance: {e!s}"

    def list_available_tools(self) -> str:
        """List all available tools in the catalog."""
        try:
            if not self.catalog:
                return "Catalog not initialized"

            tools = list(self.catalog.find_tools())
            tool_names = []
            for tool in tools:
                if hasattr(tool, "meta") and hasattr(tool.meta, "name"):
                    tool_names.append(tool.meta.name)
                else:
                    tool_names.append(str(tool))

            return f"Available tools: {tool_names}"
        except Exception as e:
            return f"Error listing tools: {e}"


def run_interactive_demo():
    """Run an interactive demo of the route planner."""
    logger.info("Starting Route Planner Agent - Interactive Demo")
    logger.info("=" * 50)

    planner = SimpleRoutePlanner()

    if not planner.catalog:
        logger.error("Failed to initialize route planner. Please check your configuration.")
        return

    logger.info("Route planner initialized successfully!")
    logger.info(planner.list_available_tools())
    logger.info("Available commands:")
    logger.info("- 'plan <query>' - Plan a route (e.g., 'plan route from New York to Boston')")
    logger.info("- 'distance <origin> to <destination>' - Calculate distance")
    logger.info("- 'tools' - List available tools")
    logger.info("- 'quit' - Exit the demo")
    logger.info("Try asking: 'plan scenic route in California' or 'distance San Francisco to Los Angeles'")
    logger.info("-" * 50)

    while True:
        try:
            user_input = input("\nEnter your request: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                logger.info("Thanks for using the Route Planner!")
                break

            if user_input.lower() == "tools":
                logger.info(planner.list_available_tools())
                continue

            if user_input.lower().startswith("plan "):
                query = user_input[5:]  # Remove 'plan ' prefix
                result = planner.plan_route(query)
                logger.info(f"Route Planning Results:\n{result}")

            elif " to " in user_input.lower() and any(word in user_input.lower() for word in ["distance", "from"]):
                # Parse distance query
                parts = user_input.lower().replace("distance", "").replace("from", "").strip().split(" to ")
                if len(parts) == 2:
                    origin = parts[0].strip()
                    destination = parts[1].strip()
                    result = planner.calculate_distance(origin, destination)
                    logger.info(f"Distance Calculation:\n{result}")
                else:
                    logger.warning("Please use format: 'distance <origin> to <destination>'")

            else:
                # Default to route planning
                result = planner.plan_route(user_input)
                logger.info(f"Route Planning Results:\n{result}")

        except KeyboardInterrupt:
            logger.info("Thanks for using the Route Planner!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def run_simple_test():
    """Run a simple test of the route planner."""
    logger.info("Running Route Planner Test")
    logger.info("=" * 30)

    planner = SimpleRoutePlanner()

    if not planner.catalog:
        logger.error("Test failed: Route planner not initialized")
        return False

    # Test tool listing
    logger.info("0. Listing available tools...")
    tools_result = planner.list_available_tools()
    logger.info(f"Tools: {tools_result}")

    # Test route planning
    logger.info("1. Testing route search...")
    result = planner.plan_route("route from San Francisco to Los Angeles")
    result_display = f"{result[:200]}..." if len(result) > 200 else f"{result}"
    logger.info(f"Result: {result_display}")

    # Test distance calculation
    logger.info("2. Testing distance calculation...")
    result = planner.calculate_distance("San Francisco", "Los Angeles")
    result_display = f"{result[:200]}..." if len(result) > 200 else f"{result}"
    logger.info(f"Result: {result_display}")

    logger.info("Test completed!")
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_simple_test()
    else:
        run_interactive_demo()
