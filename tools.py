from agentc.catalog import tool

@tool
def hello_tool(name: str) -> str:
    """A simple tool that says hello."""
    return f"Hello, {name}!" 