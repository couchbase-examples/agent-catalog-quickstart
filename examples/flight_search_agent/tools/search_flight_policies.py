import os
import sys
import agentc

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.flight_data import FLIGHT_POLICIES


@agentc.tool
def search_flight_policies(query: str) -> str:
  """Searches flight policies for a given query.

  Args:
    query: The query to search for.

  Returns:
    The flight policies that match the query.
  """
  for policy in FLIGHT_POLICIES:
    if query.lower() in policy["title"].lower() or query.lower() in policy["content"].lower():
      return f"{policy['title']}: {policy['content']}"
  return f"No information found for '{query}'."
