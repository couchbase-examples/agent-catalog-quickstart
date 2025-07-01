import os
import sys
import agentc

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.flight_data import FLIGHT_POLICIES


@agentc.catalog.tool
def search_flight_policies(query: str) -> str:
  """Searches flight policies for a given query.

  Args:
    query: The query to search for.

  Returns:
    The flight policies that match the query.
  """
  query_lower = query.lower()
  matching_policies = []
  
  for policy in FLIGHT_POLICIES:
    if query_lower in policy["title"].lower() or query_lower in policy["content"].lower():
      matching_policies.append(f"{policy['title']}: {policy['content']}")
  
  if matching_policies:
    return "\n\n".join(matching_policies)
  
  # If no exact matches, provide general policies based on common keywords
  if any(keyword in query_lower for keyword in ["baggage", "bag", "luggage"]):
    for policy in FLIGHT_POLICIES:
      if "baggage" in policy["title"].lower() or "baggage" in policy["content"].lower():
        return f"{policy['title']}: {policy['content']}"
  
  if any(keyword in query_lower for keyword in ["cancel", "refund", "change"]):
    for policy in FLIGHT_POLICIES:
      if any(term in policy["title"].lower() or term in policy["content"].lower() 
             for term in ["cancel", "refund", "change"]):
        return f"{policy['title']}: {policy['content']}"
  
  if any(keyword in query_lower for keyword in ["check", "checkin", "boarding"]):
    for policy in FLIGHT_POLICIES:
      if any(term in policy["title"].lower() or term in policy["content"].lower() 
             for term in ["check", "boarding"]):
        return f"{policy['title']}: {policy['content']}"
  
  # Return a comprehensive policy overview if no specific match
  return "Here are our main flight policies:\n\n" + "\n\n".join([
    f"{policy['title']}: {policy['content']}" for policy in FLIGHT_POLICIES[:3]
  ])
