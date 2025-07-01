import dotenv
import getpass
import os
import sys

# Add the root directory to the python path so we can import tools.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentc.catalog import Catalog
from agentc_langchain.chat import Callback
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 1. Setup Environment
def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
_set_if_undefined("OPENAI_API_KEY")

# 2. Setup Agent Catalog
# This assumes you have already run `agentc init` and `agentc index`
# from the root of the repository.
catalog = Catalog()
application_span = catalog.Span(name="LangGraph Example")

# 3. Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, callbacks=[Callback(span=application_span)])

# 4. Create a simple agent
tool_result = catalog.find("tool", name="hello_tool")
tools = [tool_result.func]
agent_executor = create_react_agent(llm, tools)


# 5. Run the agent
while (user_input := input(">> ")) != "exit":
    if not user_input:
        continue
    events = agent_executor.stream(
        {"messages": [("user", user_input)]},
    )
    for event in events:
        if "event" not in event:
            output = event.get("agent", {}).get("messages", [])
            if len(output):
                print(output[-1].content)
            continue
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Same as print(content, end="", flush=True)
                print(content, end="")
        elif kind == "on_tool_end":
            print(f"\nTool output: {event['data']['output']}\n")

print("LangGraph example complete.") 