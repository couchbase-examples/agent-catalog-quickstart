import dotenv
import getpass
import os
import sys

# Add the root directory to the python path so we can import tools.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentc.catalog import Catalog
from agentc_langchain.chat import Callback
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_openai.chat_models import ChatOpenAI

# 1. Setup Environment
def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
_set_if_undefined("OPENAI_API_KEY")

# 2. Setup Agent Catalog
catalog = Catalog()
application_span = catalog.Span(name="LangChain Example")

# 3. Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0, callbacks=[Callback(span=application_span)])

# 4. Create a simple agent
tool_result = catalog.find("tool", name="hello_tool")
tools = [Tool(name=tool_result.meta.name, func=tool_result.func, description=tool_result.meta.description)]

# Get the prompt to use - you can use any prompt from the Langchain Hub
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Run the agent
while (user_input := input(">> ")) != "exit":
    if not user_input:
        continue
    response = agent_executor.invoke({"input": user_input})
    print(response["output"])

print("LangChain example complete.")
