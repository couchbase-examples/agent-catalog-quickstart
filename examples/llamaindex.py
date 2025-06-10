import dotenv
import getpass
import os
import sys

# Add the root directory to the python path so we can import tools.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentc.catalog import Catalog
from agentc_llamaindex.chat import Callback
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from tools import hello_tool

# 1. Setup Environment
def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
_set_if_undefined("OPENAI_API_KEY")

# 2. Setup Agent Catalog
catalog = Catalog()
application_span = catalog.Span(name="LlamaIndex Example")

# 3. Setup LLM
llm = OpenAI(model="gpt-4o")
llm.callback_manager.add_handler(Callback(span=application_span))


# 4. Create a simple agent
tool_result = catalog.find("tool", name="hello_tool")
tools = [FunctionTool.from_defaults(fn=tool_result.func)]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# 5. Run the agent
while (user_input := input(">> ")) != "exit":
    if not user_input:
        continue
    response = agent.chat(user_input)
    print(response)

print("LlamaIndex example complete.")
