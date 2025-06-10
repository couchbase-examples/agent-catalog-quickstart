import dotenv
import getpass
import os

from agentc.catalog import Catalog
from langchain_openai.chat_models import ChatOpenAI
from agentc_langchain.agent import ReActAgent
from langchain_core.messages import SystemMessage
from tools import hello_tool

# 1. Setup Environment
def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))
_set_if_undefined("OPENAI_API_KEY")

# 2. Setup Agent Catalog
catalog = Catalog(tools=[hello_tool])
application_span = catalog.start_span(name="Quickstart Application")

# 3. Setup LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 4. Create a simple agent
class MyAgent(ReActAgent):
    def __init__(self, span):
        super().__init__(catalog=catalog, prompt_name="my_agent", span=span, chat_model=llm)

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        result = agent.invoke(state)
        result["messages"][-1] = SystemMessage(content=result["messages"][-1].content, name="my_agent")
        return result

# 5. Run the agent
agent = MyAgent(span=application_span)
user_input = "use the hello_tool to say hello to 'World'"
application_span.log(content={"kind": "user", "value": user_input})

# Note: This is a simplified invoke for now
# We will build up to a more complete graph execution
result = agent.invoke({"messages": [("user", user_input)]})

print(result)

print("Quickstart script setup complete.") 