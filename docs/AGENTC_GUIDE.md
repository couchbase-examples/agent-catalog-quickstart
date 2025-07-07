



Agent Catalog (Agentc) 
User Guide

Table of Contents
Table of Contents	1
Introduction	2
Step 1 - Build an Agent with LangGraph	2
Setup	3
Define tools and prompts	4
Create a Graph	5
Invoke the graph	8
Step 2: Integrate Agentc	9
Install Agentc	9
Create tools and prompts using Agentc	9
REPL Tool	10
Web Search Tool	11
System Prompt	12
Create the Catalog	13
Publish the catalogs	15
Using the Provider	16
Using the Auditor	17
Managing Catalogs	21
Additional Commands	22
Common Questions/Issues	23
Glossary	24
References	25


Introduction
Agent Catalog aims to simplify your agent development process at scale by providing a consolidated view of tools and prompts used by agents and providing traceability through a logger. The logging also helps iteratively modify your workflow, tool, and/or prompts to ensure the best outcome from your agentic system.
While agentic application development is exciting, it is hard to get started on it. In this guide, we will help you build a multi-agent application using LangGraph and subsequently show how to integrate Agentc for adding traceability, tool and prompt reusability, and governance.
Step 1 - Build an Agent with LangGraph
We will build a Multi-Agent System (hereafter, MAS) with two main nodes as described below. Refer to the notebook to follow along. Figure 1 gives a high-level view of the system. 
The two nodes are independent but must communicate to order user queries. 
Research node - responsible for searching for information 
Chart generator node - responsible for generating charts based on some data

Feel free to skip this step if you have already developed a LangGraph application and proceed to integrate the Agentc.

Figure 1: Multi-agent system with two nodes. Source: [5]

Setup
Install all required packages and set the OpenAI key (create/access key from here).

pip install -U langchain_community langchain_anthropic langchain_experimental matplotlib langgraph

import getpass
import os

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
Define tools and prompts
Define a REPL (read, evaluate, print, loop) tool responsible for the execution of Python code to generate charts. The original example does a Google search using the Tavily tool but we will be using SerpAPI by Google. Please get the free API key for the same from here (inside the “Your Account” page).

from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from serpapi import GoogleSearch

@tool
def repl_tool(code: str) -> str:
    """Tool to execute python code"""

    repl = PythonREPL()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

serpapi_params = {"engine": "google", "api_key": os.getenv("SERPAPI_KEY")}

@tool()
def web_search(query: str) -> str:
    """Finds general knowledge information using Google search. Can also be used to augment more 'general' knowledge to a previous specialist query."""

    search = GoogleSearch({**serpapi_params, "q": query, "num": 5})
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(["\n".join([x["title"], x["snippet"], x["link"]]) for x in results])
    return contexts

Next, let us define the system prompt that will be given to the MAS to define its behavior and provide it with an aim.

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        " If you or any of the other assistants have the final answer or deliverable,"
        " prefix your response with FINAL ANSWER so the team knows to stop."
        f"\n{suffix}"
    )

Create a Graph
Since we are using LangGraph to build our MAS, we need to create a graph using nodes and edges while keeping track of their state. We must also define how each node can start communicating with each other.
	
To begin, let us define the agents and their nodes.

from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing import Literal

# initialise llm - using OpenAI
llm = ChatOpenAI(model="gpt-4o", openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0)
# function to go from one node to another 
def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto

# AGENT 1 - Research agent and node 
research_agent = create_react_agent(
    llm,
    tools=[web_search],
    state_modifier=make_system_prompt(
        "You can only do research. You are working with a chart generator colleague."
    ),
)

def research_node(
    state: MessagesState,
) -> Command[Literal["chart_generator", END]]:
    result = research_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "chart_generator")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="researcher"
    )
    return Command(
        update={
            # share internal message history of research agent w other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


# AGENT 2 - Chart generator agent and node
chart_agent = create_react_agent(
    llm,
    [repl_tool],
    state_modifier=make_system_prompt(
        "You can only generate charts. You are working with a researcher colleague."
    ),
)

def chart_node(state: MessagesState) -> Command[Literal["researcher", END]]:
    result = chart_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "researcher")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="chart_generator"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

Next, let's define the workflow, i.e., how these nodes interact with each other and pass information. The following code will generate a graph that looks like Figure 2.

from langgraph.graph import StateGraph, START
from IPython.display import Image, display

workflow = StateGraph(MessagesState)
workflow.add_node("researcher", research_node)
workflow.add_node("chart_generator", chart_node)
workflow.add_edge(START, "researcher")
graph = workflow.compile()


Figure 2: Multi agent system with two nodes
# Use the following code to display the graph seen in Figure 2
from IPython.display import Image
from IPython.display import display

display(Image(graph.get_graph().draw_mermaid_png()))

Invoke the graph
Finally, let's try invoking the graph we created with user input. We will try to generate a report on the UK's GDP with text and charts.

events = graph.stream(
    {
        "messages": [
            (
                "user",
                "First, get the UK's GDP over the past 5 years, then make a line chart of it. "
                "Once you make the chart, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 10},
)

# Use the following piece of code to see the entire thought process of the agent from start to end 
for s in events:
    for key in s:
        for msg in s[key]["messages"]:
            print(msg.content)
            print("----")

Note: To view the final output, refer to the source code (the output is pretty long) but essentially it is as expected - some paragraphs on the GDP followed by a line chart.

Step 2: Integrate Agentc
After covering the previous section, we observe that as developers we would have to keep track of all tools/prompts manually and retrieve the most relevant ones based on user prompts. With Agentc, let's try to resolve this. Refer to the notebook here to follow along.
Enable the following services on your Couchbase cluster:
Data, Query, Index: For storing items and searching them.
Search: For performing vector search on items.
Analytics: For creating views on audit logs and querying the views for insights on logs.

Install Agentc
Install Agentc following the steps mentioned in the readme or the documentation.

Create tools and prompts using Agentc
All the tools and prompts that are created using Agentc are stored in catalogs locally and can be published to your Couchbase cluster once finalized. Let's take a look at how we can create these.
REPL Tool
Generate a tool template to write the REPL tool used for code execution using the following Agentc CLI command. Make sure you have created a directory called tools/ to store all your tools.

$ agentc add --record-kind python_function -o tools/
Now building a new tool / prompt file. The output will be saved to: tools
Type: python_function
Name: repl_tool // fill in
Description: Tool to execute python code // fill in
Python (function) tool written to: tools/repl_tool.py

Next, go to tools/repl_tool.py	and edit the file to look like the following:

from agentc import tool
from langchain_experimental.utilities import PythonREPL

@tool
def repl_tool(code: str) -> str:
    """Tool to execute python code"""

    repl = PythonREPL()
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."

Explore more options the add command has to offer:

Use this command to see the various options available
$ agentc add --help
Usage: agentc add [OPTIONS]

  Interactively create a new tool or prompt and save it to the filesystem (output).

Options:
  -o, --output DIRECTORY          Location to save the generated tool / prompt to. Defaults to your current working directory.
  --record-kind [python_function|sqlpp_query|semantic_search|http_request|raw_prompt|jinja_prompt]
  --help                          Show this message and exit.
Web Search Tool
Just like the REPL tool, let’s create a web search tool dedicated to searching Google in case of insufficient information.

$ agentc add --record-kind python_function -o tools
Now building a new tool / prompt file. The output will be saved to: tools
Type: python_function
Name: web_search
Description: Finds general knowledge information using Google search. Can also be used to augment more 'general' knowledge to a previous specialist query.
Python (function) tool written to: tools/web_search.py

Make sure web_search_tool looks like this:

import os
from agentc import tool
from serpapi import GoogleSearch
serpapi_params = {"engine": "google", "api_key": os.getenv("SERPAPI_KEY")}

@tool()
def web_search(query: str) -> str:
    """Finds general knowledge information using Google search. Can also be used to augment more 'general' knowledge to a previous specialist query."""

    search = GoogleSearch({**serpapi_params, "q": query, "num": 5})
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(["\n".join([x["title"], x["snippet"], x["link"]]) for x in results])
    return contexts
System Prompt
Using the same command, create the system prompt as follows, and again, make sure you have a prompts/ directory created to store all your prompts:
	
$ agentc add --record-kind jinja_prompt -o prompts                       
Now building a new tool / prompt file. The output will be saved to: prompts
Type: jinja_prompt
Name: sampleapp_system_instructions
Description: System instructions for AI assistant responsible for collaborating with other assistants
Jinja prompt written to: prompts/sampleapp_system_instructions.jinja

Make sure you fill in the prompt and it looks like this (.jinja file) :

---
record_kind: jinja_prompt
name: sampleapp_system_instructions
description: >
    System instructions for AI assistant responsible for collaborating with other assistants
---
You are a helpful AI assistant, collaborating with other assistants.
Use the provided tools to progress towards answering the question.
If you are unable to fully answer, that's OK, another assistant with different tools
will help where you left off. Execute what you can to make progress.
If you or any of the other assistants have the final answer or deliverable,
prefix your response with FINAL ANSWER so the team knows to stop.
{{ suffix }}

Create the Catalog
Now that our tools and prompts are ready, let's make sure Agentc can start using them. Commit your changes to Git and run the index command to index through the tools/ and prompts/ directories and generate the catalogs.

% agentc index tools/ prompts/

This command will go through each tool and prompt present in respective directories and generate the tool and prompt catalogs:

.agent-catalog/prompt-catalog.json

{
  "embedding_model": "sentence-transformers/all-MiniLM-L12-v2",
  "items": [
  {
    "description": "System instructions for AI assistant responsible for collaborating with other assistants\n",
    "embedding": [embeddings generated using the embedding model],
    "identifier": "prompts/system_instructions.jinja:sampleapp_system_instructions:git_5ba457ee5fd1daa8fba15390b97b360497ba09c0",
    "name": "sampleapp_system_instructions",
    "prompt": "\nYou are a helpful AI assistant, collaborating with other assistants.\nUse the provided tools to progress towards answering the question.\nIf you are unable to fully answer, that's OK, another assistant with different tools\nwill help where you left off. Execute what you can to make progress.\nIf you or any of the other assistants have the final answer or deliverable,\nprefix your response with FINAL ANSWER so the team knows to stop.\n{{ suffix }}",
    "record_kind": "jinja_prompt",
    "source": "prompts/system_instructions.jinja",
    "version":
    {
      "identifier": "5ba457ee5fd1daa8fba15390b97b360497ba09c0",
      "is_dirty": false,
      "timestamp": "2025-01-21T09:32:23.666349Z"
    }
  }],
  "kind": "prompt",
  "library_version": "v0.0.0-0-g0",
  "schema_version": "0.0.0",
  "source_dirs": ["tools", "prompts"],
  "version":
  {
    "identifier": "0431315f8479279b6609082cba237ad161dccee6",
    "is_dirty": false,
    "timestamp": "2025-01-21T09:32:17.898876Z"
  }
}

Similarly, you will also see .agent-catalog/tool-catalog.json with the same format. To explore more on the index command please run the following:

% agentc index --help
Usage: agentc index [OPTIONS] [SOURCE_DIRS]...

  Walk the source directory trees (SOURCE_DIRS) to index source files into the local catalog. Source files that will be scanned include *.py, *.sqlpp, *.yaml, etc.

Options:
  --prompts / --no-prompts     Flag to (avoid) ignoring prompts when indexing source files into the local catalog.  [default: prompts]
  --tools / --no-tools         Flag to (avoid) ignoring tools when indexing source files into the local catalog.  [default: tools]
  -em, --embedding-model TEXT  Embedding model used when indexing source files into the local catalog.  [default: sentence-transformers/all-MiniLM-L12-v2]
  --dry-run                    Flag to prevent catalog changes.
  --help                       Show this message and exit.

Publish the catalogs
Now that our catalogs are generated and available locally, let's publish and store them in our Couchbase cluster. Make sure you have a .env created with the following values:

.env
CB_CONN_STRING=couchbase://localhost
CB_USERNAME=your_uname
CB_PASSWORD=your_password

AGENT_CATALOG_CONN_STRING=couchbase://localhost
AGENT_CATALOG_USERNAME=your_uname
AGENT_CATALOG_PASSWORD=your_password
AGENT_CATALOG_BUCKET=your_bucket_name

OPENAI_API_KEY=...
SERPAPI_KEY=...

Run the publish command next which will upsert all the catalog items in your bucket under agent_catalog scope. It will also create various GSI and Vector indexes on the _catalog collections.

$ agentc publish --bucket <you_bucket_name>
// Run the following to explore more options
$ agentc publish --help                                         
Usage: agentc publish [OPTIONS] [[tool|prompt|log]]...

  Upload the local catalog and/or logs to a Couchbase instance. By default, only tools and prompts are published unless log is specified.

Options:
  --bucket TEXT                   Name of the Couchbase bucket to publish to.
  -an, --annotations <TEXT TEXT>...
                                  Snapshot level annotations to be added while publishing catalogs.
  --help                          Show this message and exit.

Now, the tools and prompts are available locally as well as remotely.

Using the Provider
Agentc allows you to interact with your tools and prompts through a Provider class. Once tools and prompts are defined in the way Agentc expects them, we can use this class to start querying them as follows:

import agentc
import dotenv
from langchain_core.tools import tool
from pydantic import SecretStr

dotenv.load_dotenv()
provider = agentc.Provider(
    decorator=lambda t: tool(t.func),
    secrets={
        "CB_CONN_STRING": SecretStr(os.getenv("CB_CONN_STRING")),
        "CB_USERNAME": SecretStr(os.getenv("CB_USERNAME")),
        "CB_PASSWORD": SecretStr(os.getenv("CB_PASSWORD")),
    },
)
Let’s update the research and charts agent to use the provider:

research_agent = create_react_agent(
    model=audit(llm, session="doc", auditor=auditor),
    tools=provider.get_item(name="web_search", item_type="tool"),
    # or use the vector search feature
    # tools=provider.get_item(query="tool to make google search", item_type="tool"),
    state_modifier=provider.get_item(name="sampleapp_system_instructions", item_type="prompt").prompt.render(
        suffix="You can only do research. You are working with a chart generator colleague."
    )
)

chart_agent = create_react_agent(
    model=audit(llm, session="doc", auditor=auditor),
    tools=provider.get_item(name="repl_tool", item_type="tool"),
    # or use the vector search feature
    # tools=provider.get_item(query="tool to visualise",item_type="tool"),
    state_modifier=provider.get_item(name="sampleapp_system_instructions", item_type="prompt").prompt.render(
        suffix="You can only generate charts. You are working with a researcher colleague."
    ),
)

This allows us to pass variable input to the prompt while rendering (thanks to the jinja template!) and call appropriate tools as per their use cases.


…

Using the Auditor
Assume a situation where an agentic application developed by an e-commerce start-up promises a 100% discount on some item to their customer and now the customer is here to get their free item! How would the developers know what went wrong?

With Agentc’s Auditor class, each session’s logs can be recorded and kept so that in such cases, they can trace the issue back to the particular agent or session, explore if a tool or prompt was faulty, and finally fix the issue and prevent it from happening in the future!

Let’s take a look at how we can integrate the same in our current example and check out a few logs. 

# Initialising the auditor to track the agents' thought processes
auditor = agentc.Auditor(agent_name="Sample Research Agent")

The same agents now also use a new model instead of our pre-defined LLM directly. This would track important values such as the session id which would be helpful later. The session id in the following example is “doc”.

research_agent = create_react_agent(
    model=audit(llm, session="doc", auditor=auditor),
    tools=provider.get_item(name="web_search", item_type="tool"),
    # or use the vector search feature
    # tools=provider.get_item(query="tool to make google search", item_type="tool"),
    state_modifier=provider.get_item(name="sampleapp_system_instructions", item_type="prompt").prompt.render(
        suffix="You can only do research. You are working with a chart generator colleague."
    )
)

chart_agent = create_react_agent(
    model=audit(llm, session="doc", auditor=auditor),
    tools=provider.get_item(name="repl_tool", item_type="tool"),
    # or use the vector search feature
    # tools=provider.get_item(query="tool to visualise",item_type="tool"),
    state_modifier=provider.get_item(name="sampleapp_system_instructions", item_type="prompt").prompt.render(
        suffix="You can only generate charts. You are working with a researcher colleague."
    ),
)

The auditor stores each message in a conversation between the User and the Agent. To view these messages, you can either access .agent-activity/llm-activity.log file or check out the keyspace <AGENT_CATALOG_BUCKET>.agent_activity.raw_logs (bucket. scope.collection) in your Couchbase cluster.

To view the messages for each session, head over to your Analytics workbench and execute the following SQL++ query which calls a view created by Agentc at the time of initializing the Auditor. 

select a.* from Sessions as a;

This will return a set of sessions, each uniquely defined by the session id (sid) you provided (doc in this example), messages (msgs) in order of occurrence, a unique identifier (vid) for the log, and the start time (start_t) of the session. Take a look at a sample session log from our example below:

{
    "vid": {
      "identifier": "0431315f8479279b6609082cba237ad161dccee6",
      "timestamp": "2025-01-21T09:32:17.898876Z"
    },
    "msgs": [
      {
        "msg_num": 1,
        "content": {
          "dump": {
            "lc": 1,
            "type": "constructor",
            "id": [
              "langchain",
              "schema",
              "messages",
              "HumanMessage"
            ],
            "kwargs": {
              "content": "First, get the UK's GDP over the past 5 years, then give a brief summary of it along with a pie chart. Once you make the chart, finish.",
              "type": "human",
              "id": "e604a71b-7f18-492f-9826-77a1d76d3e82"
            }
          },
          "content": "First, get the UK's GDP over the past 5 years, then give a brief summary of it along with a pie chart. Once you make the chart, finish."
        },
        "timestamp": "2025-01-21T15:28:58.912680+05:30",
        "kind": "human"
      },
      {
        "msg_num": 2,
        "content": {
          "dump": {
            "lc": 1,
            "type": "constructor",
            "id": [
              "langchain",
              "schema",
              "messages",
              "AIMessage"
            ],
            "kwargs": {
              "content": "",
              "additional_kwargs": {
                "tool_calls": [...],
                "refusal": null
              },
              "type": "ai",
              "tool_calls": [
                {
                  "name": "web_search",
                  "args": {
                    "query": "UK GDP data for the past 5 years"
                  },
                  "id": "call_CNQogTyj8Q5VrTfezYqBVtYD",
                  "type": "tool_call"
                }
              ],
              "usage_metadata": {...},
              "invalid_tool_calls": []
            }
          },
          "tool_calls": "web_search({'query': 'UK GDP data for the past 5 years'})"
        },
        "timestamp": "2025-01-21T15:29:00.059097+05:30",
        "kind": "llm"
      },
....more messages
    "sid": "doc",
    "start_t": "2025-01-21T15:28:58.912680+05:30"
  }

 
There are more such views automatically created at the time of initializing the auditor and can be found in the documentation here. You can run each of them in your Couchbase Analytics workbench similar to the Sessions() view - by replacing Sessions() with the name of the view you wish to explore in the SQL++ query.

Managing Catalogs
Once you have run your application with the current tools and prompts, you may wish to add more tools/prompts and work with a more complex workflow. You may add a new set of nodes that may use new tools and prompts. Let’s see how you can achieve this with the versioning capabilities of Agentc.

Agentc uses Git commit hashes to version your catalog. When you run agentc publish tools prompts, the catalogs created have a catalog identifier which is the commit hash as per your last commit. In case you wish to add to your catalog, you must follow the steps below:
Create tools/prompts - create the tools and/or prompts as displayed in the previous sections using the agentc add command
Test tools - use the agentc execute command to test individual tools before using them in your application
Commit your changes - run git commit -m “<message>” and ensure your changes are committed to Git to obtain the latest hash
Index tools/prompts - update the catalog by indexing the new tools/prompts into the existing local catalog. Run the agentc index command for this as shown in this section
Publish your catalog - to ensure you don't just locally have this updated catalog, make sure to use agentc publish and upsert the new items into your Couchbase cluster as mentioned in this section
Decide on find query - before using the new tools/prompts in your application directly, you can run agentc find with --query option to conclude on a suitable string to use to ensure the most relevant tool/prompt is returned.
Use provider - once you have the final use case string to get the right tool/prompt for your agent, use provider.get_item() to give the appropriate item to the agent.
Run your application - run the application to test these new changes!

Additional Commands
Along with the commands we have explored so far, you can try out these additional commands depending on your needs:
`clean` - Delete all or specific (catalog and/or activity) agent related files / collections. This is a destructive action so please be careful before using it!
Eg: agentc clean local catalog will remove .agent-catalog folder from your system. 
`env` - Return all agentc related environment and configuration parameters as a JSON object. This includes env variables such as AGENT_CATALOG_BUCKET, etc.
`status` -  Shows the status of the local catalog by default and DB using -db flag. Enable the --compare flag to compare the last published catalog with the local catalog.
`version` - Shows the current version of agentc. As of now, you are on version v0.0.0-0-g0.
`--help` - Run agentc --help explore the various commands agentc has to offer. Run agentc <command> --help to view all options <command> offers and its usage.
Common Questions/Issues
Dependency clash in PyTorch versions 

While installing Agentc, you may face a dependency clash between the PyTorch version installed globally in your system and the PyTorch version being installed by the Sentence transformers library in Agentc. This likely happens when the globally installed PyTorch is of a different version compared to what Agentc requires.
You can resolve this by using Anaconda or any other virtual environment manager instead of the built-in Python venv manager. We have found Anaconda to be better in terms of isolating project dependencies. In case this does not solve the issue and you are on an older OS, consider using a Virtual Machine to run your application.

Can I use any other agent framework?


Although we have validated with LangGraph, LangChain, and Controlflow, you can use any other agent framework that requires your tools to be in a Python-esque format. Agentc works one layer above agent frameworks and aims to enhance the users’ experience while building their agentic applications.

Can I use any LLM?


Agentc is not in the path of any LLM flow, so you could technically use any LLM that supports multi-function tool calling and is supported by your agent framework.
Glossary

Figure 3: Langgraph Nodes, States, and Edges. Source: [1]

Agents 
The entity that is responsible for interacting with the user and orchestrating tasks. 
Uses prompts for instructions and tools to execute actions.
State 
Represents the context or memory that is maintained and updated as the computation progresses. 
Stored as a dictionary.
Ensures that each step in the graph can access relevant information from previous steps, allowing for dynamic decision-making based on accumulated data throughout the process.
Nodes 
Represent individual computation steps or functions. 
Each node performs a specific task, such as processing inputs, making decisions, or interacting with external systems. 
Nodes can be customized to execute a wide range of operations in the flow.
Edges 
Connect nodes within the graph, defining the flow of computation from one step to the next. 
They support conditional logic, allowing the path of execution to change based on the current state and facilitate the movement of data and control between nodes, enabling complex, multi-step workflows.
Tools 
Functions with fixed inputs and outputs based on unique logic.
Used by agents to get tasks done/for computation.
Prompts 
Instructions that are provided to agents telling them their aim in natural language.
Dictate their decision making process the way you want it to.

In every agentic workflow, there will be an Agent represented by Nodes (in a graph) with its current memory stored in and tracked by its State. The state would change every time control flows from one node to another based on conditions represented by Edges. The agent will also use Tools to compute values and Prompts that dictate the agent’s aim.
References
[1] How to build AI Agents with Langgraph - a step-by-step guide (Medium article) 
[2] Agentc sphinx documentation (Link)
[3] Agentc Source Code (GitHub)
[4] User Guide - AI Services (Google Doc)
[5] Langgraph MAS Source code (GitHub)
