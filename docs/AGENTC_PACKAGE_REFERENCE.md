Package Documentation
agentc Package
pydantic settings agentc.catalog.Catalog[source]
A provider of indexed "agent building blocks" (e.g., tools, prompts, spans...).

Class Description
A Catalog instance can be configured in three ways (listed in order of precedence):

Directly (as arguments to the constructor).

Via the environment (though environment variables).

Via a .env configuration file.

In most cases, you'll want to configure your catalog via a .env file. This style of configuration means you can instantiate a Catalog instance as such:

import agentc
catalog = agentc.Catalog()
Some custom configurations can only be specified via the constructor (e.g., secrets). For example, if your secrets are managed by some external service (defined below as my_secrets_manager), you can specify them as such:

import agentc
catalog = agentc.Catalog(secrets={
    "CB_CONN_STRING": os.getenv("CB_CONN_STRING"),
    "CB_USERNAME": os.getenv("CB_USERNAME"),
    "CB_PASSWORD": my_secrets_manager.get("THE_CB_PASSWORD"),
    "CB_CERTIFICATE": my_secrets_manager.get("PATH_TO_CERT"),
})
Fields:
Span(
name,
session=None,
state=None,
iterable=False,
blacklist=None,
**kwargs
)[source]
A factory method to initialize a Span (more specifically, a GlobalSpan) instance.

Parameters:
name (str) -- Name to bind to each message logged within this span.

session (str) -- The run that this tree of spans is associated with. By default, this is a UUID.

state (Any) -- A JSON-serializable object that will be logged on entering and exiting this span.

iterable (bool) -- Whether this new span should be iterable. By default, this is False.

blacklist (set[Kind]) -- A set of content types to skip logging. By default, there is no blacklist.

kwargs -- Additional keyword arguments to pass to the Span constructor.

Return type:
Span

find(
kind,
query=None,
name=None,
annotations=None,
catalog_id='__LATEST__',
limit=1
)[source]
Return a list of tools or prompts based on the specified search criteria.

Method Description
This method is meant to act as the programmatic equivalent of the agentc find command. Whether (or not) the results are fetched from the local catalog or the remote catalog depends on the configuration of this agentc_core.catalog.Catalog instance.

For example, to find a tool named "get_sentiment_of_text", you would author:

results = catalog.find(kind="tool", name="get_sentiment_of_text")
sentiment_score = results[0].func("I love this product!")
To find a prompt named "summarize_article_instructions", you would author:

results = catalog.find(kind="prompt", name="summarize_article_instructions")
prompt_for_agent = summarize_article_instructions.content
Parameters:
kind (Literal['tool', 'prompt']) -- The type of item to search for, either 'tool' or 'prompt'.

query (str) -- A query string (natural language) to search the catalog with.

name (str) -- The specific name of the catalog entry to search for.

annotations (str) -- An annotation query string in the form of KEY="VALUE" (AND|OR KEY="VALUE")*.

catalog_id (str) -- The snapshot version to find the tools for. By default, we use the latest snapshot.

limit (int | None) -- The maximum number of results to return (ignored if name is specified).

Returns:
One of the following:

None if no results are found by name.

"tools" if kind is "tool" (see find_tools() for details).

"prompts" if kind is "prompt" (see find_prompts() for details).

Return type:
list[ToolResult] | list[PromptResult] | ToolResult | PromptResult | None

find_prompts(
query=None,
name=None,
annotations=None,
catalog_id='__LATEST__',
limit=1
)[source]
Return a list of prompts based on the specified search criteria.

Parameters:
query (str) -- A query string (natural language) to search the catalog with.

name (str) -- The specific name of the catalog entry to search for.

annotations (str) -- An annotation query string in the form of KEY="VALUE" (AND|OR KEY="VALUE")*.

catalog_id (str) -- The snapshot version to find the tools for. By default, we use the latest snapshot.

limit (int | None) -- The maximum number of results to return (ignored if name is specified).

Returns:
A list of Prompt instances, with the following attributes:

content (str | dict): The content to be served to the model.

tools (list): The list containing the tool functions associated with prompt.

output (dict): The output type of the prompt, if it exists.

meta (RecordDescriptor): The metadata associated with the prompt.

Return type:
list[PromptResult] | PromptResult | None

find_tools(
query=None,
name=None,
annotations=None,
catalog_id='__LATEST__',
limit=1
)[source]
Return a list of tools based on the specified search criteria.

Parameters:
query (str) -- A query string (natural language) to search the catalog with.

name (str) -- The specific name of the catalog entry to search for.

annotations (str) -- An annotation query string in the form of KEY="VALUE" (AND|OR KEY="VALUE")*.

catalog_id (str) -- The snapshot version to find the tools for. By default, we use the latest snapshot.

limit (int | None) -- The maximum number of results to return (ignored if name is specified).

Returns:
By default, a list of Tool instances with the following attributes:

func (typing.Callable): A Python callable representing the function.

meta (RecordDescriptor): The metadata associated with the tool.

input (dict): The argument schema (in JSON schema) associated with the tool.

If a tool_decorator is present, this method will return a list of objects decorated accordingly.

Return type:
list[ToolResult] | ToolResult | None

property version: VersionDescriptor
The version of the catalog currently being served (i.e., the latest version).

Returns:
An agentc_core.version.VersionDescriptor instance.

pydantic model agentc.span.Span[source]
A structured logging context for agent activity.

Class Description
A Span instance belongs to a tree of other Span instances, whose root is a GlobalSpan instance that is constructed using the Catalog.Span() method.

Attention

Spans should never be created directly (via constructor), as logs generated by the span must always be associated with a catalog version and some application structure.

Below we illustrate how a tree of Span instances is created:

import agentc
catalog = agentc.Catalog()
root_span = catalog.Span(name="root")
child_1_span = root_span.new(name="child_1")
child_2_span = root_span.new(name="child_2")
In practice, you'll likely use different spans for different agents and/or different tasks. Below we give a small LangGraph example using spans for different agents:

import agentc
import langgraph.graph

catalog = agentc.Catalog()
root_span = catalog.Span(name="flight_planner")

def front_desk_agent(...):
    with root_span.new(name="front_desk_agent") as front_desk_span:
        ...

def route_finding_agent(...):
    with root_span.new(name="route_finding_agent") as route_finding_span:
        ...

workflow = langgraph.graph.StateGraph()
workflow.add_node("front_desk_agent", front_desk_agent)
workflow.add_node("route_finding_agent", route_finding_agent)
workflow.set_entry_point("front_desk_agent")
workflow.add_edge("front_desk_agent", "route_finding_agent")
...
Fields:
blacklist (set[agentc_core.activity.models.content.Kind])

iterable (bool | None)

kwargs (dict[str, Any] | None)

logger (Callable[[...], agentc_core.activity.models.log.Log])

name (str)

parent (agentc_core.activity.span.Span)

state (Any)

field blacklist: set[Kind] [Optional]
List of content types to filter.

Validated by:
_initialize_iterable_logger

field iterable: bool | None = False
Flag to indicate whether or not this span should be iterable.

Validated by:
_initialize_iterable_logger

field kwargs: dict[str, Any] | None = None
Annotations to apply to all messages logged within this span.

Validated by:
_initialize_iterable_logger

field name: str [Required]
Name to bind to each message logged within this span.

Validated by:
_initialize_iterable_logger

field parent: Span = None
Parent span of this span (i.e., the span that had new() called on it).

Validated by:
_initialize_iterable_logger

field state: Any = None
A JSON-serializable object that will be logged on entering and exiting this span.

Validated by:
_initialize_iterable_logger

pydantic model Identifier[source]
The unique identifier for a Span.

Class Description
A Span is uniquely identified by two parts:

an application-defined multipart name and...

a session identifier unique to each run of the application.

Fields:
name (list[str])

session (str)

field name: list[str] [Required]
The name of the Span.

Names are built up from the root of the span tree to the leaf, thus the first element of name is the name of the root and the last element is the name of the current span (i.e., the leaf).

field session: str [Required]
The session identifier of the Span.

Sessions must be unique to each run of the application. By default, we generate these as UUIDs (see GlobalSpan.session).

enter()[source]
Record a BeginContent log entry for this span.

Method Description
The enter() method is to denote the start of the span (optionally logging the incoming state if specified). This method is also called when entering the span using the with statement. In the example below, enter() is called (implicitly).

import agentc

catalog = agentc.Catalog()
incoming_state = {"flights": []}
with catalog.Span(name="flight_planner", state=incoming_state) as span:
    flight_planner_implementation()
On entering the context, one log is generated possessing the content below:

{ "kind": "begin", "state": {"flights": []} }
Return type:
Self

exit()[source]
Record a EndContent log entry for this span.

Method Description
The exit() method is to denote the end of the span (optionally logging the outgoing state if specified). This method is also called when exiting the span using the with statement successfully. In the example below, exit() is called (implicitly).

import agentc

catalog = agentc.Catalog()
incoming_state = {"flights": []}
with catalog.Span(name="flight_planner", state=incoming_state) as span:
    ... = flight_planner_implementation(...)
    incoming_state["flights"] = [{"flight_number": "AA123", "status": "on_time"}]
On exiting the context, one log is generated possessing the content below:

{ "kind": "end", "state": {"flights": [{"flight_number": "AA123", "status": "on_time"}]} }
Note

The state of the span must be JSON-serializable and must be mutated in-place. If you are working with immutable state objects, you must set the state attribute before exiting the span (i.e., before the with statement exits or with exit() explicitly).

import agentc

catalog = agentc.Catalog()
immutable_incoming_state = {"flights": []}
with catalog.Span(name="flight_planner", state=incoming_state) as span:
    ... = flight_planner_implementation(...)
    span.state = {"flights": [{"flight_number": "AA123", "status": "on_time"}]}
log(
content,
**kwargs
)[source]
Accept some content (with optional annotations specified by kwargs) and generate a corresponding log entry.

Method Description
The heart of the Span class is the log() method. This method is used to log events that occur within the span. Users can capture events that occur in popular frameworks like LangChain and LlamaIndex using our helper packages (see agentc_langchain, agentc_langgraph, and agentc_llamaindex) but must use those packages in conjunction with this log() method to capture the full breadth of their application's activity. See here for a list of all available log content types.

Users can also use Python's [] syntax to write arbitrary JSON-serializable content as a key-value (KeyValueContent) pair. This is useful for logging arbitrary data like metrics during evaluations. In the example below, we illustrate an example of a system-wide evaluation suite that uses this [] syntax:

import my_agent_app
import my_output_evaluator
import agentc

catalog = agentc.Catalog()
evaluation_span = catalog.Span(name="evaluation_suite")
with open("my-evaluation-suite.json") as fp:
    for i, line in enumerate(fp):
        with evaluation_span.new(name=f"evaluation{i}") as span:
            output = my_agent_app(span)
            span["positive_sentiment"] = my_output_evaluator.positive(output)
            span.log(
                content={
                    "kind": "key-value",
                    "key": "negative_sentiment",
                    "value": my_output_evaluator.negative(output)
                    },
                alpha="SDGD"
            )
All keywords passed to the log() method will be applied as annotations to the log entry. In the example above, the alpha annotation is applied only to the second log entry. For span-wide annotations, use the kwargs attribute on new().

Parameters:
content (SystemContent | ToolCallContent | ToolResultContent | ChatCompletionContent | RequestHeaderContent | UserContent | AssistantContent | BeginContent | EndContent | EdgeContent | KeyValueContent) -- The content to log.

kwargs -- Additional annotations to apply to the log.

logs()[source]
Return the logs generated by the tree of Span nodes rooted from this Span instance.

Method Description
The logs() method returns an iterable of all logs generated within the span. This method is also called (implicitly) when iterating over the span (e.g., using a for loop). To use this method, you must set the iterable attribute to True when instantiating the span:

import agentc

catalog = agentc.Catalog()
span = catalog.Span(name="flight_planner", iterable=True)
for log in span:
    match log.content.kind:
        case "begin":
            ...
Tip

Generally, this method should only be used for debugging purposes. This method will keep all logs generated by the span in memory. To perform efficient aggregate analysis of your logs, consider querying the agent_activity.logs collection in your Couchbase cluster using SQL++ instead.

Return type:
Iterable[Log]

new(
name,
state=None,
iterable=False,
blacklist=None,
**kwargs
)[source]
Create a new span under the current Span.

Method Description
Spans require a name and a session (see identifier). Aside from name, state, and iterable, you can also pass additional keywords that will be applied as annotations to each log() call within a span. As an example, the following code illustrates the use of kwargs to add a span-wide "alpha" annotation:

import agentc
catalog = agentc.Catalog()
root_span = catalog.Span(name="flight_planner")
with root_span.new(name="find_airports_task", alpha="SDGD") as child_span:
    child_span.log(content=agentc.span.UserContent(value="Hello, world!", "beta": "412d"))
The example code above will generate the three logs below (for brevity, we only show the content and
annotations fields):

{ "content": { "kind": "begin" }, "annotations": { "alpha": "SDGD"} }
{ "content": { "kind": "user", "value": "Hello, world!" },
  "annotations": { "alpha": "SDGD", "beta": "412d" } }
{ "content" : { "kind": "end" }, "annotations": { "alpha": "SDGD" } }
Parameters:
name (str) -- The name of the span.

state (Any) -- The starting state of the span. This will be recorded upon entering and exiting the span.

iterable (bool) -- Whether this new span should be iterable. By default, this is False.

blacklist (set[Kind]) -- A set of content types to skip logging. By default, there is no blacklist.

kwargs -- Additional annotations to apply to the span.

Returns:
A new Span instance.

Return type:
Span

property identifier: Identifier
A unique identifier for this span.

Integration Packages
LangChain
class agentc_langchain.chat.Callback(
span,
tools=None,
output=None
)[source]
A callback that will log all LLM calls using the given span as the root.

Class Description
This class is a callback that will log all LLM calls using the given span as the root. This class will record all messages used to generated ChatCompletionContent and ToolCallContent. ToolResultContent is not logged by this class, as it is not generated by a BaseChatModel instance.

Below, we illustrate a minimal example of how to use this class:

import langchain_openai
import langchain_core.messages
import agentc_langchain.chat
import agentc

# Create a span to bind to the chat model messages.
catalog = agentc.Catalog()
root_span = catalog.Span(name="root_span")

# Create a chat model.
chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o", callbacks=[])

# Create a callback with the appropriate span, and attach it to the chat model.
my_agent_span = root_span.new(name="my_agent")
callback = agentc_langchain.chat.Callback(span=my_agent_span)
chat_model.callbacks.append(callback)
result = chat_model.invoke(messages=[
    langchain_core.messages.SystemMessage(content="Hello, world!")
])
To record the exact tools and output used by the chat model, you can pass in the tools and output to the agentc_langchain.chat.Callback constructor. For example:

import langchain_openai
import langchain_core.messages
import langchain_core.tools
import agentc_langchain.chat
import agentc

# Create a span to bind to the chat model messages.
catalog = agentc.Catalog()
root_span = catalog.Span(name="root_span")

# Create a chat model.
chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o", callbacks=[])

# Grab the correct tools and output from the catalog.
my_agent_prompt = catalog.find("prompt", name="my_agent")
my_agent_tools = [
    langchain_core.tools.StructuredTool.from_function(tool.func) for tool in my_agent_prompt.tools
]
my_agent_output = my_agent_prompt.output

# Create a callback with the appropriate span, tools, and output, and attach it to the chat model.
my_agent_span = root_span.new(name="my_agent")
callback = agentc_langchain.chat.Callback(
    span=my_agent_span,
    tools=my_agent_tools,
    output=my_agent_output
)
chat_model.callbacks.append(callback)
result = chat_model.with_structured_output(my_agent_output).invoke(messages=[
    langchain_core.messages.SystemMessage(content=my_agent_prompt.content)
])
Parameters:
span (Span)

tools (list[Tool])

output (tuple | dict)

agentc_langchain.cache.cache(
chat_model,
kind,
embeddings=None,
options=None,
**kwargs
)[source]
A function to attach a Couchbase-backed exact or semantic cache to a ChatModel.

Function Description
This function is used to set the .cache property of LangChain ChatModel instances. For all options related to this Couchbase-backed cache, see CacheOptions.

Below, we illustrate a minimal working example of how to use this function to store and retrieve LLM responses via exact prompt matching:

import langchain_openai
import agentc_langchain.cache

chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o")
caching_chat_model = agentc_langchain.cache.cache(
    chat_model=chat_model,
    kind="exact",
    create_if_not_exists=True
)

# Response #2 is served from the cache.
response_1 = caching_chat_model.invoke("Hello there!")
response_2 = caching_chat_model.invoke("Hello there!")
To use this function to store and retrieve LLM responses via semantic similarity, use the kind="semantic" argument with an langchain_core.embeddings.Embeddings instance:

import langchain_openai
import agentc_langchain.cache

chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o")
embeddings = langchain_openai.OpenAIEmbeddings(model="text-embedding-3-small")
caching_chat_model = agentc_langchain.cache.cache(
    chat_model=chat_model,
    kind="semantic",
    embeddings=embeddings,
    create_if_not_exists=True
)

# Response #2 is served from the cache.
response_1 = caching_chat_model.invoke("Hello there!")
response_2 = caching_chat_model.invoke("Hello there!!")
By default, the Couchbase initialization of the cache is separate from the cache's usage (storage and retrieval). To explicitly initialize the cache yourself, use the initialize() method.

See also

This method uses the langchain_couchbase.cache.CouchbaseCache and langchain_couchbase.cache.CouchbaseSemanticCache classes from the langchain_couchbase package. See here for more details.

Parameters:
chat_model (BaseChatModel) -- The LangChain chat model to cache responses for.

kind (Literal['exact', 'semantic']) -- The type of cache to attach to the chat model.

embeddings (Embeddings) -- The embeddings to use when attaching a 'semantic' cache to the chat model.

options (CacheOptions) -- The options to use when attaching a cache to the chat model.

kwargs -- Keyword arguments to be forwarded to a CacheOptions constructor (ignored if options is present).

Returns:
The same LangChain chat model that was passed in, but with a cache attached.

Return type:
BaseChatModel

agentc_langchain.cache.initialize(
kind,
options=None,
embeddings=None,
**kwargs
)[source]
A function to create the collections and/or indexes required to use the cache() function.

Function Description
This function is a helper function for creating the default collection (and index, in the case of kind="semantic") required for the cache() function. Below, we give a minimal working example of how to use this function to create a semantic cache backed by Couchbase.

import langchain_openai
import agentc_langchain.cache

embeddings = langchain_openai.OpenAIEmbeddings(model="text-embedding-3-small")
agentc_langchain.cache.initialize(
    kind="semantic",
    embeddings=embeddings
)

chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o")
caching_chat_model = agentc_langchain.cache.cache(
    chat_model=chat_model,
    kind="semantic",
    embeddings=embeddings,
)

# Response #2 is served from the cache.
response_1 = caching_chat_model.invoke("Hello there!")
response_2 = caching_chat_model.invoke("Hello there!!")
Parameters:
kind (Literal['exact', 'semantic']) -- The type of cache to attach to the chat model.

embeddings (Embeddings) -- The embeddings to use when attaching a 'semantic' cache to the chat model.

options (CacheOptions) -- The options to use when attaching a cache to the chat model.

kwargs -- Keyword arguments to be forwarded to a CacheOptions constructor (ignored if options is present).

Return type:
None

pydantic settings agentc_langchain.cache.CacheOptions[source]
Config:
env_prefix: str = AGENT_CATALOG_LANGCHAIN_CACHE_

env_file: str = .env

Fields:
bucket (str | None)

collection (str | None)

conn_root_certificate (str | pathlib.Path | None)

conn_string (str | None)

create_if_not_exists (bool | None)

ddl_retry_attempts (int | None)

ddl_retry_wait_seconds (float | None)

index_name (str | None)

password (pydantic.types.SecretStr | None)

scope (str | None)

score_threshold (float | None)

ttl (datetime.timedelta | None)

username (str | None)

field bucket: str | None = None
The name of the Couchbase bucket hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field collection: str | None = 'langchain_llm_cache'
The name of the Couchbase collection hosting the cache.

This field is optional and defaults to langchain_llm_cache.

Validated by:
_pull_cluster_from_agent_catalog

field conn_root_certificate: str | Path | None = None
Path to the root certificate file for the Couchbase cluster.

This field is optional and only required if the Couchbase cluster is using a self-signed certificate.

Validated by:
_pull_cluster_from_agent_catalog

field conn_string: str | None = None
The connection string to the Couchbase cluster hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field create_if_not_exists: bool | None = False
Create the required collections and/or indexes if they do not exist.

When raised (i.e., this value is set to True), the collections and indexes will be created if they do not exist. Lower this flag (set this to False) to instead raise an error if the collections & indexes do not exist.

Validated by:
_pull_cluster_from_agent_catalog

field ddl_retry_attempts: int | None = 3
Maximum number of attempts to retry DDL operations.

This value is only used on setup (i.e., the first time the cache is requested). If the number of attempts is exceeded, the command will fail. By default, this value is 3 attempts.

Validated by:
_pull_cluster_from_agent_catalog

field ddl_retry_wait_seconds: float | None = 5
Wait time (in seconds) between DDL operation retries.

This value is only used on setup (i.e., the first time the cache is requested). By default, this value is 5 seconds.

Validated by:
_pull_cluster_from_agent_catalog

field index_name: str | None = 'langchain_llm_cache_index'
The name of the Couchbase FTS index used to query the cache.

This field will only be used if the cache is of type semantic. If the cache is of type semantic and this field is not specified, this field defaults to langchain_llm_cache_index.

Validated by:
_pull_cluster_from_agent_catalog

field password: SecretStr | None = None
Password associated with the Couchbase instance hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field scope: str | None = 'agent_activity'
The name of the Couchbase scope hosting the cache.

This field is optional and defaults to agent_activity.

Validated by:
_pull_cluster_from_agent_catalog

field score_threshold: float | None = 0.8
The score threshold used to quantify what constitutes as a "good" match.

This field will only be used if the cache is of type semantic. If the cache is of type semantic and this field is not specified, this field defaults to 0.8.

Validated by:
_pull_cluster_from_agent_catalog

field ttl: timedelta | None = None
The time-to-live (TTL) for the cache.

When specified, the cached documents will be automatically removed after the specified duration. This field is optional and defaults to None.

Validated by:
_pull_cluster_from_agent_catalog

field username: str | None = None
Username associated with the Couchbase instance hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

LangGraph
class agentc_langgraph.tool.ToolNode(
span,
*args,
**kwargs
)[source]
A tool node that logs tool results to a span.

Class Description
This class will record the results of each tool invocation to the span that is passed to it (ultimately generating ToolResultContent log entries). This class does not log tool calls (i.e., ToolCallContent log entries) as these are typically logged with ChatCompletionContent log entries.

Below, we illustrate a minimal working example of how to use this class with agentc_langchain.chat.Callback to record ChatCompletionContent log entries, ToolCallContent log entries, and ToolResultContent log entries.

import langchain_openai
import langchain_core.tools
import langgraph.prebuilt
import agentc_langchain.chat
import agentc_langgraph
import agentc

# Create a span to bind to the chat model messages.
catalog = agentc.Catalog()
root_span = catalog.Span(name="root_span")

# Create a chat model.
chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o", callbacks=[])

# Create a callback with the appropriate span, and attach it to the chat model.
my_agent_span = root_span.new(name="my_agent")
callback = agentc_langchain.chat.Callback(span=my_agent_span)
chat_model.callbacks.append(callback)

# Grab the correct tools and output from the catalog.
my_agent_prompt = catalog.find("prompt", name="my_agent")
my_agent_tools = agentc_langgraph.tool.ToolNode(
    span=my_agent_span,
    tools=[
        langchain_core.tools.tool(
            tool.func,
            args_schema=tool.input,
        ) for tool in my_agent_prompt.tools
    ]
)
my_agent_output = my_agent_prompt.output

# Finally, build your agent.
my_agent = langgraph.prebuilt.create_react_agent(
    model=chat_model,
    tools=my_agent_tools,
    prompt=my_agent_prompt,
    response_format=my_agent_output
)
Note

For all constructor parameters, see the documentation for langgraph.prebuilt.ToolNode here.

Parameters:
span (Span)

class agentc_langgraph.agent.ReActAgent(
chat_model,
catalog,
span,
prompt_name=None
)[source]
A helper ReAct agent base class that integrates with Agent Catalog.

Class Description
This class is meant to handle some of the boilerplate around using Agent Catalog with LangGraph's prebuilt ReAct agent. More specifically, this class performs the following:

Fetches the prompt given the name (prompt_name) in the constructor and supplies the prompt and tools attached to the prompt to the ReAct agent constructor.

Attaches a agentc_langchain.chat.Callback to the given chat_model to record all chat-model related activity (i.e., chat completions and tool calls).

Wraps tools (if present in the prompt) in a agentc_langgraph.tool.ToolNode instance to record the results of tool calls.

Wraps the invocation of this agent in a agentc.Span context manager.

Below, we illustrate an example Agent Catalog prompt and an implementation of this class for our prompt. First, our prompt:

record_kind: prompt
name: endpoint_finding_node
description: All inputs required to assemble the endpoint finding agent.

output:
  title: Endpoints
  description: The source and destination airports for a flight / route.
  type: object
  properties:
    source:
      type: string
      description: "The IATA code for the source airport."
    dest:
      type: string
      description: "The IATA code for the destination airport."
  required: [source, dest]

content:
  agent_instructions: >
    Your task is to find the source and destination airports for a flight.
    The user will provide you with the source and destination cities.
    You need to find the IATA codes for the source and destination airports.
    Another agent will use these IATA codes to find a route between the two airports.
    If a route cannot be found, suggest alternate airports (preferring airports that are more likely to have
    routes between them).

  output_format_instructions: >
    Ensure that each IATA code is a string and is capitalized.
Next, the usage of this prompt in an implementation of this class:

import langchain_core.messages
import agentc_langgraph.agent
import agentc
import typing

class State(agentc_langgraph.state):
    endpoints: typing.Optional[dict]

class EndpointFindingAgent(agentc_langgraph.agent.ReActAgent):
    def __init__(self, catalog: agentc.Catalog, span: agentc.Span, **kwargs):
        chat_model = langchain_openai.chat_models.ChatOpenAI(model="gpt-4o", temperature=0)
        super().__init__(
            chat_model=chat_model,
            catalog=catalog,
            span=span,
            prompt_name="endpoint_finding_node",
             **kwargs
        )

    def _invoke(self, span: agentc.Span, state: State, config) -> State:
        # Give the working state to our agent.
        agent = self.create_react_agent(span)
        response = agent.invoke(input=state, config=config)

        # 'source' and 'dest' comes from the prompt's output format.
        # Note this is a direct mutation on the "state" given to the Span!
        structured_response = response["structured_response"]
        state["endpoints"] = {"source": structured_response["source"], "destination": structured_response["dest"]}
        state["messages"].append(response["messages"][-1])
        return state

if __name__ == '__main__':
    catalog = agentc.Catalog()
    span = catalog.Span(name="root_span")
    my_agent = EndpointFindingAgent(catalog=catalog, span=span)
Note

For all constructor parameters, see the documentation for langgraph.prebuilt.create_react_agent here.

Parameters:
chat_model (BaseChatModel)

catalog (Catalog)

span (Span)

prompt_name (str)

async ainvoke(
input,
config=None,
**kwargs
)[source]
Default implementation of ainvoke, calls invoke from a thread.

The default implementation allows usage of async code even if the Runnable did not implement a native async version of invoke.

Subclasses should override this method if they can run asynchronously.

Parameters:
input (State)

config (RunnableConfig | None)

Return type:
State | Command

invoke(
input,
config=None,
**kwargs
)[source]
Transform a single input into an output.

Args:
input: The input to the Runnable. config: A config to use when invoking the Runnable.

The config supports standard keys like 'tags', 'metadata' for tracing purposes, 'max_concurrency' for controlling how much work to do in parallel, and other keys. Please refer to the RunnableConfig for more details. Defaults to None.

Returns:
The output of the Runnable.

Parameters:
input (State)

config (RunnableConfig | None)

Return type:
State | Command

name: str | None
The name of the Runnable. Used for debugging and tracing.

class agentc_langgraph.agent.State[source]
An (optional) state class for use with Agent Catalog's LangGraph helper classes.

Class Description
The primary use for this class to help agentc_langgraph.agent.ReActAgent instances build agentc.span.EdgeContent logs. This class is essentially identical to the default state schema for LangGraph (i.e., messages and is_last_step) but with the inclusion of a new previous_node field.

class agentc_langgraph.graph.GraphRunnable(
*,
catalog,
span=None
)[source]
A helper class that wraps the "Runnable" interface with agentc.Span.

Class Description
This class is meant to handle some of the boilerplate around using agentc.Span instances and LangGraph compiled graphs. Specifically, this class builds a new span on instantiation and wraps all Runnable methods in a Span's context manager.

Below, we illustrate an example implementation of this class for a two-agent system.

import langgraph.prebuilt
import langgraph.graph
import langchain_openai
import langchain_core.messages
import agentc_langgraph
import agentc
import typing

class MyResearcherApp(agentc_langgraph.graph.GraphRunnable):
    def search_web(self, str: search_string) -> str:
        ...

    def summarize_results(self, str: content) -> str:
        ...

    def compile(self):
        research_agent = langgraph.prebuilt.create_react_agent(
            model=langchain_openai.ChatOpenAI(model="gpt-4o"),
            tools=[self.search_web]
        )
        summary_agent = langgraph.prebuilt.create_react_agent(
            model=langchain_openai.ChatOpenAI(model="gpt-4o"),
            tools=[self.summarize_results]
        )
        workflow = langgraph.graph.StateGraph(agentc_langgraph.graph.State)
        workflow.add_node("research_agent", research_agent)
        workflow.add_node("summary_agent", summary_agent)
        workflow.add_edge("research_agent", "summary_agent")
        workflow.add_edge("summary_agent", langgraph.graph.END)
        workflow.set_entry_point("research_agent")
        return workflow.compile()

if __name__ == '__main__':
    catalog = agentc.Catalog()
    state = MyResearchState(messages=[], is_last_step=False)
    MyResearcherApp(catalog=catalog).invoke(input=state)
Note

For more information around LangGraph's (LangChain's) Runnable interface, see LangChain's documentation here.

Tip

The example above does not use tools and prompts managed by Agent Catalog. See agentc_langgraph.agent.ReActAgent for a helper class that handles some of the boilerplate around using LangGraph's prebuilt ReAct agent and Agent Catalog.

Parameters:
catalog (Catalog)

span (Span)

class agentc_langgraph.state.CheckpointSaver(
options=None,
*,
serde=None,
**kwargs
)[source]
Checkpoint saver class to persist LangGraph states in a Couchbase instance.

Class Description
Instances of this class are used by LangGraph (passed in during compile() time) to save checkpoints of agent state.

Below, we give a minimal working example of how to use this class with LangGraph's prebuilt ReAct agent.

import langchain_openai
import langgraph.prebuilt
import agentc_langgraph.state

# Pass our checkpoint saver to the create_react_agent method.
chat_model = langchain_openai.ChatOpenAI(name="gpt-4o")
agent = langgraph.prebuilt.create_react_agent(
    model=chat_model,
    tools=list(),
    checkpointer=CheckpointSaver(create_if_not_exists=True)
)
config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [("human", "Hello!)]}, config)
To use this method with Agent Catalog's agentc_langgraph.graph.GraphRunnable class, pass the checkpoint saver to your workflow's compile() method (see the documentation for LangGraph's Graph.compile() method here for more information.

import langgraph.prebuilt
import langgraph.graph
import langchain_openai
import langchain_core.messages
import agentc_langgraph
import agentc
import typing

class MyResearcherApp(agentc_langgraph.graph.GraphRunnable):
    def search_web(self, str: search_string) -> str:
        ...

    def summarize_results(self, str: content) -> str:
        ...

    def compile(self):
        research_agent = langgraph.prebuilt.create_react_agent(
            model=langchain_openai.ChatOpenAI(model="gpt-4o"),
            tools=[self.search_web]
        )
        summary_agent = langgraph.prebuilt.create_react_agent(
            model=langchain_openai.ChatOpenAI(model="gpt-4o"),
            tools=[self.summarize_results]
        )
        workflow = langgraph.graph.StateGraph(agentc_langgraph.graph.State)
        workflow.add_node("research_agent", research_agent)
        workflow.add_node("summary_agent", summary_agent)
        workflow.add_edge("research_agent", "summary_agent")
        workflow.add_edge("summary_agent", langgraph.graph.END)
        workflow.set_entry_point("research_agent")
        checkpointer = agentc_langgraph.state.CheckpointSaver(create_if_not_exists=True)
        return workflow.compile(checkpointer=checkpointer)
Tip

See here for more information about checkpoints in LangGraph.

See also

This class is a wrapper around the langgraph_checkpointer_couchbase.CouchbaseSaver class. See here for more information.

Parameters:
options (CheckpointOptions)

serde (SerializerProtocol)

agentc_langgraph.state.initialize(
options=None,
**kwargs
)[source]
A function to create the collections required to use the checkpoint savers in this module.

Function Description
This function is a helper function for creating the default collections (the thread and tuple collections) required for the CheckpointSaver and :py:class`AsyncCheckpointSaver` classes. Below, we give a minimal working example of how to use this function to create these collections.

import langchain_openai
import langgraph.prebuilt
import agentc_langgraph.state

# Initialize our collections.
agentc_langgraph.state.initialize()

# Pass our checkpoint saver to the create_react_agent method.
chat_model = langchain_openai.ChatOpenAI(name="gpt-4o")
agent = langgraph.prebuilt.create_react_agent(
    model=chat_model,
    tools=list(),
    checkpointer=CheckpointSaver()
)
config = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": [("human", "Hello there!")]}, config)
Parameters:
options (CheckpointOptions) -- The options to use when saving checkpoints to Couchbase.

kwargs -- Keyword arguments to be forwarded to a CheckpointOptions constructor (ignored if options is present).

Return type:
None

pydantic settings agentc_langgraph.state.CheckpointOptions[source]
Config:
extra: str = allow

env_prefix: str = AGENT_CATALOG_LANGGRAPH_CHECKPOINT_

env_file: str = .env

Fields:
bucket (str | None)

checkpoint_collection (str | None)

conn_root_certificate (str | pathlib.Path | None)

conn_string (str | None)

create_if_not_exists (bool | None)

ddl_retry_attempts (int | None)

ddl_retry_wait_seconds (float | None)

password (pydantic.types.SecretStr | None)

scope (str | None)

tuple_collection (str | None)

username (str | None)

field bucket: str | None = None
The name of the Couchbase bucket hosting the checkpoints.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field checkpoint_collection: str | None = 'langgraph_checkpoint_thread'
The name of the Couchbase collection hosting the checkpoints threads.

This field is optional and defaults to langgraph_checkpoint_thread.

Validated by:
_pull_cluster_from_agent_catalog

field conn_root_certificate: str | Path | None = None
Path to the root certificate file for the Couchbase cluster.

This field is optional and only required if the Couchbase cluster is using a self-signed certificate.

Validated by:
_pull_cluster_from_agent_catalog

field conn_string: str | None = None
The connection string to the Couchbase cluster hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field create_if_not_exists: bool | None = False
Create the required collections if they do not exist.

When raised (i.e., this value is set to True), the collections will be created if they do not exist. Lower this flag (set this to False) to instead raise an error if the collections do not exist.

Validated by:
_pull_cluster_from_agent_catalog

field ddl_retry_attempts: int | None = 3
Maximum number of attempts to retry DDL operations.

This value is only used on setup (i.e., the first time the checkpointer is requested). If the number of attempts is exceeded, the command will fail. By default, this value is 3 attempts.

Validated by:
_pull_cluster_from_agent_catalog

field ddl_retry_wait_seconds: float | None = 5
Wait time (in seconds) between DDL operation retries.

This value is only used on setup (i.e., the first time the checkpointer is requested). By default, this value is 5 seconds.

Validated by:
_pull_cluster_from_agent_catalog

field password: SecretStr | None = None
Password associated with the Couchbase instance hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

field scope: str | None = 'agent_activity'
The name of the Couchbase scope hosting the checkpoints.

This field is optional and defaults to agent_activity.

Validated by:
_pull_cluster_from_agent_catalog

field tuple_collection: str | None = 'langgraph_checkpoint_tuple'
The name of the Couchbase collection hosting the checkpoints tuples.

This field is optional and defaults to langgraph_checkpoint_tuple.

Validated by:
_pull_cluster_from_agent_catalog

field username: str | None = None
Username associated with the Couchbase instance hosting the cache.

This field must be specified.

Validated by:
_pull_cluster_from_agent_catalog

LlamaIndex
class agentc_llamaindex.chat.Callback(
span,
event_starts_to_ignore=None,
event_ends_to_ignore=None
)[source]
All callback that will log all LlamaIndex events using the given span as the root.

Class Description
This class is a callback handler that will log ChatCompletionContent, ToolCallContent, and ToolResultContent using events yielded from LlamaIndex (with the given span as the root). Below, we provide an example of how to use this class.

import agentc
import llama_index.core.llms
import llama_index.llms.openai

catalog = agentc.Catalog()
root_span = catalog.Span(name="root_span")
my_prompt = catalog.find("prompt", name="talk_like_a_pirate")
chat_model = llama_index.llms.openai.OpenAI(model="gpt-4o")
chat_model.callback_manager.add_handler(Callback(span=span))
result = chat_model.chat(
    [
        llama_index.core.llms.ChatMessage(role="system", content=my_prompt.content),
        llama_index.core.llms.ChatMessage(role="user", content="What is your name"),
    ]
)
Parameters:
span (Span)

event_starts_to_ignore (list[CBEventType])

event_ends_to_ignore (list[CBEventType])