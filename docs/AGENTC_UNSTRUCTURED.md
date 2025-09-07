## Agent Catalog + Unstructured: Build a Document QA Agent (LangChain)

This tutorial shows how to create a new agent (separate from the hotel example) that uses the Unstructured library to parse files (PDF, DOCX, HTML, etc.), stores embeddings in Couchbase, and uses Agent Catalog (agentc) to wire prompts and tools.

### What you’ll build
- **Ingestion tool**: parse files with Unstructured and upsert to Couchbase Vector Store
- **Search tool**: semantic search over parsed chunks
- **Prompt**: instruct the agent to call tools and answer
- **Runner**: a `main.py` that pulls prompt/tools from agentc and runs a ReAct agent

### Prerequisites
- **Python**: 3.12+
- **Poetry**: installed
- **Couchbase**: Capella or local (`CB_CONN_STRING`, `CB_USERNAME`, `CB_PASSWORD`)
- **Models**: Capella/OpenAI/NVIDIA keys configured like other agents (`.env`)
- **Unstructured**: choose extras for your doc types, e.g. `unstructured[pdf,docx,html]`

### Project layout
Create a new folder for your agent, e.g. `notebooks/unstructured_doc_agent_langchain/` with:

```
notebooks/unstructured_doc_agent_langchain/
├── main.py
├── pyproject.toml
├── prompts/
│   └── doc_assistant.yaml
├── tools/
│   ├── ingest_documents.py
│   └── search_documents.py
├── data/
│   └── samples/            # put a few PDFs/DOCX/HTML here for testing (optional)
├── evals/                  # optional (Arize/Phoenix like other agents)
├── .env.sample
└── agentcatalog_index.json # if you define a custom vector index (optional)
```

### Dependencies (pyproject.toml)
Use a minimal per-agent `pyproject.toml` similar to other examples:

```toml
[project]
name = "unstructured-doc-agent"
version = "0.1.0"
description = "Document QA agent using Unstructured + Couchbase + Agent Catalog"
requires-python = ">=3.12,<3.13"
dependencies = [
  "python-dotenv>=1.0.0,<2.0.0",
  "langchain>=0.3.0,<0.4.0",
  "langchain-openai>=0.3.0,<0.4.0",
  "langchain-couchbase>=0.2.4,<0.3.0",
  "unstructured[pdf,docx,html]>=0.15.0,<0.16.0",
  "agentc @ ../../agent-catalog/libs/agentc",
  "agentc-langchain @ ../../agent-catalog/libs/agentc_integrations/langchain",
]

[tool.poetry]
package-mode = false

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

### Tool: ingest documents with Unstructured
Create `tools/ingest_documents.py`:

```python
import os
import glob
from typing import List

import agentc
from unstructured.partition.auto import partition
from langchain_couchbase.vectorstores import CouchbaseVectorStore

from datetime import timedelta
import couchbase.auth, couchbase.cluster, couchbase.options


def _connect_cluster():
    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password"),
    )
    options = couchbase.options.ClusterOptions(authenticator=auth)
    options.apply_profile("wan_development")
    cluster = couchbase.cluster.Cluster(os.getenv("CB_CONN_STRING", "couchbase://localhost"), options)
    cluster.wait_until_ready(timedelta(seconds=15))
    return cluster


def _get_texts_from_file(path: str) -> List[str]:
    """Extract text chunks from a file via Unstructured."""
    elements = partition(filename=path)
    # Merge small elements; keep it simple for the tutorial
    return [el.text.strip() for el in elements if getattr(el, "text", "").strip()]


def _ensure_vector_store(embeddings):
    cluster = _connect_cluster()
    return CouchbaseVectorStore(
        cluster=cluster,
        bucket_name=os.getenv("CB_BUCKET", "travel-sample"),
        scope_name=os.getenv("CB_SCOPE", "agentc_data"),
        collection_name=os.getenv("CB_COLLECTION", "unstructured_docs"),
        embedding=embeddings,
        index_name=os.getenv("CB_INDEX", "unstructured_docs_index"),
    )


@agentc.catalog.tool
def ingest_documents(path_or_glob: str) -> str:
    """Parse documents with Unstructured and upsert chunks to Couchbase vector store.

    Args:
        path_or_glob: a single path or glob (e.g. "data/samples/*.pdf")
    """
    # Lazy import to reuse shared setup from project root
    import sys, pathlib
    root = pathlib.Path(__file__).resolve()
    while root.parent != root and (root / "shared").exists() is False:
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from shared.agent_setup import setup_ai_services

    embeddings, _ = setup_ai_services(framework="langchain")
    if not embeddings:
        return "ERROR: No embeddings available"

    paths = glob.glob(path_or_glob) if any(ch in path_or_glob for ch in ["*", "?"]) else [path_or_glob]
    if not paths:
        return f"NO_FILES: {path_or_glob}"

    store = _ensure_vector_store(embeddings)

    added = 0
    for p in paths:
        try:
            chunks = _get_texts_from_file(p)
            if chunks:
                store.add_texts(texts=chunks, batch_size=16)
                added += len(chunks)
        except Exception as e:
            return f"ERROR: failed on {p} - {e}"

    return f"INGESTED: {added} chunks from {len(paths)} file(s)"
```

### Tool: search documents
Create `tools/search_documents.py`:

```python
import agentc
from langchain_couchbase.vectorstores import CouchbaseVectorStore

@agentc.catalog.tool
def search_documents(query: str) -> str:
    import os
    from datetime import timedelta
    import couchbase.auth, couchbase.cluster, couchbase.options
    from shared.agent_setup import setup_ai_services

    embeddings, _ = setup_ai_services(framework="langchain")
    if not embeddings:
        return "ERROR: No embeddings available"

    auth = couchbase.auth.PasswordAuthenticator(
        username=os.getenv("CB_USERNAME", "Administrator"),
        password=os.getenv("CB_PASSWORD", "password"),
    )
    options = couchbase.options.ClusterOptions(authenticator=auth)
    options.apply_profile("wan_development")
    cluster = couchbase.cluster.Cluster(os.getenv("CB_CONN_STRING", "couchbase://localhost"), options)
    cluster.wait_until_ready(timedelta(seconds=15))

    store = CouchbaseVectorStore(
        cluster=cluster,
        bucket_name=os.getenv("CB_BUCKET", "travel-sample"),
        scope_name=os.getenv("CB_SCOPE", "agentc_data"),
        collection_name=os.getenv("CB_COLLECTION", "unstructured_docs"),
        embedding=embeddings,
        index_name=os.getenv("CB_INDEX", "unstructured_docs_index"),
    )

    results = store.similarity_search_with_score(query, k=5)
    if not results:
        return "NO_RESULTS"
    lines = [f"DOC_{i+1}: {doc.page_content[:300]} (score={score:.3f})" for i, (doc, score) in enumerate(results)]
    return "\n\n".join(lines)
```

### Prompt
Create `prompts/doc_assistant.yaml`:

```yaml
record_kind: prompt
name: doc_assistant
description: >
  Assistant that can ingest documents with Unstructured and answer questions from them.
tools:
  - name: "ingest_documents"
  - name: "search_documents"
content: >
  You can call tools to ingest and search documents. Use at most one ingest, then search, then answer.

  Format:
  Question: {input}
  Thought: {agent_scratchpad}
  Action: one of [{tool_names}]
  Action Input: arguments for the action
  Observation: tool result
  Thought: I now have all the information needed.
  Final Answer: concise helpful answer with snippets
```

### main.py (runner)
Create `main.py` (mirrors other agents) and wire with agentc:

```python
import os, sys, json, logging
import agentc
import agentc_langchain
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

# project root discovery (same pattern as other agents)
def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.isdir(os.path.join(current, "shared")):
            return current
        current = os.path.dirname(current)
    return None

root = find_project_root()
if root and root not in sys.path:
    sys.path.insert(0, root)

from shared.agent_setup import setup_ai_services, setup_environment

logging.basicConfig(level=logging.INFO)

def build_agent():
    catalog = agentc.catalog.Catalog()
    setup_environment()

    _, llm = setup_ai_services(framework="langchain")

    tool_ingest = catalog.find("tool", name="ingest_documents")
    tool_search = catalog.find("tool", name="search_documents")
    prompt_rec = catalog.find("prompt", name="doc_assistant")

    tools = [
        Tool(name=tool_ingest.meta.name, description=tool_ingest.meta.description, func=tool_ingest.func),
        Tool(name=tool_search.meta.name, description=tool_search.meta.description, func=tool_search.func),
    ]

    prompt = PromptTemplate(
        template=prompt_rec.content.strip(),
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tools": "\n".join([f"{t.name}: {t.description}" for t in tools]),
            "tool_names": ", ".join([t.name for t in tools]),
        },
    )

    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=3)

def main():
    agent = build_agent()
    # Optional: first ingest a folder
    print(agent.invoke({"input": "Ingest documents from data/samples/*.pdf"}))
    # Then ask a question
    print(agent.invoke({"input": "Summarize the key points about policy renewal from the PDFs."}))

if __name__ == "__main__":
    main()
```

### Index with agentc and run
From repo root:

```bash
poetry -C notebooks/unstructured_doc_agent_langchain install --no-root
cp notebooks/unstructured_doc_agent_langchain/.env.sample notebooks/unstructured_doc_agent_langchain/.env
$EDITOR notebooks/unstructured_doc_agent_langchain/.env

cd notebooks/unstructured_doc_agent_langchain
agentc init
agentc index tools/
agentc index prompts/

poetry run python main.py
```

### Environment (.env)
- **Couchbase**: `CB_CONN_STRING`, `CB_USERNAME`, `CB_PASSWORD`, `CB_BUCKET=travel-sample`, `CB_SCOPE=agentc_data`, `CB_COLLECTION=unstructured_docs`, `CB_INDEX=unstructured_docs_index`
- **Models** (choose any supported):
  - Capella OpenAI-compatible: `CAPELLA_API_ENDPOINT`, `CAPELLA_API_LLM_KEY`, `CAPELLA_API_EMBEDDINGS_KEY`, `CAPELLA_API_LLM_MODEL`, `CAPELLA_API_EMBEDDING_MODEL`
  - NVIDIA: `NVIDIA_API_KEY`, `NVIDIA_API_LLM_MODEL`, `NVIDIA_API_EMBEDDING_MODEL`
  - OpenAI fallback: `OPENAI_API_KEY`

### Notes on Unstructured
- Install extras based on your formats: `unstructured[pdf,docx,html]`
- Some file types may require system deps (e.g., `libmagic`, `tesseract`, `poppler`). Consult Unstructured docs.

### Optional: Evaluation with Arize
Replicate the hotel agent’s `evals/` pattern to add Phoenix/Arize evaluation. You can reuse the evaluator class from the hotel agent and only swap the agent setup import.

### Troubleshooting
- If tools aren’t found, re-run `agentc index tools/` and `agentc index prompts/` inside the agent folder.
- If the CLI isn’t found, ensure `~/.local/bin` is on your `PATH` or re-run `scripts/setup.sh --yes`.
- For connection issues, verify Couchbase creds and `CB_CONN_STRING`.


