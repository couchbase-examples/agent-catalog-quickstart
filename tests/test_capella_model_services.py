"""Offline tests for the Capella model-service adapters in ``shared/``.

Every HTTP call is answered by an ``httpx.MockTransport``, so no Capella
endpoint or API key is needed. Run from the repository root with:

    pip install pytest httpx "langchain-core>=0.3,<0.4" "llama-index-core>=0.12,<0.13"
    python -m pytest tests
"""

import json

import httpx
import pytest

langchain_services = pytest.importorskip("shared.capella_model_services_langchain")
llamaindex_services = pytest.importorskip("shared.capella_model_services_llamaindex")


@pytest.fixture
def capella(monkeypatch):
    """Route every ``httpx.Client`` to a fake Capella endpoint and record requests."""
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append({"url": str(request.url), "headers": request.headers, "json": body})
        if request.url.path.endswith("/embeddings"):
            data = [{"embedding": [float(i), 0.5]} for i, _ in enumerate(body["input"])]
            return httpx.Response(200, json={"data": data})
        if request.url.path.endswith("/chat/completions"):
            reply = f"echo:{body['messages'][-1]['content']}"
            return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
        return httpx.Response(404)

    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", fake_client)
    return requests


ADAPTERS = {
    "langchain": langchain_services,
    "llamaindex": llamaindex_services,
}


def _embed_query(framework, embeddings, text):
    if framework == "langchain":
        return embeddings.embed_query(text)
    return embeddings.get_query_embedding(text)


def _embed_documents(framework, embeddings, texts):
    if framework == "langchain":
        return embeddings.embed_documents(texts)
    return embeddings.get_text_embedding_batch(texts)


def _ask(framework, llm, prompt):
    if framework == "langchain":
        return llm.invoke(prompt).content
    return llm.complete(prompt).text


@pytest.mark.parametrize("framework", ADAPTERS)
@pytest.mark.parametrize(
    "base_url",
    ["https://capella.test", "https://capella.test/", "https://capella.test/v1", "https://capella.test/v1/"],
)
def test_embeddings_use_v1_endpoint_and_bearer_auth(capella, framework, base_url):
    embeddings = ADAPTERS[framework].create_capella_embeddings(
        api_key="emb-key", base_url=base_url, model="nvidia/llama-3.2-nv-embedqa-1b-v2"
    )

    assert _embed_query(framework, embeddings, "hotels in Paris") == [0.0, 0.5]

    (request,) = capella
    assert request["url"] == "https://capella.test/v1/embeddings"
    assert request["headers"]["authorization"] == "Bearer emb-key"
    assert request["json"] == {
        "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
        "input": ["hotels in Paris"],
        "input_type": "query",
    }


@pytest.mark.parametrize("framework", ADAPTERS)
def test_passages_use_passage_input_type(capella, framework):
    embeddings = ADAPTERS[framework].create_capella_embeddings(
        api_key="k", base_url="https://capella.test", model="nvidia/nv-embedqa-e5-v5"
    )

    vectors = _embed_documents(framework, embeddings, ["a", "b"])

    assert vectors == [[0.0, 0.5], [1.0, 0.5]]
    assert [r["json"]["input_type"] for r in capella] == ["passage"] * len(capella)
    assert [t for r in capella for t in r["json"]["input"]] == ["a", "b"]


@pytest.mark.parametrize("framework", ADAPTERS)
def test_symmetric_models_omit_input_type(capella, framework):
    embeddings = ADAPTERS[framework].create_capella_embeddings(
        api_key="k", base_url="https://capella.test", model="intfloat/e5-mistral-7b-instruct"
    )

    _embed_query(framework, embeddings, "q")

    assert "input_type" not in capella[0]["json"]


@pytest.mark.parametrize("framework", ADAPTERS)
def test_long_inputs_are_truncated_to_token_budget(capella, framework):
    embeddings = ADAPTERS[framework].create_capella_embeddings(
        api_key="k", base_url="https://capella.test", model="m", max_tokens=10
    )

    _embed_query(framework, embeddings, "x" * 100)

    # 10 tokens * ~3 chars/token * 0.8 safety buffer
    assert capella[0]["json"]["input"] == ["x" * 24]


@pytest.mark.parametrize("framework", ADAPTERS)
@pytest.mark.parametrize(
    "base_url", ["https://capella.test", "https://capella.test/v1", "https://capella.test/v1/"]
)
def test_chat_llm_posts_chat_completions(capella, framework, base_url):
    llm = ADAPTERS[framework].create_capella_chat_llm(
        api_key="llm-key", base_url=base_url, model="meta/llama3-8b-instruct", temperature=0.2
    )

    assert _ask(framework, llm, "Hello") == "echo:Hello"

    (request,) = capella
    assert request["url"] == "https://capella.test/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer llm-key"
    assert request["json"]["model"] == "meta/llama3-8b-instruct"
    assert request["json"]["temperature"] == 0.2
    assert request["json"]["messages"] == [{"role": "user", "content": "Hello"}]


def test_api_errors_propagate(monkeypatch):
    real_client = httpx.Client

    def failing_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(lambda request: httpx.Response(401))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", failing_client)
    embeddings = langchain_services.create_capella_embeddings(
        api_key="bad", base_url="https://capella.test", model="m"
    )

    with pytest.raises(httpx.HTTPStatusError):
        embeddings.embed_query("q")
