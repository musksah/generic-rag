"""Offline tests: no Chroma server, Vertex AI or reranker model download needed."""

import uuid

import chromadb
from fastapi.testclient import TestClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

import main
from packrag import ingest, ragchain


class ReverseRanker:
    """Stands in for flashrank.Ranker: ranks passages in reverse input order."""

    def __init__(self):
        self.queries = []

    def rerank(self, request):
        self.queries.append(request.query)
        return [{**p, "score": 1.0 / (i + 1)} for i, p in enumerate(reversed(request.passages))]


def _vector_store():
    return Chroma(
        client=chromadb.EphemeralClient(),
        collection_name=f"test-{uuid.uuid4().hex}",
        embedding_function=DeterministicFakeEmbedding(size=32),
    )


def test_ingest_is_idempotent():
    store = _vector_store()
    first = ingest.ingest_data(store)
    second = ingest.ingest_data(store)
    assert first > 0
    assert first == second == store._collection.count()


def test_rerank_keeps_top_n_in_ranker_order():
    docs = [Document(page_content=t) for t in ("a", "b", "c", "d")]
    assert [d.page_content for d in ragchain.rerank("q", docs, ReverseRanker(), top_n=2)] == ["d", "c"]
    assert ragchain.rerank("q", [], ReverseRanker(), top_n=2) == []


def test_chain_retrieves_with_rewritten_query_and_answers_original():
    retrieved_with = []

    def fake_retrieve(q):
        retrieved_with.append(q)
        return [Document(page_content="chunk one"), Document(page_content="chunk two")]

    llm = FakeListChatModel(responses=["rewritten query", "final answer"])
    ranker = ReverseRanker()
    chain = ragchain.RAGChain(
        llm,
        RunnableLambda(fake_retrieve),
        PromptTemplate.from_template("{context}|{query}"),
        ranker=ranker,
        top_n=1,
    )

    assert chain.invoke("original question").text == "final answer"
    assert retrieved_with == ["rewritten query"]
    assert ranker.queries == ["rewritten query"]


def test_chat_endpoint(monkeypatch):
    class StubChain:
        def invoke(self, question):
            return AIMessage(content=f"echo: {question}")

    monkeypatch.setattr(main, "get_chain", lambda: StubChain())
    response = TestClient(main.app).post("/chat/", json={"question": "hi"})
    assert response.status_code == 200
    assert response.json() == {"answer": "echo: hi"}
