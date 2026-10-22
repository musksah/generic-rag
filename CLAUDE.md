# CLAUDE.md

Minimal RAG API: FastAPI + LangChain + Chroma (HTTP server) + Gemini on Vertex
AI, with LLM query rewriting before retrieval and FlashRank reranking after it.
The repo is public and small. Keep it readable as a teaching example.

## Layout

- `main.py`: FastAPI app. Calls `vertexai.init` at import time, builds the
  FlashRank `ContextualCompressionRetriever` once at module level, and builds a
  new `ChatVertexAI` and `RAGChain` on each `/chat/` request.
- `packrag/chromap.py`: `chroma_client()` (HttpClient on `localhost:8000`) and
  `get_retriever()` (collection `test_collection`, `text-embedding-004`).
- `packrag/ingest.py`: reads `documents/2024_state_of_the_union.txt`, uses
  `CharacterTextSplitter` (1000 chars, 200 overlap), and calls `add_documents`
  on the retriever.
- `packrag/ragchain.py`: `query_rewrite()` and `RAGChain.invoke()`. Rewrite →
  retrieve with the rewritten query → answer with the *original* query.

## Running

Run from the repo root (the ingest path is relative to the working directory):

```bash
chroma run --host localhost --port 8000 --path ./chroma_data   # terminal 1
uvicorn main:app --port 8080                                   # terminal 2
curl -X POST http://localhost:8080/ingest/
curl -X POST http://localhost:8080/chat/ -H "Content-Type: application/json" -d '{"question": "..."}'
```

Chroma owns port 8000, so never start uvicorn on its default port.
Vertex AI needs Application Default Credentials
(`gcloud auth application-default login`).

## Gotchas

- No dependency file exists. The code imports `langchain.retrievers`, which
  LangChain 1.0 moved to `langchain-classic`, so install `langchain<1.0` or
  port those imports.
- `gemini-1.5-flash-001` in `main.py` is a retired model and must be replaced
  with a current Gemini model before `/chat/` will work. Check whether
  `text-embedding-004` is still available too. Changing the embedding model
  means re-ingesting into a fresh collection, because old and new vectors are
  not comparable.
- Each `/ingest/` call duplicates every chunk, because `add_documents` is called
  without ids. To reset, delete `./chroma_data` or the collection.
- `PROJECT_ID` is hardcoded in `main.py`, and `CHROMA_PORT` there is unused
  because `chromap.py` hardcodes the port too. In `main.py`, the `ranker`
  variable is unused because `FlashrankRerank()` loads its own default model.
- `query_rewrite` returns an `AIMessage`. Callers use `.content`.

## Conventions

- Plain functions and one small class, with no framework abstractions beyond
  LangChain. Keep new code at that level.
- Comments describe pipeline stages ("Pre-retrieval", "Post-retrieval"). Match
  that wording when adding stages.
- No tests yet. Verify changes with the `rag-smoke-test` skill.
