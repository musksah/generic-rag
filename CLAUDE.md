# CLAUDE.md

Minimal RAG API: FastAPI + LangChain + Chroma (HTTP server) + Gemini on Vertex
AI (through `langchain-google-genai` with `vertexai=True`), with LLM query
rewriting before retrieval and FlashRank reranking after it. The repo is public
and small. Keep it readable as a teaching example.

## Layout

- `main.py`: FastAPI app. `get_chain()` is `lru_cache`d and builds the LLM,
  retriever and `Ranker` on the first `/chat/` request, so importing `main`
  never touches the network.
- `packrag/config.py`: every setting, read from env vars with defaults. Add
  new settings here, not as constants in other modules.
- `packrag/chromap.py`: `chroma_client()`, `get_embeddings()`, and
  `get_vector_store(client=None, embeddings=None)`. The arguments are
  injectable so tests can swap in fakes.
- `packrag/ingest.py`: `ingest_data(vector_store, path)` splits the file
  (1000 chars, 200 overlap) and upserts with `chunk_id(source, text)`, a sha256
  of source plus text. Returns the number of chunks.
- `packrag/ragchain.py`: `query_rewrite()` → `retriever.invoke(rewritten)` →
  `rerank(rewritten, docs, ranker, top_n)` → answer with the *original* query.
- `tests/test_pipeline.py`: offline tests using `chromadb.EphemeralClient`,
  `DeterministicFakeEmbedding`, `FakeListChatModel` and a stub ranker.

## Commands

```bash
uv venv .venv && uv pip install --python .venv -r requirements-dev.txt   # or pip
pytest -q                                                                 # offline, about 10 s
chroma run --host localhost --port 8000 --path ./chroma_data              # terminal 1
uvicorn main:app --port 8080                                              # terminal 2 (Chroma owns 8000)
```

`/ingest/` and `/chat/` call Vertex AI, which is billed: they need ADC
(`gcloud auth application-default login`) and `GOOGLE_CLOUD_PROJECT`. Ask the
user before running them against a real project.

## Conventions

- LangChain 1.x only. Do not import from `langchain.*` or `langchain_classic`.
  Use `langchain_core` and the partner packages. Read model output with
  `.text`, not `.content`, because Gemini 3 models may return content blocks.
- Pipeline stages are commented "Pre-retrieval" / "Retrieval" /
  "Post-retrieval". Match that wording when adding stages.
- Every new behavior gets an offline test. Never make tests depend on Vertex
  AI, a running Chroma, or downloading the FlashRank model.
- Keep `requirements.txt` ranges in sync with what CI installs
  (`.github/workflows/ci.yml` runs `pytest` on Python 3.12).

## Gotchas

- Changing `EMBEDDING_MODEL` needs a new `CHROMA_COLLECTION` and a re-ingest,
  because vectors from different models can't be compared.
- `text-embedding-005` retires on 2027-04-01. `gemini-embedding-001` is the
  long-lived replacement, but it hasn't been tested with this code yet.
- `Ranker` downloads its model (about 22 MB) on first use into flashrank's cache dir.
- The `/health_chroma` route name contains a typo (`test_conection_chroma`).
  Renaming the function is fine, but keep the path stable.
