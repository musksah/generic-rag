---
name: rag-smoke-test
description: Verify the RAG pipeline. First run the offline pytest suite, then, only with the user's OK because it bills Vertex AI, run an end-to-end check against Chroma and Gemini with known questions. Use after changing retrieval, prompts, chunking, models or dependencies, or when the user asks to "probar el RAG".
argument-hint: ["question" ...]
---

# RAG smoke test

## 1. Offline (always)

```bash
pytest -q
```

All tests must pass before going further. If one fails, fix it or report it.

## 2. End to end (ask first)

This step calls Vertex AI, which is billed. Ask the user which
`GOOGLE_CLOUD_PROJECT` to use, and never pick one yourself. Then check
`gcloud auth application-default print-access-token` without printing the token.

Start both services from the repo root, in the background, with a throwaway
collection so the user's data is untouched:

```bash
export CHROMA_COLLECTION=smoke_test
chroma run --host localhost --port 8000 --path ./.chroma_smoke
uvicorn main:app --port 8080
```

Wait for `GET http://localhost:8080/health_chroma` to return `success`, then run
`curl -s -X POST http://localhost:8080/ingest/`. Expect `"chunks": 48` for the
sample document.

Ask the user's questions if they gave any. Otherwise use:

| Question | Expect |
|---|---|
| "What did the president propose about insulin prices?" | Mentions the $35/month cap |
| "What did the president say about the CHIPS Act?" | Mentions semiconductors / chip factories |
| "Who won the 2022 World Cup?" | Says it doesn't know based on the context |

```bash
curl -s -X POST http://localhost:8080/chat/ -H "Content-Type: application/json" -d '{"question": "<q>"}'
```

## 3. Clean up and report

Stop both processes and delete `./.chroma_smoke`. Report pass/fail per check.
For a wrong answer, find out whether retrieval or generation failed before
proposing a fix: call `get_chain().retriever.invoke(query_rewrite(q, llm))` and
`rerank(...)` from `packrag.ragchain` to see which chunks reached the prompt.
