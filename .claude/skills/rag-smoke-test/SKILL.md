---
name: rag-smoke-test
description: Start Chroma and the API, ingest the sample document into a fresh collection and ask known questions to verify the RAG pipeline end to end. Use after changing retrieval, prompts, chunking, models or dependencies, or when the user asks to "probar el RAG".
argument-hint: ["question" ...]
---

# RAG smoke test

## 1. Preconditions

- Vertex AI credentials: run `gcloud auth application-default print-access-token`
  and check that it succeeds. Do not print the token.
- Dependencies are installed (see README Quickstart).
- If `main.py` still uses `gemini-1.5-flash-001`, stop and tell the user: it is
  retired and `/chat/` will fail.

## 2. Start services

Run both commands from the repo root, in the background:

```bash
chroma run --host localhost --port 8000 --path ./chroma_data_smoke
uvicorn main:app --port 8080
```

Using a separate `--path` keeps the test from duplicating chunks in the user's
real `./chroma_data`. Wait until `GET http://localhost:8080/health_chroma`
returns `success`.

## 3. Ingest once

```bash
curl -s -X POST http://localhost:8080/ingest/
```

## 4. Ask questions

Use the user's questions if they gave any. Otherwise use these, which the
sample document answers, plus one it doesn't:

| Question | Expect |
|---|---|
| "What did the president propose about insulin prices?" | Mentions the $35/month cap for insulin |
| "What did the president say about the CHIPS Act?" | Mentions semiconductors / manufacturing |
| "Who won the 2022 World Cup?" | Says it doesn't know based on the context |

```bash
curl -s -X POST http://localhost:8080/chat/ -H "Content-Type: application/json" -d '{"question": "<q>"}'
```

## 5. Clean up and report

Stop both processes and delete `./chroma_data_smoke`. Report pass/fail per
question with a one-line reason. If a factual answer is wrong, check whether
the problem is in retrieval or generation before suggesting a fix: print
`compression_retriever.invoke(query_rewrite(q, llm).content)` to see which
chunks came back.
