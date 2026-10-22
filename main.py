from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from flashrank import Ranker
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

import packrag.chromap as chromap
import packrag.config as config
import packrag.ingest as ingestmodule
import packrag.ragchain as ragmodule

PROMPT_TEMPLATE = """Use the context provided to answer the user's question below. If you do not know the answer based on the context provided, tell the user that you do not know the answer to their question based on the context provided and that you are sorry.
context: {context}

question: {query}

answer: """


class Message(BaseModel):
    question: str


app = FastAPI()


# Built on first request, so importing the app needs neither Chroma nor Vertex AI
@lru_cache
def get_chain() -> ragmodule.RAGChain:
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=0.2,
        vertexai=True,
        project=config.PROJECT_ID,
        location=config.LOCATION,
    )
    retriever = chromap.get_vector_store().as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
    return ragmodule.RAGChain(
        llm,
        retriever,
        PromptTemplate.from_template(PROMPT_TEMPLATE),
        ranker=Ranker(model_name=config.RERANK_MODEL),
        top_n=config.RERANK_TOP_N,
    )


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health_chroma")
async def test_conection_chroma():
    chromap.chroma_client().heartbeat()
    return JSONResponse({"status": "success", "message": "Chroma is connected"})


@app.post("/ingest/")
def ingest():
    chunks = ingestmodule.ingest_data(chromap.get_vector_store())
    return JSONResponse({"message": "Information updated", "chunks": chunks})


@app.post("/chat/")
def query(message: Message):
    response = get_chain().invoke(message.question)
    return {"answer": response.text}
