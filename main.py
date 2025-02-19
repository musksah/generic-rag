from fastapi import FastAPI
from fastapi.responses import JSONResponse
import vertexai
import packrag.ragchain as ragmodule
import packrag.chromap as chromap
import packrag.ingest as ingestmodule
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import PromptTemplate
from flashrank import Ranker 
from pydantic import BaseModel

CHROMA_PORT = 8000
PROJECT_ID = "master-ia-404704"
LOCATION = "us-central1"

class Message(BaseModel):
    question: str

vertexai.init(project=PROJECT_ID, location=LOCATION)
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health_chroma")
async def test_conection_chroma():
    chromap.chroma_client()
    return JSONResponse({"status": "success", "message": "Chroma is connected"})


ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Flash Rerank Compressor for Post-retrieval Rerank
compressor = FlashrankRerank()

# Update Retriever -> Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=chromap.get_retriever()
)

@app.post("/ingest/")
async def ingest():
    ingestmodule.ingest_data()
    return JSONResponse({"message": "Information updated"})

@app.post("/chat/")
async def query(message: Message):
    # Create the Prompt Template
    prompt_template = """Use the context provided to answer the user's question below. If you do not know the answer based on the context provided, tell the user that you do not know the answer to their question based on the context provided and that you are sorry.
    context: {context}

    question: {query}

    answer: """

    # Create Prompt Instance from template
    custom_rag_prompt = PromptTemplate.from_template(prompt_template)
    llm = ChatVertexAI(
        model="gemini-1.5-flash-001",
        temperature=0.2)
    # Initialize Custom RAG Chain
    rag_chain = ragmodule.RAGChain(llm, compression_retriever, custom_rag_prompt)

    # Invoke the chain
    response = rag_chain.invoke(message.question)

    # Print Output
    return {"answer": response.content}
