import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from packrag import config


def chroma_client():
    return chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT)


def get_embeddings() -> Embeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        vertexai=True,
        project=config.PROJECT_ID,
        location=config.LOCATION,
    )


def get_vector_store(client=None, embeddings: Embeddings | None = None) -> Chroma:
    # client and embeddings are injectable so tests can run without Chroma or Vertex AI
    return Chroma(
        client=client or chroma_client(),
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
    )
