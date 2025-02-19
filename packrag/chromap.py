import chromadb
from langchain_chroma import Chroma
from langchain_google_vertexai import VertexAIEmbeddings

def chroma_client():
    return chromadb.HttpClient(host='localhost', port=8000)


def get_retriever():
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    vector_store = Chroma(
        client=chroma_client(),
        collection_name="test_collection",
        embedding_function=embeddings
    )
    # Set Chroma as the Retriever
    return vector_store.as_retriever()