# Imports
from langchain_text_splitters import CharacterTextSplitter
from packrag import chromap

def ingest_data():

    # Initialize ChromaDB as Vector Store
    vector_store = chromap.get_retriever()

    # Read in State of the Union Address File
    with open("documents/2024_state_of_the_union.txt", encoding="utf8") as f:
        state_of_the_union = f.read()

    # Initialize Text Splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Create Documents (Chunks) From File
    texts = text_splitter.create_documents([state_of_the_union])

    # Save Document Chunks to Vector Store
    ids = vector_store.add_documents(texts)

