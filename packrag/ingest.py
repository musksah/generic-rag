import hashlib
from pathlib import Path

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import CharacterTextSplitter

from packrag import config


def chunk_id(source: str, text: str) -> str:
    """Deterministic id so re-ingesting the same file overwrites instead of duplicating."""
    return hashlib.sha256(f"{source}\n{text}".encode("utf-8")).hexdigest()


def ingest_data(vector_store: VectorStore, path: str = config.DOCUMENT_PATH) -> int:
    """Split a text file into chunks and upsert them. Returns the number of chunks stored."""
    source = Path(path).name
    with open(path, encoding="utf8") as f:
        text = f.read()

    # Initialize Text Splitter
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    # Create Documents (Chunks) From File
    docs = text_splitter.create_documents([text], metadatas=[{"source": source}])

    # Identical chunks would share an id, and Chroma rejects duplicate ids in one batch
    unique = {chunk_id(source, d.page_content): d for d in docs}

    # Save Document Chunks to Vector Store (Chroma upserts by id)
    vector_store.add_documents(list(unique.values()), ids=list(unique.keys()))
    return len(unique)
