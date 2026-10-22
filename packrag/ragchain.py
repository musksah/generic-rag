from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever


# Pre-retrieval Query Rewriting Function
def query_rewrite(query: str, llm: BaseChatModel) -> str:

    # Rewritten Query Prompt
    query_rewrite_prompt = f"You are a helpful assistant that takes a user's query and turns it into a short statement or paragraph so that it can be used in a semantic similarity search on a vector database to return the most similar chunks of content based on the rewritten query. Please make no comments, just return the rewritten query.\n\nquery: {query}\n\nai: "

    # Invoke LLM and return the generated retrieval query
    return llm.invoke(query_rewrite_prompt).text


# Post-retrieval Reranking Function
def rerank(query: str, docs: list[Document], ranker: Ranker, top_n: int) -> list[Document]:
    if not docs:
        return []
    passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(docs)]
    results = ranker.rerank(RerankRequest(query=query, passages=passages))
    return [docs[r["id"]] for r in results[:top_n]]


# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Custom RAG Chain Class
class RAGChain:

    # Chain Constructor
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: BaseRetriever,
        prompt: PromptTemplate,
        ranker: Ranker | None = None,
        top_n: int = 3,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.ranker = ranker
        self.top_n = top_n

    # Run Chain Function - same naming convention as LangChain
    def invoke(self, query: str) -> AIMessage:

        # Advanced RAG: Pre-retrieval Query Rewrite
        retrieval_query = query_rewrite(query, self.llm)

        # Retrieval
        docs = self.retriever.invoke(retrieval_query)

        # Advanced RAG: Post-retrieval Reranking
        if self.ranker is not None:
            docs = rerank(retrieval_query, docs, self.ranker, self.top_n)

        # Format Docs for Context String
        context = format_docs(docs)

        # Prompt Template
        final_prompt = self.prompt.format(context=context, query=query)

        # LLM Invoke
        return self.llm.invoke(final_prompt)
