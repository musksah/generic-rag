from langchain_google_vertexai import ChatVertexAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate

# Pre-retrieval Query Rewriting Function
def query_rewrite(query: str, llm: ChatVertexAI):

    # Rewritten Query Prompt
    query_rewrite_prompt = f"You are a helpful assistant that takes a user's query and turns it into a short statement or paragraph so that it can be used in a semantic similarity search on a vector database to return the most similar chunks of content based on the rewritten query. Please make no comments, just return the rewritten query.\n\nquery: {query}\n\nai: "

    # Invoke LLM
    retrieval_query = llm.invoke(query_rewrite_prompt)

    # Return Generated Retrieval Query
    return retrieval_query


# Create Document Parsing Function to String
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Custom RAG Chain Class
class RAGChain:

    # Chain Constructor
    def __init__(
        self,
        llm: ChatVertexAI,
        retriever: ContextualCompressionRetriever,
        prompt: PromptTemplate
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
    
    # Run Chain Function - same naming convention as LangChain
    def invoke(self, query: str):

        # Advanced RAG: Pre-retrieval Query Rewrite
        retrieval_query = query_rewrite(query, self.llm)
        
        # Retrieval w/ Post-retrieval Reranking
        docs = self.retriever.invoke(retrieval_query.content)

        # Format Docs for Context String
        context = format_docs(docs)

        # Prompt Template
        final_prompt = self.prompt.format(context=context, query=query)

        # LLM Invoke
        return self.llm.invoke(final_prompt)