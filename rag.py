from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class RAG:
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "mxbai-embed-large"):
        embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    
    def get_context(self, query: str, k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])