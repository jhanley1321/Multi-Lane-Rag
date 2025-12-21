from langchain_ollama import ChatOllama
from typing import Optional

class LLM:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.7, vector_db = None):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
        self.vector_db = vector_db
    
    def send_message(self, message: str, use_rag: bool = True, lane_name: Optional[str] = None) -> str:
        if use_rag and self.vector_db:
            docs = self.vector_db.search(message, lane_name=lane_name, k=3)
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
                message = f"Context:\n{context}\n\nQuestion: {message}\n\nAnswer based on the context above."
        
        response = self.llm.invoke(message)
        return response.content

if __name__ == "__main__":
    llm = LLM()
    response = llm.send_message("Hello, how are you?")
    print(response)
