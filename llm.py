from langchain_ollama import ChatOllama

class LLM:
    def __init__(self, model: str = "llama3.2", temperature: float = 0.7):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
    
    def send_message(self, message: str) -> str:
        response = self.llm.invoke(message)
        return response.content

if __name__ == "__main__":
    llm = LLM()
    response = llm.send_message("Hello, how are you?")
    print(response)
