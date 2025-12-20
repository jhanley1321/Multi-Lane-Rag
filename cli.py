class ChatCLI:
    def __init__(self, llm):
        self.llm = llm
    
    def run(self):
        print("Chat CLI - Type 'exit' or 'quit' to end the conversation\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = self.llm.send_message(user_input)
            print(f"Assistant: {response}\n")
