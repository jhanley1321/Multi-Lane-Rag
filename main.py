from llm import LLM
from cli import ChatCLI
from vector_db_manager import VectorDBManager


def main(): 
    
    vec = VectorDBManager()
    vec.check_database_exists()
    # vec.initialize_database()
    # vec.check_database_exists()
   
    # llm = LLM(model="llama3.2", temperature=0.7)
    # chat = ChatCLI(llm)
    # chat.run()




if __name__ == "__main__":
    main()
