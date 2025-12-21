from llm import LLM
from cli import ChatCLI
from vector_db_manager import VectorDBManager


def main(): 
    db_manager = VectorDBManager(persist_directory="./chroma_db")
    
    # db_manager.run_load_documents(
    #     lane_name="lane_1",
    #     collection_name="restaurant_data",
    #     lane_folder="./data",
    #     embedding_model="mxbai-embed-large"
    # )
    
    # db_manager.run_load_documents(
    #     lane_name="user_chat",
    #     collection_name="chat_logs",
    #     lane_folder="./chat_logs",
    #     embedding_model="mxbai-embed-large"
    # )
    
    db_manager.list_lanes()
    
    llm = LLM(model="llama3.2", temperature=0.7, vector_db=db_manager)
    chat = ChatCLI(llm)
    chat.run()


if __name__ == "__main__":
    main()
