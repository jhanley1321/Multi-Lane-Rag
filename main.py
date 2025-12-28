from llm import LLM
from cli import ChatCLI
from vector_db_manager import VectorDBManager


def main(): 

    # First time setup
    manager = VectorDBManager(persist_directory="./chroma_db")
    manager.run_initialize(
        lanes_config={
            "lane_1": "./data",
            "user_chat": "./chat_logs"
        },
        embedding_model="mxbai-embed-large")

    # Load initial documents
    manager.run_add_documents(lane_name="lane_1", collection_name="restaurant_data")
    manager.run_add_documents(lane_name="user_chat", collection_name="chat_logs")

    # Query the database
    results = manager.run_query("What's the best restaurant?", lane_name="lane_1")

    # Later: add more documents
    manager.run_add_documents(
        lane_name="lane_1",
        file_paths=["./data/realistic_restaurant_reviews.csv"])




if __name__ == "__main__":
    main()
