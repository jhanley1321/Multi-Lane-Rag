# from llm import LLM
# from cli import ChatCLI
from vector_db_manager import VectorDBManager


from rag_manager import RAGManager


def main(): 
    
    
    
    
    # Initialize RAGManager
    rag = RAGManager(persist_directory="./chroma_db")
    
    # Clear existing database to start fresh
    rag.vector_db.clear_database()


    # Initialize database and lanes
    rag.vector_db.run_initialize(
        lanes_config={
            "lane_1": "./data",
            "user_chat": "./chat_logs",
            "test_lane": "./test_data"

        },
        embedding_model="mxbai-embed-large"
    )


    # Add documents to the vector database
    rag.vector_db.run_add_documents(lane_name="test_lane", collection_name="test_lanes")


    # rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="restaurant_data")
    # rag.vector_db.run_add_documents(lane_name="user_chat", collection_name="chat_logs")
    # rag.vector_db.run_add_documents(lane_name="test_lane", collection_name="chat_logs")

   
    # rag.vector_db.run_add_documents(
    #     lane_name="lane_1",
    #     # file_paths=["./data/new_reviews.csv"])
    #     # File_paths=["./data"]
    #     )

    # Query the database
    # Test Lane
    test1 = rag.vector_db.run_query("What is Password does Peter Parker Have?", lane_name="test_lane")
    test2 = rag.vector_db.run_query("What is Password does Bruce Wayne Have?", lane_name="test_lane")

    test3 = rag.vector_db.run_query("What is Password does Peter Parker Have?", lane_name="chat_logs")
    test4 = rag.vector_db.run_query("What is Password does Bruce Wayne Have?", lane_name="lane_1")
    
    test5 = rag.vector_db.run_query("Tell me about a restruant review", lane_name="lane_1")
    test6 = rag.vector_db.run_query("Tell me about a restruant review", lane_name="test_lane")

    test7 = rag.vector_db.run_query("What is Password does Bruce Wayne Have?", lane_name=["test_lane", "lane_1"])


    

    print('==============================================================')
    print('test1:', '/n')
    print(test1)
    print('==============================================================')
    print('test2:', '/n')
    print(test2)
    print('==============================================================')
    print('test3:', '/n')
    print(test3)
    print('==============================================================')
    print('test4:', '/n')
    print(test4)
    print('==============================================================')
    print('test5:', '/n')
    print(test5)
    print('==============================================================')
    print('test6:', '/n')
    print(test6)
    print('==============================================================')
    print('test7:', '/n')
    print(test7)



if __name__ == "__main__":
    main()