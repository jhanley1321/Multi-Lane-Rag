from vector_db_manager import VectorDBManager

if __name__ == "__main__":
    manager = VectorDBManager(persist_directory="./test_chroma_db")
    
    print("=== Testing VectorDBManager ===\n")
    
    print("1. Testing database initialization...")
    manager.check_database_exists()
    
    print("\n2. Creating lanes...")
    manager.create_lane("lane_1")
    manager.create_lane("lane_2")
    
    print("\n3. Setting lane folders...")
    manager.set_lane_folder("lane_1", "./data")
    manager.set_lane_folder("lane_2", "./chroma_db")
    
    print("\n4. Listing lanes...")
    manager.list_lanes()
    
    print("\n5. Testing run_load_documents workflow...")
    manager.run_load_documents(
        lane_name="lane_1",
        collection_name="test_docs",
        lane_folder="./data",
        embedding_model="mxbai-embed-large"
    )
    
    print("\n=== Test Complete ===")
