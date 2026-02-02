from pathlib import Path
from typing import Optional, List, Dict, Union
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorDBManager:
    """
    Main interface for vector database operations.
    Orchestrates ConfigManager, DocumentLoader, LaneManager, and VectorStoreManager.
    """
    
    def __init__(
        self,
        config_manager,
        document_loader,
        lane_manager,
        vectorstore_manager
    ):
        self.config_manager = config_manager
        self.document_loader = document_loader
        self.lane_manager = lane_manager
        self.vectorstore_manager = vectorstore_manager
        self.persist_directory = vectorstore_manager.persist_directory
    
    def run_initialize(
        self,
        lanes_config: Dict[str, str],
        embedding_model: str = "nomic-embed-text",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> None:
        """Initialize database and set up lanes."""
        print("\n🔧 Initializing vector database system...")

        self.document_loader.chunk_size = chunk_size
        self.document_loader.chunk_overlap = chunk_overlap
        self.document_loader.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not self.vectorstore_manager.vectorstore:
            self.vectorstore_manager.initialize(embedding_model=embedding_model)
        else:
            print("✅ Database already initialized")

        for lane_name, folder_path in lanes_config.items():
            if not self.lane_manager.lane_exists(lane_name):
                self.lane_manager.create_lane(lane_name)
            else:
                print(f"⚠️ Lane '{lane_name}' already exists")

            self.lane_manager.set_lane_folder(lane_name, folder_path)

        print("\n✅ Initialization complete!")
        self.lane_manager.list_all()
    
    def run_add_documents(
        self,
        lane_name: str,
        collection_name: Optional[str] = None,
        file_paths: Optional[List[str]] = None
    ) -> None:
        """Add documents to an existing lane and collection."""
        print(f"\n📥 Adding documents to lane: {lane_name}")
        
        if not self.vectorstore_manager.vectorstore:
            print("❌ Error: No vectorstore loaded. Run run_initialize() first.")
            return
        
        if not self.lane_manager.lane_exists(lane_name):
            print(f"❌ Error: Lane '{lane_name}' does not exist. Run run_initialize() first.")
            return
        
        if collection_name is None:
            collection_name = lane_name
            print(f"📋 Using lane name as collection name: {collection_name}")
        
        if not self.lane_manager.collection_exists(lane_name, collection_name):
            self.lane_manager.add_collection(lane_name, collection_name)
        
        if file_paths is None:
            folder = self.lane_manager.get_lane_folder(lane_name)
            if not folder:
                print(f"❌ Error: No folder set for lane '{lane_name}'")
                return
            
            folder_path = Path(folder)
            if not folder_path.exists():
                print(f"❌ Error: Lane folder does not exist: {folder}")
                return
            
            file_paths = [str(f) for f in folder_path.iterdir() if f.is_file()]
            print(f"📋 Found {len(file_paths)} files in lane folder: {folder}")
        
        if not file_paths:
            print(f"⚠️ No files to load")
            return
        
        print(f"📋 Loading {len(file_paths)} document(s)...")
        for file_path in file_paths:
            print(f"  Processing: {Path(file_path).name}")
            self._process_file(file_path, lane_name, collection_name)
        
        print(f"\n✅ Documents added successfully!")
    
    def run_query(
        self,
        query: str,
        lane_name: Optional[Union[str, List[str]]] = None,
        k: int = 3
    ) -> List[Document]:
        """Query the vector database."""
        print(f"\n🔍 Querying database...")
        
        if not self.vectorstore_manager.vectorstore:
            print("❌ Error: No vectorstore loaded. Run run_initialize() first.")
            return []
        
        results = self.vectorstore_manager.search(query=query, lane_name=lane_name, k=k)
        print(f"✅ Found {len(results)} relevant document(s)")
        
        return results
    
    def _process_file(self, file_path: str, lane_name: str, collection_name: str) -> None:
        """Process a single file: load, chunk, add metadata, store."""
        try:
            documents = self.document_loader.load_file(file_path)
            chunks = self.document_loader.chunk_documents(documents)
            chunks = self.document_loader.add_metadata(
                chunks, lane_name, collection_name, file_path
            )
            self.vectorstore_manager.add_documents(chunks)
            
        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
    
    def list_lanes(self) -> None:
        """List all lanes and their collections."""
        self.lane_manager.list_all()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        stats = self.vectorstore_manager.get_stats()
        stats['lanes'] = len(self.lane_manager.lanes)
        stats['persist_directory'] = self.persist_directory
        
        print("\n📊 Database Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total lanes: {stats['lanes']}")
        print(f"  Persist directory: {stats['persist_directory']}")
        
        return stats
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        self.vectorstore_manager.clear()
        self.lane_manager.lanes = {}
        self.lane_manager.lane_folders = {}
