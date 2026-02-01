from pathlib import Path
from typing import Optional, List, Dict, Union
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json


class ConfigManager:
    """Manages configuration persistence for lanes and folders."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = Path(persist_directory)
        self.lanes_config_path = self.persist_directory / "lanes_config.json"
        self.lane_folders_path = self.persist_directory / "lane_folders.json"
    
    def load_lanes(self) -> Dict[str, List[str]]:
        """Load lane configuration."""
        if self.lanes_config_path.exists():
            with open(self.lanes_config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_lanes(self, lanes: Dict[str, List[str]]) -> None:
        """Save lane configuration."""
        self.lanes_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lanes_config_path, 'w') as f:
            json.dump(lanes, f, indent=2)
    
    def load_lane_folders(self) -> Dict[str, str]:
        """Load lane folder mappings."""
        if self.lane_folders_path.exists():
            with open(self.lane_folders_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_lane_folders(self, lane_folders: Dict[str, str]) -> None:
        """Save lane folder mappings."""
        self.lane_folders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lane_folders_path, 'w') as f:
            json.dump(lane_folders, f, indent=2)


class DocumentLoader:
    """Loads and processes documents from various file formats."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_file(self, file_path: str) -> List[Document]:
        """Load a file based on its extension."""
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.csv':
            return self._load_csv(file_path)
        elif file_extension == '.json':
            return self._load_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV file."""
        loader = CSVLoader(file_path=file_path)
        return loader.load()
    
    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        json_string = json.dumps(json_data, indent=2)
        doc = Document(
            page_content=json_string,
            metadata={"source": file_path}
        )
        return [doc]
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk documents into smaller pieces."""
        return self.text_splitter.split_documents(documents)
    
    def add_metadata(
        self, 
        documents: List[Document], 
        lane_name: str, 
        collection_name: str, 
        file_path: str
    ) -> List[Document]:
        """Add metadata to documents."""
        for doc in documents:
            doc.metadata['lane'] = lane_name
            doc.metadata['collection'] = collection_name
            doc.metadata['source_file'] = Path(file_path).name
        return documents


class LaneManager:
    """Manages lanes and their collections."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.lanes: Dict[str, List[str]] = config_manager.load_lanes()
        self.lane_folders: Dict[str, str] = config_manager.load_lane_folders()
    
    def create_lane(self, lane_name: str) -> None:
        """Create a new lane."""
        if lane_name in self.lanes:
            print(f"⚠️ Lane '{lane_name}' already exists")
            return
        
        self.lanes[lane_name] = []
        self.config_manager.save_lanes(self.lanes)
        print(f"✅ Created lane: {lane_name}")
    
    def lane_exists(self, lane_name: str) -> bool:
        """Check if lane exists."""
        return lane_name in self.lanes
    
    def add_collection(self, lane_name: str, collection_name: str) -> None:
        """Add a collection to a lane."""
        if lane_name not in self.lanes:
            raise ValueError(f"Lane '{lane_name}' does not exist")
        
        if collection_name in self.lanes[lane_name]:
            print(f"⚠️ Collection '{collection_name}' already exists in lane '{lane_name}'")
            return
        
        self.lanes[lane_name].append(collection_name)
        self.config_manager.save_lanes(self.lanes)
        print(f"✅ Added collection '{collection_name}' to lane '{lane_name}'")
    
    def collection_exists(self, lane_name: str, collection_name: str) -> bool:
        """Check if collection exists in lane."""
        return lane_name in self.lanes and collection_name in self.lanes[lane_name]
    
    def get_collections(self, lane_name: str) -> List[str]:
        """Get all collections in a lane."""
        return self.lanes.get(lane_name, [])
    
    def set_lane_folder(self, lane_name: str, folder_path: str) -> None:
        """Set folder path for a lane."""
        if lane_name not in self.lanes:
            raise ValueError(f"Lane '{lane_name}' does not exist")
        
        self.lane_folders[lane_name] = folder_path
        self.config_manager.save_lane_folders(self.lane_folders)
        print(f"✅ Set folder for lane '{lane_name}': {folder_path}")
    
    def get_lane_folder(self, lane_name: str) -> Optional[str]:
        """Get folder path for a lane."""
        return self.lane_folders.get(lane_name)
    
    def list_all(self) -> None:
        """List all lanes and their collections."""
        if not self.lanes:
            print("⚠️ No lanes exist")
            return
        
        print("\n📋 Lanes:")
        for lane_name, collections in self.lanes.items():
            folder = self.get_lane_folder(lane_name)
            folder_info = f" (folder: {folder})" if folder else ""
            print(f"  • {lane_name}{folder_info}")
            if collections:
                for collection in collections:
                    print(f"    - {collection}")
            else:
                print(f"    (no collections)")


class VectorStoreManager:
    """Manages Chroma vectorstore operations."""
    
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.vectorstore: Optional[Chroma] = None
    
    def exists(self) -> bool:
        """Check if database exists."""
        db_path = Path(self.persist_directory)
        exists = db_path.exists() and any(db_path.iterdir())
        
        if exists:
            print(f"✅ Database exists at: {self.persist_directory}")
        else:
            print(f"⚠️ Database does not exist at: {self.persist_directory}")
        
        return exists
    
    def initialize(self, embedding_model: str = "nomic-embed-text") -> None:
        """Initialize a new vector database."""
        print(f"🔧 Initializing database at: {self.persist_directory}")
        print(f"📦 Using embedding model: {embedding_model}")
        
        embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        
        print("✅ Database initialized successfully")
    
    def load(self, embedding_model: str = "nomic-embed-text") -> None:
        """Load an existing database."""
        if not self.exists():
            raise ValueError("Cannot load database - it does not exist")
        
        print(f"📂 Loading existing database from: {self.persist_directory}")
        print(f"📦 Using embedding model: {embedding_model}")
        
        embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        
        print("✅ Database loaded successfully")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to vectorstore."""
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded")
        
        self.vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} chunks to vectorstore")
    
    def search(
        self,
        query: str,
        lane_name: Optional[Union[str, List[str]]] = None,
        k: int = 3
    ) -> List[Document]:
        """Search the vectorstore."""
        if not self.vectorstore:
            raise ValueError("No vectorstore loaded")
        
        if lane_name is None or lane_name == "all":
            return self.vectorstore.similarity_search(query, k=k)
        elif isinstance(lane_name, str):
            return self.vectorstore.similarity_search(
                query, k=k, filter={"lane": lane_name}
            )
        elif isinstance(lane_name, list):
            return self.vectorstore.similarity_search(
                query, k=k, filter={"lane": {"$in": lane_name}}
            )
        else:
            raise ValueError(f"Invalid lane_name type: {type(lane_name)}")
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        if not self.vectorstore:
            return {"total_documents": 0}
        
        collection = self.vectorstore._collection
        return {"total_documents": collection.count()}
    
    def clear(self) -> None:
        """Clear all data from the database."""
        import shutil
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory)
            print(f"✅ Cleared database at: {self.persist_directory}")
        
        self.vectorstore = None


class VectorDBManager:
    """
    Main interface for vector database operations.
    Orchestrates ConfigManager, DocumentLoader, LaneManager, and VectorStoreManager.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        self.config_manager = ConfigManager(persist_directory)
        self.document_loader = DocumentLoader()
        self.lane_manager = LaneManager(self.config_manager)
        self.vectorstore_manager = VectorStoreManager(persist_directory)
    
    def run_initialize(
        self,
        lanes_config: Dict[str, str],
        embedding_model: str = "nomic-embed-text"
    ) -> None:
        """Initialize database and set up lanes."""
        print("\n🔧 Initializing vector database system...")
        
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
