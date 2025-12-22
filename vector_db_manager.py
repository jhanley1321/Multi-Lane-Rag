from pathlib import Path
from typing import Optional, List, Dict, Union
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


class VectorDBManager:
    """
    Manages vector database operations with multi-lane support.
    Each lane contains isolated collections for different data sources.
    """

    def __init__(self, vectorstore: Optional[Chroma] = None, persist_directory: str = "./chroma_db"):
        """
        Initialize the VectorDBManager.
        
        Args:
            vectorstore: Optional Chroma vectorstore instance to manage
            persist_directory: Directory where the vector database is stored
        """
        self.vectorstore = vectorstore
        self.persist_directory = persist_directory
        self.lanes: Dict[str, List[str]] = {}
        self.lanes_config_path = Path(persist_directory) / "lanes_config.json"
        self._load_lanes_config()

    def _load_lanes_config(self) -> None:
        """Load lane configuration from disk if it exists."""
        if self.lanes_config_path.exists():
            with open(self.lanes_config_path, 'r') as f:
                self.lanes = json.load(f)

    def _save_lanes_config(self) -> None:
        """Save lane configuration to disk."""
        self.lanes_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lanes_config_path, 'w') as f:
            json.dump(self.lanes, f, indent=2)

    def _load_lane_folders_config(self) -> Dict[str, str]:
        """Load lane folder mappings from config."""
        lane_folders_path = Path(self.persist_directory) / "lane_folders.json"
        if lane_folders_path.exists():
            with open(lane_folders_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_lane_folders_config(self, lane_folders: Dict[str, str]) -> None:
        """Save lane folder mappings to config."""
        lane_folders_path = Path(self.persist_directory) / "lane_folders.json"
        lane_folders_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lane_folders_path, 'w') as f:
            json.dump(lane_folders, f, indent=2)

    def _ensure_database_loaded(self, embedding_model: str) -> None:
        """Ensure the database is loaded, initialize if needed."""
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded. Initializing database...")
            self.initialize_database(embedding_model=embedding_model)

    def _ensure_lane_exists(self, lane_name: str) -> None:
        """Ensure the lane exists, create if needed."""
        if lane_name not in self.lanes:
            print(f"⚠️ Lane '{lane_name}' does not exist. Creating it...")
            self.create_lane(lane_name)

    def _ensure_collection_exists(self, lane_name: str, collection_name: str) -> None:
        """Ensure the collection exists in the lane, add if needed."""
        if collection_name not in self.lanes[lane_name]:
            print(f"⚠️ Collection '{collection_name}' not in lane '{lane_name}'. Adding it...")
            self.add_collection_to_lane(lane_name, collection_name)

    def _get_default_collection_name(self, lane_name: str) -> str:
        """Get default collection name for a lane (uses lane name)."""
        return lane_name

    def _ensure_lane_folder_set(self, lane_name: str, lane_folder: Optional[str] = None) -> str:
        """
        Ensure lane folder is set, either from parameter or from config.
        
        Args:
            lane_name: Name of the lane
            lane_folder: Optional folder path to use
            
        Returns:
            The folder path for the lane
            
        Raises:
            ValueError: If no folder is set and none provided
        """
        if lane_folder:
            return lane_folder
        
        folder = self.get_lane_folder(lane_name)
        if not folder:
            raise ValueError(
                f"No folder set for lane '{lane_name}'. "
                f"Please provide lane_folder parameter or use set_lane_folder() first."
            )
        return folder

    def _load_csv_file(self, file_path: str) -> List:
        """
        Load a CSV file into documents.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            List of documents loaded from the CSV
        """
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        return documents


    def _load_json_file(self, file_path: str) -> List:
        """
        Load a JSON file into documents.
        Handles arbitrary JSON structures by converting the entire content to a string.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of documents loaded from the JSON
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        json_string = json.dumps(json_data, indent=2)
        
        doc = Document(
            page_content=json_string,
            metadata={"source": file_path}
        )
        
        return [doc]


    def _load_json_file(self, file_path: str) -> List:
        """
        Load a JSON file into documents.
        Handles arbitrary JSON structures by converting the entire content to a string.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of documents loaded from the JSON
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        json_string = json.dumps(json_data, indent=2)
        
        doc = Document(
            page_content=json_string,
            metadata={"source": file_path}
        )
        
        return [doc]

    def _chunk_documents(self, documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
        """
        Chunk documents into smaller pieces.
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def _add_metadata_to_documents(self, documents: List, lane_name: str, collection_name: str, file_path: str) -> List:
        """
        Add metadata to documents.
        
        Args:
            documents: List of documents
            lane_name: Name of the lane
            collection_name: Name of the collection
            file_path: Path to the source file
            
        Returns:
            List of documents with added metadata
        """
        for doc in documents:
            doc.metadata['lane'] = lane_name
            doc.metadata['collection'] = collection_name
            doc.metadata['source_file'] = Path(file_path).name
        return documents

    def _add_documents_to_collection(self, documents: List) -> None:
        """
        Add documents to the vectorstore.
        
        Args:
            documents: List of documents to add
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded")
            return
        
        self.vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} chunks to vectorstore")

    def _load_document_file(self, file_path: str, lane_name: str, collection_name: str) -> None:
        """
        Load a document file into the vectorstore.
        Orchestrates loading, chunking, metadata addition, and storage.
        
        Args:
            file_path: Path to the document file
            lane_name: Name of the lane
            collection_name: Name of the collection
        """
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            print(f"⚠️ File not found: {file_path}")
            return
        
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.csv':
            documents = self._load_csv_file(file_path)
        elif file_extension == '.json':
            documents = self._load_json_file(file_path)
        else:
            print(f"⚠️ Unsupported file type: {file_extension}")
            return
        
        chunks = self._chunk_documents(documents)
        chunks = self._add_metadata_to_documents(chunks, lane_name, collection_name, file_path)
        self._add_documents_to_collection(chunks)

    def set_vectorstore(self, vectorstore: Chroma) -> None:
        """
        Set the vectorstore instance.
        
        Args:
            vectorstore: Chroma vectorstore instance
        """
        self.vectorstore = vectorstore
        print("✅ Vectorstore set successfully")

    def check_database_exists(self) -> bool:
        """
        Check if the vector database exists.
        
        Returns:
            True if database exists, False otherwise
        """
        db_path = Path(self.persist_directory)
        exists = db_path.exists() and any(db_path.iterdir())
        
        if exists:
            print(f"✅ Database exists at: {self.persist_directory}")
        else:
            print(f"⚠️ Database does not exist at: {self.persist_directory}")
        
        return exists

    def initialize_database(self, embedding_model: str = "nomic-embed-text") -> None:
        """
        Initialize a new vector database.
        
        Args:
            embedding_model: Name of the embedding model to use
        """
        print(f"🔧 Initializing database at: {self.persist_directory}")
        print(f"📦 Using embedding model: {embedding_model}")
        
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        
        print("✅ Database initialized successfully")

    def create_lane(self, lane_name: str) -> None:
        """
        Create a new lane.
        
        Args:
            lane_name: Name of the lane to create
        """
        if lane_name in self.lanes:
            print(f"⚠️ Lane '{lane_name}' already exists")
            return
        
        self.lanes[lane_name] = []
        self._save_lanes_config()
        print(f"✅ Created lane: {lane_name}")

    def get_lane_collections(self, lane_name: str) -> List[str]:
        """
        Get all collections in a lane.
        
        Args:
            lane_name: Name of the lane
            
        Returns:
            List of collection names in the lane
        """
        if lane_name not in self.lanes:
            print(f"⚠️ Lane '{lane_name}' does not exist")
            return []
        
        return self.lanes[lane_name]

    def add_collection_to_lane(self, lane_name: str, collection_name: str) -> None:
        """
        Add a collection to a lane.
        
        Args:
            lane_name: Name of the lane
            collection_name: Name of the collection to add
        """
        if lane_name not in self.lanes:
            print(f"⚠️ Lane '{lane_name}' does not exist. Create it first.")
            return
        
        if collection_name in self.lanes[lane_name]:
            print(f"⚠️ Collection '{collection_name}' already exists in lane '{lane_name}'")
            return
        
        self.lanes[lane_name].append(collection_name)
        self._save_lanes_config()
        print(f"✅ Added collection '{collection_name}' to lane '{lane_name}'")

    def set_lane_folder(self, lane_name: str, folder_path: str) -> None:
        """
        Set the folder path for a lane.
        
        Args:
            lane_name: Name of the lane
            folder_path: Path to the folder containing lane documents
        """
        if lane_name not in self.lanes:
            print(f"⚠️ Lane '{lane_name}' does not exist. Create it first.")
            return
        
        lane_folders = self._load_lane_folders_config()
        lane_folders[lane_name] = folder_path
        self._save_lane_folders_config(lane_folders)
        print(f"✅ Set folder for lane '{lane_name}': {folder_path}")

    def get_lane_folder(self, lane_name: str) -> Optional[str]:
        """
        Get the folder path for a lane.
        
        Args:
            lane_name: Name of the lane
            
        Returns:
            Folder path for the lane, or None if not set
        """
        lane_folders = self._load_lane_folders_config()
        return lane_folders.get(lane_name)

    def list_lanes(self) -> None:
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

    def view_contents(self) -> None:
        """View the contents of the vector database."""
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded")
            return
        
        print("\n📊 Database Contents:")
        collection = self.vectorstore._collection
        print(f"Total documents: {collection.count()}")

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded")
            return {}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        stats = {
            "total_documents": count,
            "lanes": len(self.lanes),
            "persist_directory": self.persist_directory
        }
        
        print("\n📊 Database Statistics:")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Total lanes: {stats['lanes']}")
        print(f"  Persist directory: {stats['persist_directory']}")
        
        return stats

    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded")
            return
        
        import shutil
        if Path(self.persist_directory).exists():
            shutil.rmtree(self.persist_directory)
            print(f"✅ Cleared database at: {self.persist_directory}")
        
        self.vectorstore = None
        self.lanes = {}

    def run_load_documents(
        self, 
        lane_name: str, 
        collection_name: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        lane_folder: Optional[str] = None,
        embedding_model: str = "nomic-embed-text"
    ) -> None:
        """
        Run method to load documents into a lane and collection.
        Handles the full workflow: setup database, lane, folder, collection, then load documents.
        
        Args:
            lane_name: Name of the lane to load into
            collection_name: Name of the collection (defaults to lane name if not provided)
            file_paths: List of specific file paths to load (if None, loads all files from lane folder)
            lane_folder: Folder path for the lane (uses default if not provided)
            embedding_model: Embedding model to use if database needs initialization
        """
        print(f"\n🚀 Starting document load workflow for lane: {lane_name}")
        
        self._ensure_database_loaded(embedding_model)
        self._ensure_lane_exists(lane_name)
        
        if lane_folder:
            self.set_lane_folder(lane_name, lane_folder)
        
        folder = self._ensure_lane_folder_set(lane_name, lane_folder)
        print(f"📁 Lane folder: {folder}")
        
        if collection_name is None:
            collection_name = lane_name
            print(f"📋 Using lane name as collection name: {collection_name}")
        
        self._ensure_collection_exists(lane_name, collection_name)
        
        if file_paths is None:
            folder_path = Path(folder)
            if not folder_path.exists():
                print(f"⚠️ Lane folder does not exist: {folder}")
                print(f"   Please create the folder and add files, or specify file_paths explicitly")
                return
            
            file_paths = [str(f) for f in folder_path.iterdir() if f.is_file()]
            print(f"📋 Found {len(file_paths)} files in lane folder")
        
        if not file_paths:
            print(f"⚠️ No files to load")
            return
        
        print(f"📋 Loading {len(file_paths)} document(s)...")
        for file_path in file_paths:
            print(f"  Processing: {Path(file_path).name}")
            self._load_document_file(file_path, lane_name, collection_name)
        
        print(f"\n✅ Document loading workflow complete!")

    def search(self, query: str, lane_name: Optional[Union[str, List[str]]] = None, k: int = 3) -> List:
        """
        Search for relevant documents in the vector database.
        
        Args:
            query: The search query
            lane_name: Lane(s) to search in. Can be:
                       - None or "all": Search all lanes (default)
                       - str: Search specific lane (e.g., "lane_1")
                       - List[str]: Search multiple lanes (e.g., ["lane_1", "user_chat"])
            k: Number of documents to return
        
        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore loaded. Please initialize the database first.")
            return []
        
        results = self.vectorstore.similarity_search(query=query, k=k)
        
        if lane_name is None or lane_name == "all":
            return results
        
        if isinstance(lane_name, str):
            results = [doc for doc in results if doc.metadata.get('lane') == lane_name]
        elif isinstance(lane_name, list):
            results = [doc for doc in results if doc.metadata.get('lane') in lane_name]
        
        return results
