from pathlib import Path
from typing import Optional, List, Dict, Union
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


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
