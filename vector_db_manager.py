from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


class VectorDBManager:
    """
    Manages vector database operations like viewing, inspecting, and managing contents.
    Separate from RAG to maintain single responsibility principle.
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

    def set_vectorstore(self, vectorstore: Chroma) -> None:
        """
        Attach a vectorstore to manage.
        
        Args:
            vectorstore: Chroma vectorstore instance
        """
        self.vectorstore = vectorstore

    def check_database_exists(self) -> bool:
        """
        Check if the vector database exists in the persistence directory.
        
        Returns:
            bool: True if database exists, False otherwise
        """
        db_path = Path(self.persist_directory)
        exists = db_path.exists() and db_path.is_dir() and any(db_path.iterdir())
        
        if exists:
            print(f"✅ Vector database found at: {self.persist_directory}")
        else:
            print(f"❌ Vector database not found at: {self.persist_directory}")
        
        return exists

    def initialize_database(self, embedding_model: str = "nomic-embed-text") -> Chroma:
        """
        Initialize/create a new vector database.
        
        Args:
            embedding_model: Name of the Ollama embedding model to use
            
        Returns:
            Chroma: The initialized vectorstore instance
        """
        print(f"🔧 Initializing vector database at: {self.persist_directory}")
        
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )
        
        print(f"✅ Vector database initialized successfully")
        return self.vectorstore

    def view_contents(self, limit: int = 10) -> None:
        """
        View the contents and metadata of documents in the vector database.
        
        Args:
            limit: Maximum number of documents to display (default: 10)
        
        Displays:
        - Document type (chat_log, csv_data, etc.)
        - Source file
        - Preview of content
        - All metadata fields
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore attached. Use set_vectorstore() first.")
            return
        
        collection = self.vectorstore._collection
        results = collection.get(limit=limit, include=["documents", "metadatas"])
        
        if not results["ids"]:
            print("📭 Vector database is empty")
            return
        
        print(f"\n📊 Vector Database Contents (showing {len(results['ids'])} documents):\n")
        
        for i, (doc_id, content, metadata) in enumerate(zip(results["ids"], results["documents"], results["metadatas"]), 1):
            doc_type = metadata.get("type", "unknown")
            source = Path(metadata.get("source", "unknown")).name
            
            print(f"Document {i}:")
            print(f"  ID: {doc_id}")
            print(f"  Type: {doc_type}")
            print(f"  Source: {source}")
            print(f"  Metadata: {metadata}")
            print(f"  Content Preview: {content[:150]}...")
            print()

    def get_stats(self) -> None:
        """
        Display statistics about the vector database.
        
        Shows:
        - Total number of documents
        - Breakdown by document type
        - List of unique sources
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore attached. Use set_vectorstore() first.")
            return
        
        collection = self.vectorstore._collection
        results = collection.get(include=["metadatas"])
        
        if not results["ids"]:
            print("📭 Vector database is empty")
            return
        
        total_docs = len(results["ids"])
        type_counts = {}
        sources = set()
        
        for metadata in results["metadatas"]:
            doc_type = metadata.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            sources.add(metadata.get("source", "unknown"))
        
        print(f"\n📊 Vector Database Statistics:\n")
        print(f"Total Documents: {total_docs}")
        print(f"\nDocument Types:")
        for doc_type, count in type_counts.items():
            print(f"  {doc_type}: {count}")
        print(f"\nUnique Sources: {len(sources)}")
        for source in sorted(sources):
            print(f"  - {Path(source).name}")
        print()

    def clear_database(self) -> None:
        """
        Clear all documents from the vector database.
        
        Warning: This operation cannot be undone!
        """
        if not self.vectorstore:
            print("⚠️ No vectorstore attached. Use set_vectorstore() first.")
            return
        
        collection = self.vectorstore._collection
        results = collection.get()
        
        if not results["ids"]:
            print("📭 Vector database is already empty")
            return
        
        count = len(results["ids"])
        collection.delete(ids=results["ids"])
        print(f"🗑️ Cleared {count} documents from the vector database")
