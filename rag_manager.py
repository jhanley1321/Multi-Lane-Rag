
from vector_db_manager import VectorDBManager, ConfigManager, DocumentLoader, LaneManager, VectorStoreManager
from llm import LLM


class RAGManager:
    """
    Simple wrapper that holds all the classes in one place.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.vector_db = VectorDBManager(persist_directory=persist_directory)
        self.config = self.vector_db.config_manager
        self.loader = self.vector_db.document_loader
        self.lanes = self.vector_db.lane_manager
        self.vectorstore = self.vector_db.vectorstore_manager
        self.llm = LLM()