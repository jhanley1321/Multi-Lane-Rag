
from rag.vector_db_manager import VectorDBManager
from rag.config_manager import ConfigManager
from rag.document_loader import DocumentLoader
from rag.lane_manager import LaneManager
from rag.vectorstore_manager import VectorStoreManager
from llm_models.llm import LLM


class RAGManager:
    """
    Simple wrapper that holds all the classes in one place.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        model_name: str = "llama3.2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.config = ConfigManager(persist_directory)
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.lanes = LaneManager(self.config)
        self.vectorstore = VectorStoreManager(persist_directory)
        self.vector_db = VectorDBManager(
            config_manager=self.config,
            document_loader=self.loader,
            lane_manager=self.lanes,
            vectorstore_manager=self.vectorstore
        )
        self.llm = LLM(model=model_name)