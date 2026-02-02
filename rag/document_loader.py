from pathlib import Path
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import json


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
