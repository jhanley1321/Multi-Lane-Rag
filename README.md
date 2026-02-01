# Multi-Lane RAG System

A modular Retrieval-Augmented Generation (RAG) system with multi-lane support for organizing and querying different data sources independently.

## Table of Contents
- [What are RAG Lanes?](#what-are-rag-lanes)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## What are RAG Lanes?

**RAG Lanes** are isolated data channels within a single vector database (or lane be database) that allow you to:

- **Organize data by source**: Keep different data types separate and organized
- **Query specific lanes**: Search only relevant data sources instead of the entire database
- **Maintain data isolation**: Each lane has its own collections and folder mappings
- **Scale efficiently**: Add new data sources without affecting existing lanes

### Why Use Lanes?

Without lanes, all your data is mixed together in one vector database. With lanes, you can:

```python
# Query only data from lane_1
results = rag.vector_db.run_query("search query", lane_name="lane_1")

# Query only data from lane_2
results = rag.vector_db.run_query("search query", lane_name="lane_2")

# Query multiple specific lanes
results = rag.vector_db.run_query("search query", lane_name=["lane_1", "lane_2"])

# Query all lanes
results = rag.vector_db.run_query("search query", lane_name=None)
```

---

## Architecture

The system is built with clean separation of concerns using 5 core classes:

```
RAGManager (Wrapper)
    ├── VectorDBManager (Main Orchestrator)
    │   ├── ConfigManager (Configuration persistence)
    │   ├── DocumentLoader (File loading & chunking)
    │   ├── LaneManager (Lane & collection management)
    │   └── VectorStoreManager (Vector database operations)
    └── LLM (Language model interface)
```

### Class Responsibilities

1. **ConfigManager**: Handles all configuration persistence (lanes, folders)
2. **DocumentLoader**: Loads CSV/JSON files and chunks documents
3. **LaneManager**: Manages lanes, collections, and folder mappings
4. **VectorStoreManager**: Handles vector database operations (initialize, search, add documents)
5. **VectorDBManager**: Main orchestrator that coordinates all managers
6. **RAGManager**: Simple wrapper providing unified access to all components

---

## Installation

```bash
pip install langchain langchain-chroma langchain-ollama langchain-community
```

**Required dependencies:**
- `langchain`
- `langchain-chroma`
- `langchain-ollama`
- `langchain-community`
- `ollama` (for embeddings)

---

## Quick Start

```python
from rag_manager import RAGManager

# 1. Initialize the system
rag = RAGManager(persist_directory="./chroma_db")

# 2. Set up lanes and database
rag.vector_db.run_initialize(
    lanes_config={
        "lane_1": "./data/source_1",
        "lane_2": "./data/source_2",
        "lane_3": "./data/source_3"
    },
    embedding_model="mxbai-embed-large"
)

# 3. Add documents to lanes
rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")
rag.vector_db.run_add_documents(lane_name="lane_2", collection_name="collection_2")

# 4. Query specific lanes
results = rag.vector_db.run_query("your search query", lane_name="lane_1", k=5)

# 5. Use with LLM
context = "\n\n".join([doc.page_content for doc in results])
answer = rag.llm.generate(f"Context: {context}\n\nQuestion: Your question here?")
```

---

## Core Concepts

### Lanes

A **lane** is an isolated data channel with:
- A unique name
- One or more collections
- A folder path for loading documents

```python
# Create a lane
rag.lanes.create_lane("lane_1")

# Set folder for the lane
rag.lanes.set_lane_folder("lane_1", "./data/source_1")

# Add collections to the lane
rag.lanes.add_collection_to_lane("lane_1", "collection_1")
```

### Collections

A **collection** is a named group of documents within a lane. Each lane can have multiple collections.

```python
# Add multiple collections to a lane
rag.lanes.add_collection_to_lane("lane_1", "collection_1")
rag.lanes.add_collection_to_lane("lane_1", "collection_2")

# View all collections in a lane
collections = rag.lanes.get_lane_collections("lane_1")
```

### Folder Mappings

Each lane has a **folder path** where documents are stored. When you call `run_add_documents()` without specifying files, it loads all CSV/JSON files from the lane's folder.

```python
# Set folder for a lane
rag.lanes.set_lane_folder("lane_1", "./data/source_1")

# Load all files from the lane's folder
rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")
```

### Metadata

All documents are automatically tagged with metadata:
- `lane`: The lane name
- `collection`: The collection name
- `source`: The file path

This metadata enables precise filtering during queries.

---

## API Reference

### RAGManager

The main wrapper class providing unified access to all components.

#### `__init__(persist_directory="./chroma_db", model_name="llama3.2")`

Initialize the RAG system.

**Parameters:**
- `persist_directory` (str): Path to store the vector database
- `model_name` (str): Name of the Ollama model for LLM

**Attributes:**
- `vector_db`: VectorDBManager instance
- `config`: ConfigManager instance
- `loader`: DocumentLoader instance
- `lanes`: LaneManager instance
- `vectorstore`: VectorStoreManager instance
- `llm`: LLM instance

**Example:**
```python
rag = RAGManager(persist_directory="./chroma_db", model_name="llama3.2")
```

---

### VectorDBManager

Main orchestrator for all vector database operations.

#### `run_initialize(lanes_config, embedding_model="mxbai-embed-large", chunk_size=1000, chunk_overlap=200)`

Initialize the database and create lanes.

**Parameters:**
- `lanes_config` (dict): Dictionary mapping lane names to folder paths
- `embedding_model` (str): Ollama embedding model name
- `chunk_size` (int): Size of text chunks
- `chunk_overlap` (int): Overlap between chunks

**Returns:** None

**Example:**
```python
rag.vector_db.run_initialize(
    lanes_config={
        "lane_1": "./data/source_1",
        "lane_2": "./data/source_2"
    },
    embedding_model="mxbai-embed-large",
    chunk_size=1000,
    chunk_overlap=200
)
```

#### `run_add_documents(lane_name, collection_name, file_paths=None)`

Add documents to a lane's collection.

**Parameters:**
- `lane_name` (str): Name of the lane
- `collection_name` (str): Name of the collection
- `file_paths` (list, optional): Specific files to load. If None, loads all files from lane's folder

**Returns:** None

**Example:**
```python
# Load all files from lane's folder
rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")

# Load specific files
rag.vector_db.run_add_documents(
    lane_name="lane_1",
    collection_name="collection_1",
    file_paths=["./data/file1.csv", "./data/file2.json"]
)
```

#### `run_query(query, lane_name=None, k=5)`

Query the vector database.

**Parameters:**
- `query` (str): Search query
- `lane_name` (str, list, optional): Lane(s) to query. If None, queries all lanes
- `k` (int): Number of results to return

**Returns:** List of Document objects

**Example:**
```python
# Query specific lane
results = rag.vector_db.run_query("search query", lane_name="lane_1", k=5)

# Query multiple lanes
results = rag.vector_db.run_query("search query", lane_name=["lane_1", "lane_2"], k=5)

# Query all lanes
results = rag.vector_db.run_query("search query", lane_name=None, k=5)
```

#### `list_lanes()`

List all lanes and their collections.

**Returns:** Dictionary mapping lane names to lists of collections

**Example:**
```python
lanes = rag.vector_db.list_lanes()
# Output: {"lane_1": ["collection_1", "collection_2"], "lane_2": ["collection_3"]}
```

#### `get_stats()`

Get database statistics.

**Returns:** Dictionary with database stats

**Example:**
```python
stats = rag.vector_db.get_stats()
print(f"Total documents: {stats['total_documents']}")
```

#### `clear_database()`

Delete all data from the vector database.

**Returns:** None

**Example:**
```python
rag.vector_db.clear_database()
```

---

### ConfigManager

Handles configuration persistence.

#### `load_lanes_config()`

Load lanes configuration.

**Returns:** Dictionary of lanes and their collections

#### `save_lanes_config(lanes)`

Save lanes configuration.

**Parameters:**
- `lanes` (dict): Lanes configuration to save

#### `load_lane_folders_config()`

Load lane folder mappings.

**Returns:** Dictionary mapping lane names to folder paths

#### `save_lane_folders_config(lane_folders)`

Save lane folder mappings.

**Parameters:**
- `lane_folders` (dict): Lane folder mappings to save

---

### DocumentLoader

Handles file loading and document chunking.

#### `load_csv_file(file_path)`

Load documents from a CSV file.

**Parameters:**
- `file_path` (str): Path to CSV file

**Returns:** List of Document objects

#### `load_json_file(file_path)`

Load documents from a JSON file.

**Parameters:**
- `file_path` (str): Path to JSON file

**Returns:** List of Document objects

#### `chunk_documents(documents, chunk_size=1000, chunk_overlap=200)`

Split documents into chunks.

**Parameters:**
- `documents` (list): List of Document objects
- `chunk_size` (int): Size of each chunk
- `chunk_overlap` (int): Overlap between chunks

**Returns:** List of chunked Document objects

---

### LaneManager

Manages lanes, collections, and folder mappings.

#### `create_lane(lane_name)`

Create a new lane.

**Parameters:**
- `lane_name` (str): Name of the lane

**Returns:** None

#### `add_collection_to_lane(lane_name, collection_name)`

Add a collection to a lane.

**Parameters:**
- `lane_name` (str): Name of the lane
- `collection_name` (str): Name of the collection

**Returns:** None

#### `get_lane_collections(lane_name)`

Get all collections in a lane.

**Parameters:**
- `lane_name` (str): Name of the lane

**Returns:** List of collection names

#### `set_lane_folder(lane_name, folder_path)`

Set the folder path for a lane.

**Parameters:**
- `lane_name` (str): Name of the lane
- `folder_path` (str): Path to the folder

**Returns:** None

#### `get_lane_folder(lane_name)`

Get the folder path for a lane.

**Parameters:**
- `lane_name` (str): Name of the lane

**Returns:** Folder path (str)

#### `list_lanes()`

List all lanes and their collections.

**Returns:** Dictionary mapping lane names to lists of collections

---

### VectorStoreManager

Handles vector database operations.

#### `initialize_database(embedding_model="mxbai-embed-large")`

Initialize the vector database.

**Parameters:**
- `embedding_model` (str): Ollama embedding model name

**Returns:** None

#### `add_documents_to_collection(documents, collection_name)`

Add documents to a collection.

**Parameters:**
- `documents` (list): List of Document objects
- `collection_name` (str): Name of the collection

**Returns:** None

#### `search(query, lane_name=None, k=5)`

Search the vector database.

**Parameters:**
- `query` (str): Search query
- `lane_name` (str, list, optional): Lane(s) to search
- `k` (int): Number of results

**Returns:** List of Document objects

#### `get_stats()`

Get database statistics.

**Returns:** Dictionary with stats

#### `clear_database()`

Clear all data from the database.

**Returns:** None

---

### LLM

Language model interface using Ollama.

#### `__init__(model_name="llama3.2")`

Initialize the LLM.

**Parameters:**
- `model_name` (str): Name of the Ollama model

#### `generate(prompt)`

Generate text from a prompt.

**Parameters:**
- `prompt` (str): Input prompt

**Returns:** Generated text (str)

**Example:**
```python
context = "\n\n".join([doc.page_content for doc in results])
prompt = f"Context: {context}\n\nQuestion: Your question here?"
answer = rag.llm.generate(prompt)
```

---

## Usage Examples

### Example 1: Basic Setup

```python
from rag_manager import RAGManager

rag = RAGManager(persist_directory="./chroma_db")

rag.vector_db.run_initialize(
    lanes_config={"lane_1": "./data/source_1"},
    embedding_model="mxbai-embed-large"
)

rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")

results = rag.vector_db.run_query("search query", lane_name="lane_1", k=5)
```

### Example 2: Multiple Collections per Lane

```python
rag.vector_db.run_initialize(
    lanes_config={"lane_1": "./data/source_1"}
)

rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")
rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_2")

results = rag.vector_db.run_query("search query", lane_name="lane_1", k=5)
```

### Example 3: Query Multiple Lanes

```python
rag.vector_db.run_initialize(
    lanes_config={
        "lane_1": "./data/source_1",
        "lane_2": "./data/source_2"
    }
)

rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")
rag.vector_db.run_add_documents(lane_name="lane_2", collection_name="collection_2")

results = rag.vector_db.run_query("search query", lane_name=["lane_1", "lane_2"], k=5)
```

### Example 4: Using with LLM

```python
results = rag.vector_db.run_query("search query", lane_name="lane_1", k=5)

context = "\n\n".join([doc.page_content for doc in results])
prompt = f"Context: {context}\n\nQuestion: Your question here?"
answer = rag.llm.generate(prompt)
print(answer)
```

### Example 5: Adding Documents Incrementally

```python
rag.vector_db.run_initialize(
    lanes_config={"lane_1": "./data/source_1"}
)

rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")

rag.vector_db.run_add_documents(
    lane_name="lane_1",
    collection_name="collection_1",
    file_paths=["./data/new_file.csv"]
)
```

### Example 6: Starting Fresh

```python
rag.vector_db.clear_database()

rag.vector_db.run_initialize(
    lanes_config={"lane_1": "./data/source_1"}
)

rag.vector_db.run_add_documents(lane_name="lane_1", collection_name="collection_1")
```

---

## Best Practices

1. **Organize by Data Source**: Create separate lanes for different data sources
2. **Use Descriptive Names**: Name lanes and collections clearly
3. **Set Folder Paths**: Always set folder paths for lanes to enable automatic file loading
4. **Query Specific Lanes**: Query only relevant lanes to improve performance
5. **Clear Before Reinitializing**: Use `clear_database()` before reinitializing to avoid conflicts
6. **Check Stats**: Use `get_stats()` to monitor database size
7. **Chunk Size**: Adjust `chunk_size` based on your data (smaller for short documents, larger for long documents)

---

## Troubleshooting

### Error: "Lane does not exist"

**Cause**: Trying to add documents to a lane that hasn't been created.

**Solution**: Initialize the lane first using `run_initialize()` or create it manually with `rag.lanes.create_lane()`.

### Error: "No folder set for lane"

**Cause**: Calling `run_add_documents()` without `file_paths` when the lane has no folder mapping.

**Solution**: Set the folder path using `rag.lanes.set_lane_folder()` or provide `file_paths` explicitly.

### Error: "Database not initialized"

**Cause**: Trying to query before initializing the database.

**Solution**: Call `run_initialize()` first.

### No Results from Query

**Cause**: Querying a lane that has no documents or using the wrong lane name.

**Solution**: 
- Check that documents were added: `rag.vector_db.get_stats()`
- Verify lane names: `rag.vector_db.list_lanes()`
- Try querying all lanes: `run_query(query, lane_name=None)`

### Documents Not Loading

**Cause**: Files are not in CSV or JSON format, or folder path is incorrect.

**Solution**:
- Verify file formats (only CSV and JSON are supported)
- Check folder path: `rag.lanes.get_lane_folder("lane_name")`
- Provide explicit file paths: `run_add_documents(lane_name, collection_name, file_paths=[...])`

---

## License

MIT License
