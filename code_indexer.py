#!/usr/bin/env python3
"""
Shared indexing logic for the claude-augment project.
"""

import os
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

CODE_EXTENSIONS = [
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".java", ".rs", ".cpp", ".c", ".h",
    ".rb", ".php", ".swift", ".kt", ".cs",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json",
    ".sh", ".sql", ".ipynb", ".env.example",
]

EXCLUDE_DIRS = [
    "**/node_modules/**", "**/.venv/**", "**/venv/**",
    "**/dist/**", "**/build/**", "**/target/**",
    "**/.next/**", "**/coverage/**", "**/.mypy_cache/**",
    "**/__pycache__/**", "**/.git/**",
]

EXCLUDE_FILES = [
    "**/package-lock.json", "**/yarn.lock", "**/pnpm-lock.yaml",
]

MAX_CHARS = 30000

def get_collection_name_from_path(repo_path: str) -> str:
    """Generates a collection name from a repo path."""
    return os.path.basename(os.path.abspath(repo_path)).replace(" ", "_").replace("-", "_").lower()

def perform_indexing(repo_path: str, collection_name: str, embed_model, show_progress: bool = False) -> str:
    """
    Core logic to index a repository into ChromaDB.
    Returns a status string.
    """
    Settings.embed_model = embed_model

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(collection_name)
        if show_progress:
            print(f"Deleted existing collection '{collection_name}' for fresh index.")
    except Exception:
        pass

    chroma_collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if show_progress:
        print("Loading files...")
    try:
        documents = SimpleDirectoryReader(
            input_dir=repo_path,
            recursive=True,
            required_exts=CODE_EXTENSIONS,
            exclude_hidden=True,
            exclude=EXCLUDE_DIRS + EXCLUDE_FILES,
        ).load_data()
    except Exception as e:
        return f"Failed to load files: {e}"

    if not documents:
        return "No supported files found in this repo."

    original_count = len(documents)
    documents = [d for d in documents if len(d.text) <= MAX_CHARS]
    skipped_count = original_count - len(documents)
    if skipped_count > 0 and show_progress:
        print(f"Skipped {skipped_count} file(s) exceeding {MAX_CHARS} char limit.")

    if not documents:
        return "All files exceeded the size limit. Nothing to index."

    if show_progress:
        print(f"Found {len(documents)} files. Generating embeddings...\n")

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=show_progress,
    )

    count = chroma_collection.count()
    status = f"Indexed {len(documents)} files ({count} chunks) from '{repo_path}' into collection '{collection_name}'."
    
    if show_progress:
        print(f"\nDone! Indexed {len(documents)} files ({count} chunks) into '{collection_name}'.")
        print(f"Index saved at: {CHROMA_DB_PATH}\n")

    return status