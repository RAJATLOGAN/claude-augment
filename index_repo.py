#!/usr/bin/env python3
"""
Index a code repository into ChromaDB using Ollama embeddings.
Usage: python index_repo.py /path/to/your/repo
"""

import sys
import os
import argparse
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# Supported code/doc file extensions
CODE_EXTENSIONS = [
    ".py", ".ts", ".tsx", ".js", ".jsx",
    ".go", ".java", ".rs", ".cpp", ".c", ".h",
    ".rb", ".php", ".swift", ".kt", ".cs",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json",
    ".sh", ".env.example", ".sql", ".ipynb",
]

CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def index_repo(repo_path: str, collection_name: str = "codebase"):
    print(f"\n{'='*50}")
    print(f"Indexing: {repo_path}")
    print(f"Collection: {collection_name}")
    print(f"DB Path: {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    # Configure Ollama
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

    # Set up ChromaDB (persistent on disk)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection to re-index fresh
    try:
        chroma_client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}' for fresh index.")
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load files from repo
    print("Loading files...")
    try:
        documents = SimpleDirectoryReader(
            input_dir=repo_path,
            recursive=True,
            required_exts=CODE_EXTENSIONS,
            exclude_hidden=True,
        ).load_data()
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    if not documents:
        print("No supported files found in the repository.")
        sys.exit(1)

    # Skip documents that exceed the embedding model's context limit (~30k chars ≈ 8k tokens)
    MAX_CHARS = 30000
    filtered = [d for d in documents if len(d.text) <= MAX_CHARS]
    skipped = len(documents) - len(filtered)
    if skipped:
        print(f"Skipped {skipped} file(s) exceeding {MAX_CHARS} char limit.")
    documents = filtered

    if not documents:
        print("All files exceeded the size limit. Nothing to index.")
        sys.exit(1)

    print(f"Found {len(documents)} files. Generating embeddings...\n")

    # Index documents
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nDone! {len(documents)} files indexed into '{collection_name}'.")
    print(f"Index saved at: {CHROMA_DB_PATH}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a code repository")
    parser.add_argument("repo_path", help="Path to the repository to index")
    parser.add_argument(
        "--collection",
        default="codebase",
        help="ChromaDB collection name (default: codebase)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        print(f"Error: '{args.repo_path}' is not a valid directory.")
        sys.exit(1)

    index_repo(args.repo_path, args.collection)
