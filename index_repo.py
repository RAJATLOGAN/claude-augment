#!/usr/bin/env python3
"""
Index a code repository into ChromaDB using Ollama embeddings.
Usage: python index_repo.py /path/to/your/repo
"""

import sys
import os
import argparse
from llama_index.embeddings.ollama import OllamaEmbedding
from code_indexer import (
    perform_indexing,
    get_collection_name_from_path,
    CHROMA_DB_PATH,
)


def index_repo(repo_path: str, collection_name: str):
    print(f"\n{'='*50}")
    print(f"Indexing: {repo_path}")
    print(f"Collection: {collection_name}")
    print(f"DB Path: {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    status = perform_indexing(
        repo_path=repo_path,
        collection_name=collection_name,
        embed_model=embed_model,
        show_progress=True
    )
    if "Failed" in status or "No supported" in status or "exceeded the size limit" in status:
        print(f"Error: {status}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a code repository")
    parser.add_argument("repo_path", help="Path to the repository to index")
    parser.add_argument(
        "--collection",
        default=None,
        help="ChromaDB collection name (default: generated from repo path)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.repo_path):
        print(f"Error: '{args.repo_path}' is not a valid directory.")
        sys.exit(1)
    
    collection_name = args.collection or get_collection_name_from_path(args.repo_path)
    
    index_repo(args.repo_path, collection_name)
