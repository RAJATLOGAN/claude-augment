#!/usr/bin/env python3
"""
MCP Server — auto-detects the current repo from CWD,
auto-indexes it if not already indexed, then exposes search to Claude Code.
"""

import os
import chromadb
from fastmcp import FastMCP
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from code_indexer import (
    perform_indexing,
    get_collection_name_from_path,
    CHROMA_DB_PATH,
)

mcp = FastMCP("codebase-search")

# Per-collection cache
_embed_model = None
_collections: dict = {}


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = OllamaEmbedding(model_name="nomic-embed-text")
    return _embed_model


def get_repo_info():
    """Returns (repo_path, collection_name) based on CWD."""
    cwd = os.getcwd()
    collection_name = get_collection_name_from_path(cwd)
    return cwd, collection_name


def collection_exists(collection_name: str) -> bool:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return collection_name in [c.name for c in client.list_collections()]


def do_index(repo_path: str, collection_name: str) -> str:
    """Index a repo into ChromaDB. Returns a status string."""
    status = perform_indexing(
        repo_path=repo_path,
        collection_name=collection_name,
        embed_model=get_embed_model(),
        show_progress=False
    )

    # Invalidate cached collection
    _collections.pop(collection_name, None)
    return status


def get_collection(collection_name: str):
    if collection_name not in _collections:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collections[collection_name] = client.get_collection(collection_name)
    return _collections[collection_name]


@mcp.tool()
def search_codebase(query: str, top_k: int = 5, collection: str = "", repo_path: str = "") -> str:
    """
    Search an indexed codebase for code relevant to the query.
    Auto-indexes the current repo if it hasn't been indexed yet.
    Use this to find implementations, understand architecture, or locate specific logic.
    Pass 'collection' (e.g. 'malaria_detection') or 'repo_path' to search a specific indexed repo.
    Use list_indexed_repos to see available collection names.
    """
    if repo_path:
        collection_name = get_collection_name_from_path(repo_path)
    elif collection:
        collection_name = collection
    else:
        repo_path, collection_name = get_repo_info()

    # Auto-index if not indexed yet
    if not collection_exists(collection_name):
        if not repo_path:
            return f"Collection '{collection_name}' not found. Index it first with index_repo.py."
        status = do_index(repo_path, collection_name)
        if "Failed" in status or "No supported" in status:
            return status

    try:
        embed_model = get_embed_model()
        chroma_col = get_collection(collection_name)

        embedding = embed_model.get_text_embedding(query)
        results = chroma_col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return "No relevant code found for this query."

        output = [f"Collection: {collection_name}\n"]
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
            start=1,
        ):
            file_path = meta.get("file_path", meta.get("file_name", "unknown"))
            # With cosine space: distance = 1 - cosine_similarity, so similarity = 1 - distance
            relevance = round((1 - dist) * 100, 1)
            output.append(
                f"[{i}] {file_path} (relevance: {relevance}%)\n"
                f"{'-'*60}\n"
                f"{doc.strip()}\n"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search error: {e}"


@mcp.tool()
def reindex_repo() -> str:
    """
    Re-index the current repo from scratch.
    Use this after major code changes to refresh the search index.
    """
    repo_path, collection_name = get_repo_info()
    return do_index(repo_path, collection_name)


@mcp.tool()
def index_status() -> str:
    """
    Show indexing status for the current repo.
    Returns how many chunks are indexed and when it was last indexed.
    """
    repo_path, collection_name = get_repo_info()

    if not collection_exists(collection_name):
        return (
            f"Repo '{repo_path}' has NOT been indexed yet.\n"
            "It will be auto-indexed on your first search."
        )

    try:
        chroma_col = get_collection(collection_name)
        count = chroma_col.count()
        return (
            f"Repo: {repo_path}\n"
            f"Collection: {collection_name}\n"
            f"Indexed chunks: {count}\n"
            f"Index DB: {CHROMA_DB_PATH}"
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def list_indexed_repos() -> str:
    """List all repos that have been indexed so far."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        if not collections:
            return "No repos indexed yet."
        lines = [f"Indexed repos ({len(collections)}):"]
        for c in collections:
            count = client.get_collection(c.name).count()
            lines.append(f"  - {c.name}  ({count} chunks)")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
