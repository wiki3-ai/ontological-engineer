#!/usr/bin/env python3
"""
Ingest Docling JSON documents into Weaviate for RAG Q&A.

Uses:
  - langchain-docling DoclingLoader to chunk the Docling JSON files
  - Ollama nomic-embed-text for embeddings (same model as Weaviate's text2vec-ollama)
  - Weaviate v4 as vector store

Usage:
    python rag_ingest.py [--docs-dir /path/to/docling] [--collection DoclingPapers]
                         [--weaviate-host host.docker.internal]
                         [--ollama-host host.docker.internal]
                         [--embed-model nomic-embed-text:latest]
                         [--recreate]
"""

import argparse
import glob
import json
import os
import sys
import time

import weaviate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore


DEFAULT_DOCS_DIR = "/workspaces/pup/docling"
DEFAULT_COLLECTION = "DoclingPapers"
DEFAULT_WEAVIATE_HOST = "host.docker.internal"
DEFAULT_WEAVIATE_PORT = 8080
DEFAULT_WEAVIATE_GRPC_PORT = 50051
DEFAULT_OLLAMA_HOST = "host.docker.internal"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_LENGTH = 50  # skip chunks shorter than this (headers, glyphs, etc.)


def get_json_files(docs_dir: str) -> list[str]:
    """Find all Docling JSON files in the given directory."""
    pattern = os.path.join(docs_dir, "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No JSON files found in {docs_dir}")
        sys.exit(1)
    return files


def _extract_text_from_docling_json(filepath: str) -> tuple[str, str]:
    """Extract plain text from a Docling JSON file, preserving section structure.

    Returns (paper_title, full_text).
    """
    with open(filepath) as f:
        doc = json.load(f)

    title = doc.get("name", os.path.basename(filepath))
    texts = doc.get("texts", [])

    parts = []
    for item in texts:
        label = item.get("label", "")
        text = item.get("text", "").strip()
        if not text:
            continue
        if label == "section_header":
            level = item.get("level", 1)
            prefix = "#" * min(level + 1, 4)
            parts.append(f"\n{prefix} {text}\n")
        elif label == "page_header" or label == "page_footer":
            continue  # skip headers/footers
        else:
            parts.append(text)

    return title, "\n\n".join(parts)


def load_and_chunk(json_files: list[str]) -> list[Document]:
    """Load Docling JSON files and return LangChain Document chunks."""
    print(f"Loading {len(json_files)} Docling JSON files...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )

    all_docs: list[Document] = []
    skipped = 0
    for filepath in json_files:
        title, full_text = _extract_text_from_docling_json(filepath)
        if not full_text.strip():
            continue
        chunks = splitter.create_documents(
            texts=[full_text],
            metadatas=[{"paper_title": title, "source": filepath}],
        )
        for chunk in chunks:
            if len(chunk.page_content.strip()) < MIN_CHUNK_LENGTH:
                skipped += 1
                continue
            all_docs.append(chunk)

    print(f"  → {len(all_docs)} chunks produced from {len(json_files)} files ({skipped} short chunks skipped)")
    return all_docs


def connect_weaviate(host: str, port: int, grpc_port: int) -> weaviate.WeaviateClient:
    """Connect to Weaviate and verify readiness."""
    client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    if not client.is_ready():
        print(f"Weaviate at {host}:{port} is not ready")
        sys.exit(1)
    print(f"Connected to Weaviate at {host}:{port}")
    return client


def ingest(
    docs: list,
    client: weaviate.WeaviateClient,
    collection_name: str,
    embeddings: OllamaEmbeddings,
    recreate: bool = False,
) -> WeaviateVectorStore:
    """Ingest document chunks into Weaviate."""
    # Optionally delete existing collection
    if recreate and client.collections.exists(collection_name):
        print(f"Deleting existing collection '{collection_name}'...")
        client.collections.delete(collection_name)

    print(f"Ingesting {len(docs)} chunks into Weaviate collection '{collection_name}'...")
    start = time.time()

    # Batch ingest in groups to show progress
    batch_size = 50
    vs = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        if vs is None:
            vs = WeaviateVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                client=client,
                index_name=collection_name,
                text_key="text",
            )
        else:
            vs.add_documents(batch)
        done = min(i + batch_size, len(docs))
        elapsed = time.time() - start
        print(f"  → {done}/{len(docs)} chunks ingested ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"Ingestion complete in {elapsed:.1f}s")
    return vs


def main():
    parser = argparse.ArgumentParser(description="Ingest Docling papers into Weaviate for RAG")
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR, help="Directory with Docling JSON files")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Weaviate collection name")
    parser.add_argument("--weaviate-host", default=DEFAULT_WEAVIATE_HOST)
    parser.add_argument("--weaviate-port", type=int, default=DEFAULT_WEAVIATE_PORT)
    parser.add_argument("--weaviate-grpc-port", type=int, default=DEFAULT_WEAVIATE_GRPC_PORT)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--ollama-port", type=int, default=DEFAULT_OLLAMA_PORT)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the collection")
    args = parser.parse_args()

    # 1. Find JSON files
    json_files = get_json_files(args.docs_dir)
    print(f"Found {len(json_files)} Docling JSON files in {args.docs_dir}")

    # 2. Load and chunk via DoclingLoader
    docs = load_and_chunk(json_files)

    # 3. Set up embeddings (Ollama nomic-embed-text)
    ollama_url = f"http://{args.ollama_host}:{args.ollama_port}"
    embeddings = OllamaEmbeddings(model=args.embed_model, base_url=ollama_url)
    print(f"Using embeddings: {args.embed_model} via {ollama_url}")

    # 4. Connect to Weaviate
    client = connect_weaviate(args.weaviate_host, args.weaviate_port, args.weaviate_grpc_port)

    try:
        # 5. Ingest
        ingest(docs, client, args.collection, embeddings, recreate=args.recreate)
        print(f"\nDone! Collection '{args.collection}' is ready for queries.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
