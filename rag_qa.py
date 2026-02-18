#!/usr/bin/env python3
"""
RAG Q&A over Docling papers stored in Weaviate.

Uses:
  - Weaviate as vector store (with Ollama nomic-embed-text embeddings)
  - LM Studio (OpenAI-compatible API) as the Q&A LLM
  - LangChain LCEL chain

Usage:
    python rag_qa.py [--question "What is APT?"]
    python rag_qa.py --interactive

Environment:
    LM_STUDIO_MODEL  - model name loaded in LM Studio (auto-detected if not set)
"""

import argparse
import json
import os
import sys
import urllib.request
import warnings

import weaviate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore


DEFAULT_COLLECTION = "DoclingPapers"
DEFAULT_WEAVIATE_HOST = "host.docker.internal"
DEFAULT_WEAVIATE_PORT = 8080
DEFAULT_WEAVIATE_GRPC_PORT = 50051
DEFAULT_OLLAMA_HOST = "host.docker.internal"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_LM_STUDIO_HOST = "host.docker.internal"
DEFAULT_LM_STUDIO_PORT = 1234
DEFAULT_TOP_K = 6
DEFAULT_ALPHA = 0.25  # hybrid search weighting: 0=pure BM25, 1=pure vector


RAG_PROMPT_TEMPLATE = """\
You are a helpful research assistant answering questions about academic papers.
Use ONLY the provided context to answer. If the context doesn't contain enough
information, say so — do not make things up.

When possible, mention which paper(s) the information comes from.

Context:
{context}

Question: {question}

Answer:"""


def detect_lm_studio_model(host: str, port: int) -> str:
    """Auto-detect the first loaded model in LM Studio."""
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        models = data.get("data", [])
        if models:
            model_id = models[0]["id"]
            print(f"Auto-detected LM Studio model: {model_id}")
            return model_id
    except Exception as e:
        print(f"Warning: Could not auto-detect LM Studio model: {e}")
    return "local-model"


def format_docs(docs: list) -> str:
    """Format retrieved documents into a context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(
    client: weaviate.WeaviateClient,
    collection_name: str,
    embeddings: OllamaEmbeddings,
    llm: ChatOpenAI,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = DEFAULT_ALPHA,
):
    """Build a LangChain LCEL RAG chain over the Weaviate vector store."""
    vs = WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="text",
        embedding=embeddings,
        attributes=["paper_title"],
    )

    retriever = vs.as_retriever(search_kwargs={"k": top_k, "alpha": alpha})

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def format_sources(source_docs: list) -> str:
    """Format source documents for display."""
    seen = set()
    lines = []
    for doc in source_docs:
        title = doc.metadata.get("paper_title", "Unknown")
        if title not in seen:
            seen.add(title)
            lines.append(f"  - {title}")
    return "\n".join(lines)


def ask(chain, retriever, question: str, show_sources: bool = True) -> str:
    """Ask a question and return the answer."""
    # Run retrieval separately so we can show sources
    source_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    if show_sources and source_docs:
        sources = format_sources(source_docs)
        return f"{answer}\n\nSources:\n{sources}"
    return answer


def interactive_loop(chain, retriever):
    """Run an interactive Q&A session."""
    print("\n=== RAG Q&A over Docling Papers ===")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            question = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        answer = ask(chain, retriever, question)
        print(f"A: {answer}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A over Docling papers in Weaviate")
    parser.add_argument("--question", "-q", type=str, help="Single question to ask")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--weaviate-host", default=DEFAULT_WEAVIATE_HOST)
    parser.add_argument("--weaviate-port", type=int, default=DEFAULT_WEAVIATE_PORT)
    parser.add_argument("--weaviate-grpc-port", type=int, default=DEFAULT_WEAVIATE_GRPC_PORT)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--ollama-port", type=int, default=DEFAULT_OLLAMA_PORT)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--lm-studio-host", default=DEFAULT_LM_STUDIO_HOST)
    parser.add_argument("--lm-studio-port", type=int, default=DEFAULT_LM_STUDIO_PORT)
    parser.add_argument("--lm-studio-model", default=None, help="LM Studio model name (auto-detected if not set)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Hybrid search alpha: 0=pure BM25, 1=pure vector (default: 0.25)")
    parser.add_argument("--no-sources", action="store_true", help="Don't show source documents")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        print("\nProvide --question or --interactive")
        sys.exit(1)

    # 1. Embeddings (Ollama)
    ollama_url = f"http://{args.ollama_host}:{args.ollama_port}"
    embeddings = OllamaEmbeddings(model=args.embed_model, base_url=ollama_url)

    # 2. LLM (LM Studio — OpenAI-compatible)
    lm_studio_url = f"http://{args.lm_studio_host}:{args.lm_studio_port}/v1"
    model_name = args.lm_studio_model or os.environ.get("LM_STUDIO_MODEL") or detect_lm_studio_model(args.lm_studio_host, args.lm_studio_port)

    llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="lm-studio",  # LM Studio doesn't need a real key
        model=model_name,
        temperature=0.1,
    )
    print(f"LLM: {model_name} via {lm_studio_url}")

    # 3. Connect to Weaviate
    client = weaviate.connect_to_local(
        host=args.weaviate_host,
        port=args.weaviate_port,
        grpc_port=args.weaviate_grpc_port,
    )
    if not client.is_ready():
        print("Weaviate is not ready")
        sys.exit(1)
    print(f"Weaviate connected at {args.weaviate_host}:{args.weaviate_port}")

    try:
        # 4. Build chain
        chain, retriever = build_qa_chain(client, args.collection, embeddings, llm, args.top_k, args.alpha)

        # 5. Run
        if args.question:
            answer = ask(chain, retriever, args.question, show_sources=not args.no_sources)
            print(f"\n{answer}")
        else:
            interactive_loop(chain, retriever)
    finally:
        client.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")
    main()
