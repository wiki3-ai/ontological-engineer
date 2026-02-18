#!/usr/bin/env python3
"""
Deep Research Agent over Docling papers stored in Weaviate.

A multi-step research agent that iteratively searches, retrieves, and
synthesizes information from the paper corpus to produce comprehensive
research reports.

Uses:
  - LangGraph create_agent (ReAct loop with tool calling)
  - Weaviate hybrid search (BM25 + vector) over ingested Docling papers
  - Ollama nomic-embed-text for query embeddings
  - LM Studio (OpenAI-compatible) for the reasoning LLM

Architecture (inspired by langchain-ai/open_deep_research):
  1. Scope  — the agent decomposes the question into sub-queries
  2. Research — iteratively calls tools to gather evidence
  3. Report — synthesizes findings into a structured answer with citations

Usage:
    python rag_research.py -q "Compare the approaches to code generation from ACL2"
    python rag_research.py -i
"""

import argparse
import json
import os
import sys
import textwrap
import urllib.request
import warnings
from typing import Optional

import weaviate
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = "DoclingPapers"
DEFAULT_WEAVIATE_HOST = "host.docker.internal"
DEFAULT_WEAVIATE_PORT = 8080
DEFAULT_WEAVIATE_GRPC_PORT = 50051
DEFAULT_OLLAMA_HOST = "host.docker.internal"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
DEFAULT_LM_STUDIO_HOST = "host.docker.internal"
DEFAULT_LM_STUDIO_PORT = 1234
DEFAULT_ALPHA = 0.25
DEFAULT_TOP_K = 8
DEFAULT_MAX_ITERATIONS = 15  # max ReAct tool-call loops

# ---------------------------------------------------------------------------
# System prompt — instructs the agent how to do deep research
# ---------------------------------------------------------------------------
RESEARCH_SYSTEM_PROMPT = """\
You are a Deep Research Agent with access to a knowledge base of academic \
papers about AI, formal methods, program synthesis, theorem proving, and \
related topics. Your job is to answer research questions thoroughly by \
searching the knowledge base, gathering evidence from multiple papers, and \
synthesizing a comprehensive, well-cited answer.

## Your Research Process

1. **Decompose** — Break complex questions into sub-questions.
2. **Search** — Use `search_papers` to find relevant passages. Try multiple \
   query phrasings (keywords, full phrases, related terms) for best coverage.
3. **Drill down** — Use `get_paper_chunks` to read more from promising papers.
4. **Cross-reference** — Use `list_papers` to discover papers you may have missed.
5. **Synthesize** — Once you have sufficient evidence, write your final answer.

## Guidelines

- Always search before answering. Never rely on prior knowledge alone.
- Conduct at least 2-3 searches with different query angles.
- Cite papers by name when reporting findings.
- If the knowledge base lacks information, say so explicitly.
- Prefer depth over breadth — read multiple chunks from key papers.
- Structure long answers with Markdown headings and bullet points.
- When comparing approaches, use a table or side-by-side format.

Today's date: {date}
"""

# ---------------------------------------------------------------------------
# Global state — set up during main(), used by tool functions
# ---------------------------------------------------------------------------
_weaviate_client: Optional[weaviate.WeaviateClient] = None
_vector_store: Optional[WeaviateVectorStore] = None
_embeddings: Optional[OllamaEmbeddings] = None
_collection_name: str = DEFAULT_COLLECTION
_alpha: float = DEFAULT_ALPHA
_top_k: int = DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# Tools — these are what the agent can call
# ---------------------------------------------------------------------------

@tool
def search_papers(query: str) -> str:
    """Search the knowledge base of academic papers using hybrid (keyword + semantic) search.

    Use this to find relevant passages about a topic. Returns the top matching
    chunks with their source paper titles. Try different query phrasings for
    best coverage — e.g. search for "APT program transformations" and also
    "automated refinement ACL2".

    Args:
        query: The search query — can be keywords, a phrase, or a question.
    """
    if _vector_store is None:
        return "Error: vector store not initialized"

    try:
        results = _vector_store.similarity_search(query, k=_top_k, alpha=_alpha)
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No results found for: {query}"

    output_parts = []
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("paper_title", "Unknown")
        text = doc.page_content.strip()
        # Truncate very long chunks for the agent's context window
        if len(text) > 800:
            text = text[:800] + "..."
        output_parts.append(f"[{i}] Paper: {title}\n{text}")

    return "\n\n---\n\n".join(output_parts)


@tool
def get_paper_chunks(paper_title: str, max_chunks: int = 10) -> str:
    """Retrieve all text chunks from a specific paper by title.

    Use this after search_papers finds an interesting paper and you want to
    read more of it. Returns up to max_chunks chunks from the paper.

    Args:
        paper_title: The exact paper title (as returned by search_papers).
        max_chunks: Maximum number of chunks to return (default: 10).
    """
    if _weaviate_client is None:
        return "Error: Weaviate client not initialized"

    try:
        col = _weaviate_client.collections.get(_collection_name)
        response = col.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("paper_title").equal(paper_title),
            limit=max_chunks,
            return_properties=["text", "paper_title"],
        )
    except Exception as e:
        return f"Error fetching paper: {e}"

    if not response.objects:
        # Try a partial match via BM25
        try:
            response = col.query.bm25(
                query=paper_title,
                limit=max_chunks,
                return_properties=["text", "paper_title"],
            )
        except Exception as e:
            return f"Error in fallback search: {e}"

    if not response.objects:
        return f"No chunks found for paper: {paper_title}"

    output_parts = []
    actual_title = response.objects[0].properties.get("paper_title", paper_title)
    for i, obj in enumerate(response.objects, 1):
        text = obj.properties.get("text", "").strip()
        if len(text) > 600:
            text = text[:600] + "..."
        output_parts.append(f"[Chunk {i}] {text}")

    header = f"Paper: {actual_title} ({len(response.objects)} chunks)"
    return header + "\n\n" + "\n\n---\n\n".join(output_parts)


@tool
def list_papers() -> str:
    """List all papers available in the knowledge base.

    Use this to discover what papers are available, or to find the exact
    title of a paper you want to drill into with get_paper_chunks.
    Returns paper titles sorted alphabetically.
    """
    if _weaviate_client is None:
        return "Error: Weaviate client not initialized"

    try:
        col = _weaviate_client.collections.get(_collection_name)
        titles: set[str] = set()
        cursor = None
        while True:
            if cursor is None:
                resp = col.query.fetch_objects(
                    limit=100, return_properties=["paper_title"]
                )
            else:
                resp = col.query.fetch_objects(
                    limit=100, return_properties=["paper_title"], after=cursor
                )
            if not resp.objects:
                break
            for obj in resp.objects:
                titles.add(obj.properties.get("paper_title", ""))
            cursor = resp.objects[-1].uuid

        sorted_titles = sorted(titles - {""})
        output = f"Knowledge base contains {len(sorted_titles)} papers:\n\n"
        for i, t in enumerate(sorted_titles, 1):
            output += f"  {i}. {t}\n"
        return output
    except Exception as e:
        return f"Error listing papers: {e}"


# ---------------------------------------------------------------------------
# LM Studio model detection
# ---------------------------------------------------------------------------

def detect_lm_studio_model(host: str, port: int) -> str:
    """Auto-detect the best model loaded in LM Studio.

    Prefers models with 'qwen3' in the name (best tool-calling support),
    then the largest model available.  Skips embedding models.
    """
    url = f"http://{host}:{port}/v1/models"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        models = data.get("data", [])

        # Filter out embedding models
        EMBED_KEYWORDS = {"embed", "nomic", "modernbert"}
        candidates = [
            m["id"]
            for m in models
            if not any(kw in m["id"].lower() for kw in EMBED_KEYWORDS)
        ]
        if not candidates:
            return models[0]["id"] if models else "local-model"

        # Prefer qwen3-next (reasoning), then qwen3-coder, then largest name
        for preference in ["qwen3-next", "qwen3-coder", "qwen/qwen3"]:
            for c in candidates:
                if preference in c.lower():
                    return c

        # Fall back to first non-tiny candidate (skip 0.5b, 1b)
        TINY = {"0.5b", "1b", "nano"}
        for c in candidates:
            if not any(t in c.lower() for t in TINY):
                return c

        return candidates[0]
    except Exception:
        pass
    return "local-model"


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_research_agent(
    llm: ChatOpenAI,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """Build the deep research agent with all tools."""
    import datetime

    tools = [search_papers, get_paper_chunks, list_papers]

    system_prompt = RESEARCH_SYSTEM_PROMPT.format(
        date=datetime.date.today().isoformat(),
    )

    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )
    return agent


# ---------------------------------------------------------------------------
# Streaming output
# ---------------------------------------------------------------------------

def run_research(agent, question: str, verbose: bool = False) -> str:
    """Run the research agent on a question and return the final answer."""
    print(f"\n{'='*60}")
    print(f"Research question: {question}")
    print(f"{'='*60}\n")

    step_count = 0
    final_answer = ""

    for event in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        last_msg = event["messages"][-1]
        step_count += 1

        if last_msg.type == "ai":
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tc in last_msg.tool_calls:
                    args_str = json.dumps(tc["args"], ensure_ascii=False)
                    if len(args_str) > 120:
                        args_str = args_str[:120] + "..."
                    print(f"  🔧 {tc['name']}({args_str})")
            elif last_msg.content:
                final_answer = last_msg.content
                if verbose:
                    print(f"\n  📝 Agent thinking... ({len(last_msg.content)} chars)")

        elif last_msg.type == "tool":
            content_preview = last_msg.content[:150] if last_msg.content else ""
            if verbose:
                print(f"  📄 Tool result: {content_preview}...")

    print(f"\n{'='*60}")
    print(f"Research complete ({step_count} steps)")
    print(f"{'='*60}\n")

    return final_answer


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_loop(agent):
    """Run an interactive deep research session."""
    print("\n=== Deep Research Agent ===")
    print("Ask complex research questions about the paper corpus.")
    print("The agent will search, gather evidence, and synthesize answers.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Research Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = run_research(agent, question, verbose=True)
        print(f"\n{answer}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _weaviate_client, _vector_store, _embeddings
    global _collection_name, _alpha, _top_k

    parser = argparse.ArgumentParser(
        description="Deep Research Agent over Docling papers in Weaviate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag_research.py -q "Compare approaches to code generation from ACL2"
              python rag_research.py -q "What formal verification tools exist for software?" -v
              python rag_research.py -i
        """),
    )
    parser.add_argument("--question", "-q", type=str, help="Research question")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool results")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--weaviate-host", default=DEFAULT_WEAVIATE_HOST)
    parser.add_argument("--weaviate-port", type=int, default=DEFAULT_WEAVIATE_PORT)
    parser.add_argument("--weaviate-grpc-port", type=int, default=DEFAULT_WEAVIATE_GRPC_PORT)
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--ollama-port", type=int, default=DEFAULT_OLLAMA_PORT)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--lm-studio-host", default=DEFAULT_LM_STUDIO_HOST)
    parser.add_argument("--lm-studio-port", type=int, default=DEFAULT_LM_STUDIO_PORT)
    parser.add_argument("--lm-studio-model", default=None, help="LM Studio model (auto-detected if omitted)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Hybrid search alpha: 0=BM25, 1=vector (default: 0.25)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Chunks per search (default: 8)")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                        help="Max agent tool-call iterations (default: 15)")
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.print_help()
        print("\nProvide --question or --interactive")
        sys.exit(1)

    # Store config in globals for tools
    _collection_name = args.collection
    _alpha = args.alpha
    _top_k = args.top_k

    # 1. Embeddings (Ollama)
    ollama_url = f"http://{args.ollama_host}:{args.ollama_port}"
    _embeddings = OllamaEmbeddings(model=args.embed_model, base_url=ollama_url)

    # 2. LLM (LM Studio)
    lm_studio_url = f"http://{args.lm_studio_host}:{args.lm_studio_port}/v1"
    model_name = (
        args.lm_studio_model
        or os.environ.get("LM_STUDIO_MODEL")
        or detect_lm_studio_model(args.lm_studio_host, args.lm_studio_port)
    )
    llm = ChatOpenAI(
        base_url=lm_studio_url,
        api_key="lm-studio",
        model=model_name,
        temperature=0.2,
    )
    print(f"LLM: {model_name} via {lm_studio_url}")

    # 3. Connect to Weaviate
    _weaviate_client = weaviate.connect_to_local(
        host=args.weaviate_host,
        port=args.weaviate_port,
        grpc_port=args.weaviate_grpc_port,
    )
    if not _weaviate_client.is_ready():
        print("Weaviate is not ready")
        sys.exit(1)
    print(f"Weaviate connected at {args.weaviate_host}:{args.weaviate_port}")

    # 4. Vector store (for search_papers tool)
    _vector_store = WeaviateVectorStore(
        client=_weaviate_client,
        index_name=_collection_name,
        text_key="text",
        embedding=_embeddings,
        attributes=["paper_title"],
    )

    try:
        # 5. Build agent
        agent = build_research_agent(llm, args.max_iterations)

        # 6. Run
        if args.question:
            answer = run_research(agent, args.question, verbose=args.verbose)
            print(answer)
        else:
            interactive_loop(agent)
    finally:
        _weaviate_client.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ResourceWarning, message="unclosed.*socket")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main()
