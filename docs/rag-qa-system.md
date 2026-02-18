# RAG Q&A System for Docling Papers

A retrieval-augmented generation (RAG) system that lets you ask natural-language
questions over a corpus of academic papers. Papers are stored as Docling JSON
files, chunked and embedded into Weaviate, and answered by a local LLM served
via LM Studio.

## Architecture

```
┌────────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Docling JSON      │     │    Weaviate       │     │    LM Studio      │
│  /workspaces/pup/  │────▷│  Vector Store     │     │  (OpenAI API)     │
│  docling/*.json    │     │  :8080 / :50051   │     │  :1234/v1         │
└────────────────────┘     └────────┬──────────┘     └────────┬──────────┘
        ▲ ingest                    │ retrieve                │ generate
        │                           ▼                         ▼
   rag_ingest.py              rag_qa.py ──────────────────────┘
                                    │
                               ┌────┴────┐
                               │  Ollama  │
                               │ nomic-   │
                               │ embed-   │
                               │ text     │
                               │ :11434   │
                               └─────────┘
```

### Components

| Component | Role | Default endpoint |
|-----------|------|------------------|
| **Weaviate** | Vector + keyword store (HNSW index, BM25) | `host.docker.internal:8080` (HTTP), `:50051` (gRPC) |
| **Ollama** | Embedding model (`nomic-embed-text`, 768-dim) | `host.docker.internal:11434` |
| **LM Studio** | Q&A LLM (OpenAI-compatible API) | `host.docker.internal:1234/v1` |
| **rag_ingest.py** | Parse Docling JSON → chunk → embed → store in Weaviate | — |
| **rag_qa.py** | Query Weaviate → build prompt → call LLM → print answer | — |

### Data flow

1. **Ingestion** — `rag_ingest.py` reads each Docling JSON file, extracts text
   (preserving section headers as Markdown headings, skipping page
   headers/footers), splits into ~1000-character chunks with 200-char overlap,
   embeds each chunk via Ollama `nomic-embed-text`, and stores the chunk text +
   vector + metadata in the Weaviate collection `DoclingPapers`.

2. **Query** — `rag_qa.py` takes a question, runs Weaviate **hybrid search**
   (combined BM25 keyword + vector similarity), retrieves the top-k most
   relevant chunks, inserts them as context into a prompt, and sends the prompt
   to the LM Studio LLM for answer generation.

---

## Prerequisites

### Services (must be running on the host)

1. **Weaviate** — run via Docker:
   ```bash
   docker volume create weaviate_data
   docker build -f Dockerfile-weaviate -t weaviate-ollama .
   docker run -d --name weaviate \
     -p 8080:8080 -p 50051:50051 \
     -v weaviate_data:/var/lib/weaviate \
     weaviate-ollama
   ```

2. **Ollama** — install from [ollama.com](https://ollama.com), then pull the
   embedding model:
   ```bash
   ollama pull nomic-embed-text:latest
   ```

3. **LM Studio** — install from [lmstudio.ai](https://lmstudio.ai), load a
   model (e.g. `qwen/qwen3-coder-next`), and start the local server on port
   1234.

### Python packages

```bash
pip install -r requirements.txt
```

Key dependencies: `weaviate-client>=4.0`, `langchain-weaviate`,
`langchain-ollama`, `langchain-openai`, `langchain-text-splitters`,
`langchain-docling`.

### Docling JSON files

The corpus lives in `/workspaces/pup/docling/*.json`. These are
[DoclingDocument](https://ds4sd.github.io/docling/) v1.9.0 files produced by
running Docling's PDF-to-JSON pipeline over academic papers. Each file contains
a `texts` array with labeled elements (`section_header`, `text`,
`page_header`, `page_footer`, etc.) plus provenance and layout metadata.

---

## Step 1 — Ingest papers into Weaviate

```bash
python rag_ingest.py
```

This reads all `*.json` files from the docling directory, chunks them, embeds
them, and loads them into the `DoclingPapers` collection. With the default 138
papers, this produces ~14,000 chunks and takes roughly 7 minutes.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--docs-dir` | `/workspaces/pup/docling` | Directory containing Docling JSON files |
| `--collection` | `DoclingPapers` | Weaviate collection name |
| `--recreate` | off | Delete and recreate collection (full re-ingest) |
| `--weaviate-host` | `host.docker.internal` | Weaviate HTTP host |
| `--weaviate-port` | `8080` | Weaviate HTTP port |
| `--weaviate-grpc-port` | `50051` | Weaviate gRPC port |
| `--ollama-host` | `host.docker.internal` | Ollama host |
| `--ollama-port` | `11434` | Ollama port |
| `--embed-model` | `nomic-embed-text:latest` | Ollama embedding model |

### Re-ingesting

To rebuild from scratch (e.g. after adding papers or changing chunking params):

```bash
python rag_ingest.py --recreate
```

### Chunk details

- **Splitter**: `RecursiveCharacterTextSplitter` with chunk size 1000 chars,
  200-char overlap, and separators tuned for Markdown headings
  (`\n## `, `\n### `, `\n#### `, `\n\n`, `\n`, `. `, ` `).
- **Minimum length filter**: Chunks shorter than 50 characters are discarded.
  This prevents noise from section-header-only fragments, stray glyphs, and
  OCR artifacts that would otherwise pollute vector search results.
- **Metadata**: Each chunk carries `paper_title` (from the Docling `name`
  field) and `source` (file path).

---

## Step 2 — Ask questions

### Single question

```bash
python rag_qa.py -q "What is APT in ACL2?"
```

### Specify the LLM model

```bash
python rag_qa.py -q "What is APT in ACL2?" --lm-studio-model "qwen/qwen3-coder-next"
```

If `--lm-studio-model` is not provided, the script checks the
`LM_STUDIO_MODEL` environment variable, then auto-detects the first loaded
model in LM Studio.

### Interactive mode

```bash
python rag_qa.py -i --lm-studio-model "qwen/qwen3-coder-next"
```

Type questions at the `Q:` prompt; type `quit` to exit.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-q`, `--question` | — | Single question to ask |
| `-i`, `--interactive` | off | Interactive Q&A loop |
| `--lm-studio-model` | auto-detect | LM Studio model name |
| `--top-k` | `6` | Number of chunks to retrieve |
| `--alpha` | `0.25` | Hybrid search weighting (see below) |
| `--no-sources` | off | Suppress source paper list in output |
| `--collection` | `DoclingPapers` | Weaviate collection name |
| `--weaviate-host` | `host.docker.internal` | Weaviate HTTP host |
| `--weaviate-port` | `8080` | Weaviate HTTP port |
| `--weaviate-grpc-port` | `50051` | Weaviate gRPC port |
| `--ollama-host` | `host.docker.internal` | Ollama host |
| `--ollama-port` | `11434` | Ollama port |
| `--embed-model` | `nomic-embed-text:latest` | Ollama embedding model |
| `--lm-studio-host` | `host.docker.internal` | LM Studio host |
| `--lm-studio-port` | `1234` | LM Studio port |

---

## Hybrid Search and the `--alpha` Parameter

Weaviate's hybrid search combines two retrieval strategies:

- **BM25** (keyword/term frequency) — excellent for exact terms, acronyms,
  proper nouns
- **Vector similarity** (cosine distance on embeddings) — excellent for
  semantic/conceptual matching

The `alpha` parameter controls the blend:

| Alpha | Meaning |
|-------|---------|
| `0.0` | Pure BM25 (keyword only) |
| `0.25` | **Default** — 75% BM25, 25% vector |
| `0.5` | Equal mix |
| `0.75` | Weaviate's own default — 75% vector, 25% BM25 |
| `1.0` | Pure vector similarity |

### Why alpha=0.25?

Our corpus contains academic papers with technical terms, acronyms (e.g.
"APT", "ACL2", "HNSW"), and proper nouns that benefit heavily from exact
keyword matching. Additionally, short/noisy chunks (OCR artifacts, stray
glyphs) can produce generic embedding vectors that score spuriously well in
pure vector search. A BM25-heavy alpha ensures that queries containing specific
terms find the right documents, while the 25% vector component still provides
semantic broadening for conceptual queries.

**Empirical results on our corpus** (query: "What is APT in ACL2?"):

| Alpha | Top results |
|-------|-------------|
| 0.75 | Garbage — glyphs, unrelated papers |
| 0.50 | Mixed — APT papers start appearing |
| 0.25 | All top-5 results are APT/ACL2 papers ✓ |
| 0.00 | All correct, but no semantic capability |

You can override per-query:
```bash
python rag_qa.py -q "general concepts of program synthesis" --alpha 0.5
```

---

## Troubleshooting

### No results or wrong results

1. **Check Weaviate is running**: `curl http://host.docker.internal:8080/v1/.well-known/ready`
2. **Check Ollama is running**: `curl http://host.docker.internal:11434/api/tags`
3. **Verify collection exists**: Run `python -c "import weaviate; c = weaviate.connect_to_local(host='host.docker.internal'); print(c.collections.list_all()); c.close()"`
4. **Try lower alpha**: If results seem semantically off, try `--alpha 0.1` or `--alpha 0.0` for pure keyword search
5. **Increase top-k**: Try `--top-k 10` or `--top-k 15` to retrieve more context

### Ingestion is slow

Embedding 14,000 chunks via Ollama takes ~7 minutes. This is a one-time cost.
If you need to re-ingest, use `--recreate` to start fresh rather than adding
duplicates.

### LM Studio model auto-detection picks the wrong model

Specify the model explicitly:
```bash
python rag_qa.py -q "..." --lm-studio-model "qwen/qwen3-coder-next"
```

Or set the environment variable:
```bash
export LM_STUDIO_MODEL="qwen/qwen3-coder-next"
```

### "Weaviate is not ready"

Make sure the Weaviate Docker container is running and healthy:
```bash
docker ps | grep weaviate
docker logs weaviate
```

---

## File Reference

| File | Purpose |
|------|---------|
| `rag_ingest.py` | Ingestion pipeline: Docling JSON → chunks → Weaviate |
| `rag_qa.py` | Query pipeline: question → hybrid retrieval → LLM answer |
| `Dockerfile-weaviate` | Weaviate Docker image with Ollama modules enabled |
| `requirements.txt` | Python dependencies |

---

## Adding New Papers

1. Run the paper through [Docling](https://ds4sd.github.io/docling/) to produce
   a JSON file.
2. Place the JSON file in `/workspaces/pup/docling/`.
3. Re-run ingestion:
   ```bash
   python rag_ingest.py --recreate
   ```
4. Query as usual.
