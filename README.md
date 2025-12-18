# wiki3 Knowledge Graph Extraction Notebook

Extract structured knowledge from Wikipedia using LLMs (via LangChain JS), generate embeddings, and prepare data for browser-based semantic search with DuckDB-Wasm.

## Architecture

```
Wikipedia Article → LangChain WikipediaQueryRun → Text Splitter (1024 chars)
  ↓
LLM-based Extraction (GPT-4-mini or similar)
  ├─ Entities: {id, label, type, description}
  └─ Relations: {source_id, target_id, type, description}
  ↓
Embedding Generation (mock or real model)
  ↓
DuckDB-Wasm Format (Parquet/Arrow)
  ↓
Browser Query Engine
  ├─ Graph traversal (SQL joins on triples)
  └─ Semantic similarity (VSS/HNSW index)
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (set `OPENAI_API_KEY`)

### Run

```bash
export OPENAI_API_KEY="sk-..."
docker-compose up --build
# Open http://localhost:8888
```

### Run Notebook

1. Open `notebooks/wiki3-kg-extraction.ipynb`
2. Select **Deno** kernel
3. Run cells in order:
   - Setup (load LangChain JS)
   - Wikipedia Loader (fetch article)
   - Configure LLM (set up schema)
   - Extract KG (run extraction)
   - Inspect Results (view entities/relations)
   - Generate Embeddings (vector generation)
   - Export (format for browser)

## Kernel Support

| Kernel | Status | Notes |
|--------|--------|-------|
| **Deno** | ✅ Recommended | Full TypeScript, npm packages, Web APIs |
| **TSLab** | ✅ Works | Use `rnowotniak/jupyter-tslab` base image |
| **IJavaScript** | ✅ Works | Basic Node.js (no TypeScript) |
| **JupyterLite** | ⏳ Limited | Python only currently; JS/TS coming soon |

## Customization

### Change Article
```typescript
const articleTitle = "Albert Einstein";  // ← Change this
```

### Use Different LLM

**Anthropic Claude:**
```typescript
import { ChatAnthropic } from "@langchain/anthropic";
const llm = new ChatAnthropic({
  modelName: "claude-3-sonnet",
  apiKey: Deno.env.get("ANTHROPIC_API_KEY"),
});
```

**Local Ollama:**
```typescript
import { ChatOllama } from "@langchain/ollama";
const llm = new ChatOllama({
  model: "mistral",
  baseUrl: "http://localhost:11434",
});
```

## Browser Integration (Next Step)

After extraction, export as Parquet and load in DuckDB-Wasm:

```typescript
// Browser code
import * as duckdb from '@duckdb/wasm';

const db = new duckdb.Database();
const conn = db.connect();

// Load Parquet
const resp = await fetch('entities.parquet');
const buffer = await resp.arrayBuffer();
await conn.insertArrowFromIPCStream(new Uint8Array(buffer), { name: 'entities' });

// Similarity query
const results = await conn.query(`
  SELECT id, label, distance(embedding, $vec) as sim
  FROM entities ORDER BY sim LIMIT 10
`, { vec: queryEmbedding });
```

## References

- [LangChain JS](https://js.langchain.com/)
- [DuckDB-Wasm](https://duckdb.org/docs/api/wasm)
- [Deno Jupyter](https://docs.deno.com/runtime/reference/cli/jupyter/)
- [neo4j-labs/llm-graph-builder](https://github.com/neo4j-labs/llm-graph-builder)

## License

MIT
