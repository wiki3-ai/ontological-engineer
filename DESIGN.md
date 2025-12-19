# Wiki3 Knowledge Graph Pipeline Design

## Overview

The `wiki_to_kg_pipeline.ipynb` notebook implements a multi-stage pipeline for converting Wikipedia articles into RDF Knowledge Graphs. The pipeline uses intermediate Jupyter notebook files for each transformation stage, enabling human inspection, correction, and incremental reprocessing.

**Repository**: https://github.com/wiki3-ai/ontological-engineer

## Design Goals

1. **Transparency**: Each stage produces human-readable output that can be inspected and edited
2. **Provenance**: Full traceability from source content to final RDF triples
3. **Incremental Processing**: Only regenerate content when source material changes
4. **Interruptibility**: Pipeline can be stopped and resumed without losing progress
5. **Editability**: Humans (or other agents) can modify intermediate outputs
6. **Vocabulary Compliance**: Use real schema.org vocabulary via embedding-based lookup tools

## Architecture

```
Wikipedia Article
       │
       ▼
┌─────────────────┐
│ Fetch & Chunk   │  wiki_to_kg_pipeline.ipynb
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/{timestamp}/            │
│   chunks.ipynb                              │  Markdown cells with source text
│                                             │  + context breadcrumbs + CID sigs
└────────┬────────────────────────────────────┘
         │ LLM extraction (per-chunk)
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/{timestamp}/            │
│   facts.ipynb                               │  Markdown cells with factual statements
│                                             │  + CID signatures
└────────┬────────────────────────────────────┘
         │ LLM conversion (per-chunk, with tool-based triple output)
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/{timestamp}/            │
│   rdf.ipynb                                 │  Raw cells with Turtle RDF
│                                             │  (triples emitted via tools, grouped by statement)
└────────┬────────────────────────────────────┘
         │ Export
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/{timestamp}/            │
│   triples.ttl                               │  Combined Turtle file
│   registry.json                             │  Entity registry snapshot
└─────────────────────────────────────────────┘

Schema Vocabulary Support:
┌─────────────────┐
│ schema_setup.   │  One-time setup: fetch schema.org,
│ ipynb           │  build embedding index
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ data/vocab_     │  Cached vocabulary embeddings
│ cache/          │  (schema.org classes & properties)
└────────┬────────┘
         │ loaded by
         ▼
┌─────────────────┐
│ schema_matcher. │  LLM tools for RDF generation
│ py              │  
└─────────────────┘

Source Code Modules (src/):
┌─────────────────┐
│ cid.py          │  Content ID (CID) hashing & signatures
│ entity_registry │  Entity tracking with stable URIs
│ notebook_gen... │  Notebook file generation
│ processors.py   │  Facts extraction & RDF generation loops
│ prompts.py      │  LLM prompt templates
│ rdf_tools.py    │  Tool definitions for vocabulary lookup & triple emission
│ section_parser  │  Wikipedia section hierarchy parsing
│ utils.py        │  Logging, chunking utilities
└─────────────────┘
```

### Output Directory Structure

Each article gets its own subdirectory under `data/`, with a timestamp subdirectory for each run. This allows files within a run to reference each other with simple relative paths:

```
data/
├── vocab_cache/                  # Shared schema.org embeddings
│   ├── config.json               # Embedding model configuration
│   ├── schema.json               # Schema.org vocabulary terms
│   └── schema.npy                # Embedding vectors
└── albert_einstein/              # Per-article output
    ├── 20241218_143022/          # Run timestamp directory
    │   ├── source.ipynb          # Raw Wikipedia content (before chunking)
    │   ├── chunks.ipynb          # Source text chunks with context
    │   ├── facts.ipynb           # Extracted facts  
    │   ├── rdf.ipynb             # RDF triples (per-statement)
    │   ├── triples.ttl           # Combined Turtle export
    │   └── registry.json         # Entity registry snapshot
    └── 20241219_091500/          # Another run (preserved)
        ├── source.ipynb
        ├── chunks.ipynb
        ├── facts.ipynb
        └── ...
```

**Benefits of timestamp-as-subdirectory:**
- Simple filenames (`chunks.ipynb` instead of `chunks_20241218_143022.ipynb`)
- Files can reference each other with relative paths (e.g., `source_notebook: source.ipynb`)
- Multiple runs are preserved for comparison
- Clean directory listing per run

## Content ID (CID) System

Each generated cell includes a cryptographic signature for dependency tracking, using **IPFS Content Identifiers (CIDs)** and **REPRODUCE-ME ontology** for provenance representation.

### IPFS CIDs

Content identifiers are computed using the IPFS CID format (CIDv1, 'raw' codec, SHA2-256 hash) via the `multiformats` library:

```python
from multiformats import CID
import hashlib

sha256 = hashlib.sha256(content.encode('utf-8')).digest()
cid = CID("base32", 1, "raw", ("sha2-256", sha256))
# Result: "bafkreihdwdcefgh4dqkjv67uzcmw7o..." (base32 encoded)
```

### REPRODUCE-ME Provenance

Provenance follows the **REPRODUCE-ME ontology** (https://w3id.org/reproduceme), an extension of PROV-O designed for scientific computation with Jupyter notebooks. Signatures are stored as **JSON-LD** documents using standard PROV-O vocabulary.

### JSON-LD Signature Format

Signatures use JSON-LD with PROV-O/REPRODUCE-ME vocabulary, making them directly usable as linked data:

```json
{
  "@context": {
    "prov": "http://www.w3.org/ns/prov#",
    "repro": "https://w3id.org/reproduceme#",
    "dcterms": "http://purl.org/dc/terms/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov:wasDerivedFrom": {"@type": "@id"}
  },
  "@id": "ipfs://bafkreihdwdcefgh4dqkjv67uzcmw7o...",
  "@type": ["prov:Entity", "repro:InputData"],
  "dcterms:identifier": "bafkreihdwdcefgh4dqkjv67uzcmw7o...",
  "prov:label": "source:Albert Einstein",
  "prov:wasDerivedFrom": {"@id": "ipfs://bafkreiabcdef..."},
  "_cell": 1,
  "_type": "source"
}
```

**Key JSON-LD properties:**
- `@id` - IPFS URI of this entity (`ipfs://<CID>`)
- `@type` - RDF types (`prov:Entity` + REPRODUCE-ME subclass)
- `dcterms:identifier` - Raw CID string for indexing
- `prov:label` - Human-readable label
- `prov:wasDerivedFrom` - Link to source entity (the provenance chain)

**Pipeline metadata properties** (prefixed with `_`):
- `_cell` - Cell number in notebook
- `_type` - Content type (source, chunk, facts, rdf)
- `_stmt_key`, `_chunk_num`, `_stmt_idx` - RDF statement tracking

For RDF cells, signatures include statement-level tracking:

```json
{
  "@context": {...},
  "@id": "ipfs://bafkreixyz...",
  "@type": ["prov:Entity", "repro:OutputData"],
  "dcterms:identifier": "bafkreixyz...",
  "prov:label": "rdf:3_2",
  "prov:wasDerivedFrom": {"@id": "ipfs://bafkreiabc..."},
  "_cell": 5,
  "_type": "rdf",
  "_stmt_key": "3_2",
  "_chunk_num": 3,
  "_stmt_idx": 2
}
```

### REPRODUCE-ME Class Mapping

| Pipeline Stage | REPRODUCE-ME Class | Notes |
|---------------|-------------------|-------|
| `source.ipynb` | `repro:InputData` | Raw Wikipedia content |
| `chunks.ipynb` | `repro:Data` | Chunked text with context |
| `facts.ipynb` | `repro:Data` | Extracted factual statements |
| `rdf.ipynb` | `repro:OutputData` | Final RDF triples |

### CID Functions (src/cid.py)

- `compute_cid(content)` - Compute IPFS CID (CIDv1, raw, SHA2-256) for string/bytes content
- `cid_to_uri(cid)` - Convert CID string to `ipfs://` URI
- `uri_to_cid(uri)` - Extract CID from `ipfs://` URI
- `make_signature(...)` - Create JSON-LD signature with PROV-O/REPRODUCE-ME vocabulary
- `parse_signature(raw_content)` - Parse JSON-LD signature from raw cell (with legacy format support)
- `extract_signatures(notebook)` - Extract all signatures keyed by cell number
- `extract_statement_signatures(notebook)` - Extract RDF signatures keyed by `stmt_key`
- `generate_provenance_ttl(signatures)` - Generate Turtle format from JSON-LD signatures
- `collect_pipeline_signatures(output_dir)` - Collect all signatures from pipeline notebooks

### Incremental Processing Logic

For each source cell to process:

1. Look up existing signature for that cell number in target notebook
2. **No signature exists**: Generate new content, append content cell + signature
3. **Signature exists, `from_cid` matches**: Skip (content is up-to-date)
4. **Signature exists, `from_cid` differs**: Remove old cells, regenerate, append

For RDF generation (per-chunk):
1. Parse all statements from the facts chunk
2. Compute combined CID for all statements in the chunk
3. If chunk-level CID matches, skip entire chunk
4. Otherwise, process all statements in one LLM call with statement IDs
5. Group emitted triples by statement ID and write per-statement cells
6. Save notebook after each chunk for incremental persistence

This ensures:
- Changed source content triggers regeneration
- Unchanged content is preserved (including human edits)
- Interrupted runs can resume from last saved chunk

## Intermediate Notebook Structure

**Note:** These notebooks use a **natural-language-as-code** paradigm:
- **Code cells** contain the content being processed (source text, chunks, facts, RDF triples)
- **Raw cells** contain metadata (entity registry, JSON-LD signatures)
- **Markdown cells** contain human-readable documentation about the processing

### Source Notebook (`source.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance metadata (YAML) |
| 1 | Raw | Entity registry (JSON) |
| 2 | Code | Raw Wikipedia content (exact text from source) |
| 3 | Raw | Source CID signature (JSON-LD) |

The source notebook preserves the exact Wikipedia content as fetched, before any chunking or processing. The `prov:wasDerivedFrom` links to the source URL's CID. This provides the root of the provenance chain.

### Chunks Notebook (`chunks.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance metadata (YAML) |
| 1 | Raw | Entity registry (JSON) |
| 2 | Markdown | Chunk 1 content with context |
| 3 | Raw | Chunk 1 CID signature (`from_cid` = source CID) |
| 4 | Markdown | Chunk 2 content with context |
| 5 | Raw | Chunk 2 CID signature (`from_cid` = source CID) |
| ... | ... | ... |

**Chunk cell format:**
```markdown
**Context:** Albert Einstein > Early life
**Chunk:** 1 of 63

---

<source text content>
```

### Facts Notebook (`facts.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance + prompt template |
| 1 | Raw | Entity registry (JSON) |
| 2 | Markdown | Facts for chunk 1 |
| 3 | Raw | Facts 1 CID signature |
| ... | ... | ... |

**Facts cell format:**
```markdown
**Context:** Albert Einstein > Early life
**Chunk:** 1 of 63

---

- Albert Einstein was born on March 14, 1879.
- Albert Einstein was born in Ulm, in the Kingdom of Württemberg.
- ...
```

### RDF Notebook (`rdf.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance + prefixes + prompt template |
| 1 | Raw | Entity registry (JSON) |
| 2 | Raw | RDF triples for statement 1.1 |
| 3 | Raw | Statement 1.1 CID signature |
| 4 | Raw | RDF triples for statement 1.2 |
| 5 | Raw | Statement 1.2 CID signature |
| ... | ... | ... |

**RDF cell format:**
```turtle
# Statement [1.1]: Albert Einstein was born on March 14, 1879.
<#person_albert_einstein> rdf:type schema:Person .
<#person_albert_einstein> schema:birthDate "1879-03-14"^^xsd:date .
```

Each statement gets its own cell with:
- Comment showing chunk.statement index and original statement text
- Triples emitted for that specific statement
- If no triples emitted, includes debug info about tool calls made
```

## Entity Registry

Tracks entities across chunks with stable URIs derived from the source URL.

```python
@dataclass
class EntityRegistry:
    source_url: str
    entities: dict   # normalized_key -> entity
    aliases: dict    # alias -> canonical_key
```

### Entity Structure

```json
{
  "id": "person_albert_einstein",
  "uri": "https://en.wikipedia.org/wiki/Albert_Einstein#person_albert_einstein",
  "label": "Albert Einstein",
  "type": "Person",
  "descriptions": ["Subject of Wikipedia article"],
  "source_chunks": [1, 2, 5],
  "aliases": ["Einstein"]
}
```

### URI Strategy

- **Entities**: `{source_url}#{type}_{normalized_label}`
  - Example: `https://en.wikipedia.org/wiki/Albert_Einstein#person_albert_einstein`
- **Pipeline vocabulary**: `https://wiki3.ai/vocab/` prefix
- **Standard vocabulary**: `https://schema.org/` for common predicates

## RDF Vocabulary

### Prefixes

```turtle
@prefix schema: <https://schema.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix wiki3: <https://wiki3.ai/vocab/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix repro: <https://w3id.org/reproduceme#> .
@prefix pplan: <http://purl.org/net/p-plan#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@base <{source_url}> .
```

### Guidelines

- Use `schema:` for standard predicates (birthDate, birthPlace, worksFor, etc.)
- Use `wiki3:` for domain-specific predicates not in schema.org
- Use `xsd:` datatypes for dates, numbers
- Include `rdfs:label` for entity names

## Section Hierarchy & Breadcrumbs

Wikipedia `== headers ==` are parsed into a hierarchical structure:

```python
def extract_section_hierarchy(content: str) -> list[dict]
def get_section_context(position: int, sections: list, article_title: str) -> dict
```

Each chunk includes a breadcrumb like:
```
Albert Einstein > Early life > Education
```

This provides context for the LLM and helps with entity disambiguation.

## Timeout Handling

LLM calls are protected with HTTP-level timeouts on the `ChatOpenAI` client:

```python
CELL_TIMEOUT_SECONDS = 60  # 1 minute per cell

llm = ChatOpenAI(
    ...
    timeout=CELL_TIMEOUT_SECONDS,  # HTTP timeout
    max_retries=0,  # Don't retry on timeout
)
```

> **Note**: SIGALRM-based timeouts don't work for blocking HTTP I/O operations. 
> The HTTP-level timeout parameter is required for proper timeout behavior.

Cells that timeout are marked with error content:
```
# Error: TimeoutError after 60s
```

## Configuration

```python
ARTICLE_TITLE = "Albert Einstein"
OUTPUT_DIR = "data"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 128
CELL_TIMEOUT_SECONDS = 60   # HTTP timeout per LLM call
MAX_ITERATIONS = 150        # Max tool-calling iterations per chunk

LLM_CONFIG = {
    "provider": "lm_studio",
    "model": "qwen/qwen3-coder-30b",
    "temperature": 1,
    "base_url": "http://host.docker.internal:1234/v1",
}

VOCAB_CACHE_DIR = "data/vocab_cache"

# Continue from a previous run (set to None for fresh run)
CONTINUE_FROM_RUN = None  # e.g., "data/albert_einstein/20241218_143022"
```

## Output Files

Files are organized by article in timestamped subdirectories:

| File Pattern | Description |
|------|-------------|
| `data/{article}/{timestamp}/source.ipynb` | Raw Wikipedia content (before chunking) |
| `data/{article}/{timestamp}/chunks.ipynb` | Chunked source text with context |
| `data/{article}/{timestamp}/facts.ipynb` | Extracted factual statements |
| `data/{article}/{timestamp}/rdf.ipynb` | RDF triples (per-statement cells) |
| `data/{article}/{timestamp}/triples.ttl` | Combined Turtle file for import |
| `data/{article}/{timestamp}/registry.json` | Entity registry snapshot |
| `data/vocab_cache/config.json` | Embedding model configuration |
| `data/vocab_cache/schema.json` | Cached schema.org vocabulary |
| `data/vocab_cache/schema.npy` | Cached schema.org embeddings |

## Workflow

### First-Time Setup

1. Run `schema_setup.ipynb` to build vocabulary embedding index
2. This creates `data/vocab_cache/` with schema.org terms
3. Only needs to be done once (or when updating vocabularies)

### Initial Run

1. Configure article title and LLM settings in `wiki_to_kg_pipeline.ipynb`
2. Run all cells in sequence
3. Pipeline generates all intermediate notebooks
4. Final `.ttl` file is exported

### Incremental Update

1. Re-run pipeline cells
2. CID system detects which cells are up-to-date
3. Only changed/new content is regenerated
4. Existing content (including human edits) is preserved

### Manual Correction

1. Open intermediate notebook (e.g., `_facts.ipynb`)
2. Edit content cells as needed
3. **Important**: Also update or delete the corresponding signature cell
4. Re-run export cell to regenerate `.ttl`

To force regeneration of a specific cell:
- Delete both the content cell and its signature cell
- Re-run the processing cell

## License & Attribution

- Source content: Wikipedia (CC BY-SA 4.0)
- Provenance metadata includes license, attribution, and source URL
- Generated RDF includes provenance comments

## Schema Vocabulary Matching

The pipeline includes an embedding-based vocabulary matcher (`schema_matcher.py`) to ensure the LLM uses real schema.org terms instead of inventing custom `wiki3:` predicates.

### Setup Notebook (`schema_setup.ipynb`)

Run once to build the vocabulary index:

1. Fetches schema.org vocabulary (JSON-LD)
2. Extracts ~800 classes and ~1400 properties
3. Builds embedding index using local LM Studio model
4. Saves index to `data/vocab_cache/` for reuse

### Components

- **`VocabTerm`**: Dataclass representing a class or property from an RDF vocabulary
- **`SchemaVocabulary`**: Manages a vocabulary with embedding-based search
- **`SchemaMatcher`**: Multi-vocabulary matcher with methods callable as LLM tools

### LLM Tool Integration

The RDF generation stage provides four tools to the LLM via LangChain's tool calling:

**Lookup Tools** (find vocabulary terms):
```python
@tool
def find_rdf_class(description: str) -> str:
    """Find the best schema.org class/type for an entity."""
    # Returns top 5 matches with URIs, scores, descriptions

@tool  
def find_rdf_property(description: str, subject_type: str = "", object_type: str = "") -> str:
    """Find the best schema.org property/predicate for a relationship."""
    # Returns top 5 matches with URIs, domains, ranges, descriptions
```

**Output Tools** (emit structured triples - must include statement_id):
```python
@tool
def emit_triple(statement_id: str, subject: str, predicate: str, object_value: str) -> str:
    """Emit a single RDF triple with statement provenance."""
    # Validates inputs, returns OK/INVALID with guidance
    # statement_id links triple back to source statement (e.g., "1", "2")

@tool
def emit_triples(triples: List[dict]) -> str:
    """Emit multiple RDF triples at once (more efficient)."""
    # Each dict has: statement_id, subject, predicate, object keys
    # Flexible key matching (accepts variations like 'id', 's', 'p', 'o')
    # Returns OK/PARTIAL/ERROR with details on what was accepted/rejected
```

### Tool-Based Triple Output

Instead of asking the LLM to generate Turtle syntax directly (which often includes markdown commentary), the pipeline uses output tools:

1. LLM receives all statements from a chunk with numbered IDs (e.g., `[1] Statement text...`)
2. LLM calls `find_rdf_class` and `find_rdf_property` to discover vocabulary terms
3. LLM calls `emit_triple` or `emit_triples` for each statement, including the statement_id
4. Pipeline collects emitted triples and groups them by statement_id
5. Each statement gets its own cell in the RDF notebook

Benefits:
- Clean Turtle output without markdown artifacts
- Structured triple data for validation
- Statement-level provenance tracking
- Per-chunk LLM calls (faster than per-statement)
- Tool validation with corrective feedback allows LLM to self-correct

### Tool Validation & Self-Correction

The emit tools validate inputs and return actionable feedback:

```python
# Valid call
emit_triple("1", "<#einstein>", "schema:birthDate", '"1879-03-14"^^xsd:date')
# Returns: "OK: recorded triple for statement 1"

# Invalid call (missing required arg)
emit_triple("", "<#einstein>", "schema:birthDate", '"1879-03-14"^^xsd:date')
# Returns: "INVALID - please fix and retry: statement_id is required (e.g., '1', '2')"

# Partial success with emit_triples
emit_triples([{...valid...}, {...missing object...}])
# Returns: "PARTIAL: recorded 1 triples, skipped 1: item 1: missing ['object'], got keys: ['statement_id', 'subject', 'predicate']"
```

This allows the LLM to correct mistakes in subsequent iterations.

### Tool-Calling Loop

The RDF generation processes one chunk at a time, with all statements from that chunk:

```
┌─────────────────────────────────────────────────────┐
│ LLM receives chunk's statements with IDs:           │
│   [1] Statement one...                              │
│   [2] Statement two...                              │
│   [3] Statement three...                            │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │  LLM decides next action  │◄────────────────┐
         └─────────────┬─────────────┘                 │
                       │                               │
        ┌──────────────┼──────────────┐                │
        ▼              ▼              ▼                │
┌───────────┐  ┌───────────┐  ┌───────────┐           │
│find_class │  │find_prop  │  │emit_triple│           │
│           │  │           │  │emit_triples           │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │
      │              │              │                  │
      │              │     (includes statement_id)     │
      └──────────────┼──────────────┘                  │
                     │                                 │
         ┌───────────▼───────────┐                     │
         │ Tool result added to  │─────────────────────┘
         │ conversation (OK/ERROR)│
         └───────────────────────┘
                       │
                       │ (no more tool calls)
                       ▼
         ┌───────────────────────┐
         │ Return summary +      │
         │ collected triples     │
         │ (grouped by stmt_id)  │
         └───────────────────────┘
```

### Per-Chunk Processing Flow

```python
# In processors.py process_rdf_generation():
for facts_chunk in facts_data:
    # Parse statements: ["stmt1", "stmt2", ...]
    statements = parse_statements(facts_chunk["facts_text"])
    
    # Format for prompt with IDs
    statements_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(statements)])
    
    # Single LLM call for entire chunk
    summary, triples, iterations, tool_log = call_llm_with_tools(...)
    
    # Group triples by statement_id
    triples_by_stmt = {}
    for t in triples:
        stmt_id = t["statement_id"]  # "1", "2", etc.
        triples_by_stmt.setdefault(stmt_id, []).append(t)
    
    # Write one cell per statement
    for stmt_id, stmt_triples in triples_by_stmt.items():
        # Write Turtle with statement comment
        # Write signature cell
    
    # Save notebook after each chunk (incremental)
    nbformat.write(rdf_nb, rdf_path)
```

### Embedding Model

Configured in LM Studio:
- **Model**: `s3dev-ai/text-embedding-nomic-embed-text-v1.5`
- **Dimensions**: 768
- **API**: OpenAI-compatible at `http://host.docker.internal:1234/v1`

Alternative models:
- `nomic-ai/nomic-embed-text-v1.5` - 768 dimensions
- `BAAI/bge-small-en-v1.5` - 384 dimensions, faster

### Example Tool Usage

```python
# Lookup tools - find vocabulary
find_rdf_class("a famous scientist who won the Nobel Prize")
# Returns: schema:Person (0.82), schema:Researcher (0.78), ...

find_rdf_property("the date someone was born", subject_type="Person")
# Returns: schema:birthDate (0.91), schema:deathDate (0.72), ...

# Output tools - emit triples with statement_id
emit_triple("1", "<#person_einstein>", "rdf:type", "schema:Person")
# Returns: "OK: recorded triple for statement 1"

emit_triples([
    {"statement_id": "1", "subject": "<#person_einstein>", "predicate": "schema:name", "object": '"Albert Einstein"'},
    {"statement_id": "2", "subject": "<#person_einstein>", "predicate": "schema:birthDate", "object": '"1879-03-14"^^xsd:date'}
])
# Returns: "OK: recorded 2 triples"
```

## Source Code Modules

| File | Description |
|------|-------------|
| `src/cid.py` | Content ID hashing, signature creation/parsing |
| `src/entity_registry.py` | Entity tracking with stable URI generation |
| `src/notebook_generators.py` | Generate chunks/facts/rdf notebook files |
| `src/processors.py` | Main processing loops for facts extraction and RDF generation |
| `src/prompts.py` | LLM prompt templates for facts and RDF stages |
| `src/rdf_tools.py` | Tool definitions (find_rdf_class, emit_triple, etc.) |
| `src/section_parser.py` | Wikipedia section hierarchy parsing |
| `src/utils.py` | Logging utilities, contextual chunk creation |

## Future Considerations

- Entity extraction during facts stage (currently seeded manually)
- Cross-article entity linking
- Validation of generated RDF syntax (rdflib parsing)
- SHACL shape validation
- Multi-article batch processing
- Additional vocabularies (Dublin Core, FOAF, etc.)
- Parallel tool calls for faster vocabulary lookup
- Caching of common term lookups
- Human review workflow integration

## Files Overview

| File | Description |
|------|-------------|
| `wiki_to_kg_pipeline.ipynb` | Main orchestrator notebook |
| `schema_setup.ipynb` | One-time vocabulary index setup |
| `schema_matcher.py` | Embedding-based vocabulary matcher |
| `src/` | Python modules (see Source Code Modules above) |
| `DESIGN.md` | This design document |
| `data/{article}/{timestamp}/source.ipynb` | Raw Wikipedia content (provenance root) |
| `data/{article}/{timestamp}/chunks.ipynb` | Chunked source text with context |
| `data/{article}/{timestamp}/facts.ipynb` | Extracted factual statements |
| `data/{article}/{timestamp}/rdf.ipynb` | RDF triples (per-statement) |
| `data/{article}/{timestamp}/triples.ttl` | Combined Turtle file for import |
| `data/{article}/{timestamp}/registry.json` | Entity registry snapshot |
| `data/vocab_cache/` | Cached schema.org embeddings |
