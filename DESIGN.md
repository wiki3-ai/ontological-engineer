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
│ data/{article_slug}/                        │
│   chunks_{timestamp}.ipynb                  │  Markdown cells with source text
│                                             │  + context breadcrumbs + CID sigs
└────────┬────────────────────────────────────┘
         │ LLM extraction
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/                        │
│   facts_{timestamp}.ipynb                   │  Markdown cells with factual statements
│                                             │  + CID signatures
└────────┬────────────────────────────────────┘
         │ LLM conversion (with tool-based triple output)
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/                        │
│   rdf_{timestamp}.ipynb                     │  Raw cells with Turtle RDF
│                                             │  (triples emitted via tools)
└────────┬────────────────────────────────────┘
         │ Export
         ▼
┌─────────────────────────────────────────────┐
│ data/{article_slug}/                        │
│   triples_{timestamp}.ttl                   │  Combined Turtle file
│   registry_{timestamp}.json                 │  Entity registry snapshot
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
```

### Output Directory Structure

Each article gets its own subdirectory under `data/` based on the article slug:

```
data/
├── vocab_cache/           # Shared schema.org embeddings
│   ├── schema.json
│   └── schema.npy
└── albert_einstein/       # Per-article output
    ├── chunks_20241218_143022.ipynb
    ├── facts_20241218_143022.ipynb
    ├── rdf_20241218_143022.ipynb
    ├── triples_20241218_143022.ttl
    └── registry_20241218_143022.json
```

Timestamped filenames allow multiple runs to be preserved for comparison.

## Content ID (CID) System

Each generated cell includes a cryptographic signature for dependency tracking:

```json
{
  "cell": 1,
  "type": "chunk|facts|rdf",
  "cid": "<SHA256 hash of cell content>",
  "from_cid": "<SHA256 hash of source cell content>"
}
```

### CID Functions

- `compute_cid(content)` - SHA256 hash of string content
- `make_signature(cell_num, type, cid, from_cid)` - Create signature dict
- `parse_signature(raw_content)` - Parse JSON signature from raw cell
- `extract_signatures(notebook)` - Extract all signatures keyed by cell number

### Incremental Processing Logic

For each source cell to process:

1. Look up existing signature for that cell number in target notebook
2. **No signature exists**: Generate new content, append content cell + signature
3. **Signature exists, `from_cid` matches**: Skip (content is up-to-date)
4. **Signature exists, `from_cid` differs**: Remove old cells, regenerate, append

This ensures:
- Changed source content triggers regeneration
- Unchanged content is preserved (including human edits)
- No placeholder content needed to detect missing cells

## Intermediate Notebook Structure

### Chunks Notebook (`{article}_chunks.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance metadata (YAML) |
| 1 | Raw | Entity registry (JSON) |
| 2 | Markdown | Chunk 1 content with context |
| 3 | Raw | Chunk 1 CID signature |
| 4 | Markdown | Chunk 2 content with context |
| 5 | Raw | Chunk 2 CID signature |
| ... | ... | ... |

**Chunk cell format:**
```markdown
**Context:** Albert Einstein > Early life
**Chunk:** 1 of 63

---

<source text content>
```

### Facts Notebook (`{article}_facts.ipynb`)

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

### RDF Notebook (`{article}_rdf.ipynb`)

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance + prefixes + prompt template |
| 1 | Raw | Entity registry (JSON) |
| 2 | Raw | RDF triples for facts 1 |
| 3 | Raw | RDF 1 CID signature |
| ... | ... | ... |

**RDF cell format:**
```turtle
# Context: Albert Einstein > Early life
# Cell: 1 of 63

<https://en.wikipedia.org/wiki/Albert_Einstein#person_albert_einstein>
    <https://schema.org/birthDate> "1879-03-14"^^<http://www.w3.org/2001/XMLSchema#date> ;
    <https://schema.org/birthPlace> <https://en.wikipedia.org/wiki/Albert_Einstein#place_ulm> .
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
CELL_TIMEOUT_SECONDS = 60  # HTTP timeout per LLM call

LLM_CONFIG = {
    "provider": "lm_studio",
    "model": "qwen/qwen3-coder-30b",
    "temperature": 1,
    "base_url": "http://host.docker.internal:1234/v1",
}

VOCAB_CACHE_DIR = "data/vocab_cache"

# Generated at runtime
RUN_TIMESTAMP = "20241218_143022"  # YYYYMMDD_HHMMSS
ARTICLE_OUTPUT_DIR = "data/albert_einstein"
```

## Output Files

Files are organized by article in timestamped subdirectories:

| File Pattern | Description |
|------|-------------|
| `data/{article}/chunks_{timestamp}.ipynb` | Chunked source text with context |
| `data/{article}/facts_{timestamp}.ipynb` | Extracted factual statements |
| `data/{article}/rdf_{timestamp}.ipynb` | RDF triples via tool-based output |
| `data/{article}/triples_{timestamp}.ttl` | Combined Turtle file for import |
| `data/{article}/registry_{timestamp}.json` | Entity registry snapshot |
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

**Output Tools** (emit structured triples):
```python
@tool
def emit_triple(subject: str, predicate: str, object_value: str) -> str:
    """Emit a single RDF triple."""
    # Collects triple for later conversion to Turtle

@tool
def emit_triples(triples: List[dict]) -> str:
    """Emit multiple RDF triples at once (more efficient)."""
    # Each dict has: subject, predicate, object keys
```

### Tool-Based Triple Output

Instead of asking the LLM to generate Turtle syntax directly (which often includes markdown commentary), the pipeline uses output tools:

1. LLM calls `find_rdf_class` and `find_rdf_property` to discover vocabulary terms
2. LLM calls `emit_triple` or `emit_triples` to output each triple structurally
3. LLM provides a brief summary with any quality/accuracy concerns
4. Pipeline collects emitted triples and converts to Turtle format

Benefits:
- Clean Turtle output without markdown artifacts
- Structured triple data for validation
- LLM can report concerns separately from data
- Easier to post-process or transform

### Tool-Calling Loop

```
┌─────────────────────────────────────────────────────┐
│ LLM receives facts + prompt                          │
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
      └──────────────┼──────────────┘                  │
                     │                                 │
         ┌───────────▼───────────┐                     │
         │ Tool result added to  │─────────────────────┘
         │ conversation          │
         └───────────────────────┘
                       │
                       │ (no more tool calls)
                       ▼
         ┌───────────────────────┐
         │ Return summary +      │
         │ collected triples     │
         └───────────────────────┘
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

# Output tools - emit triples
emit_triple("<#person_einstein>", "rdf:type", "schema:Person")
# Returns: "Triple recorded: <#person_einstein> rdf:type schema:Person"

emit_triples([
    {"subject": "<#person_einstein>", "predicate": "schema:name", "object": '"Albert Einstein"'},
    {"subject": "<#person_einstein>", "predicate": "schema:birthDate", "object": '"1879-03-14"^^xsd:date'}
])
# Returns: "Recorded 2 triples"
```

## Future Considerations

- Entity extraction during facts stage (currently seeded manually)
- Cross-article entity linking
- Validation of generated RDF syntax
- SHACL shape validation
- Multi-article batch processing
- Additional vocabularies (Dublin Core, FOAF, etc.)
- Parallel tool calls for faster vocabulary lookup
- Caching of common term lookups

## Files Overview

| File | Description |
|------|-------------|
| `wiki_to_kg_pipeline.ipynb` | Main orchestrator notebook |
| `schema_setup.ipynb` | One-time vocabulary index setup |
| `schema_matcher.py` | Embedding-based vocabulary matcher |
| `DESIGN.md` | This design document |
| `data/{article}_chunks.ipynb` | Chunked source text with context |
| `data/{article}_facts.ipynb` | Extracted factual statements |
| `data/{article}_rdf.ipynb` | RDF triples in Turtle format |
| `data/{article}.ttl` | Combined Turtle file for import |
| `data/entity_registry.json` | Shared entity registry |
| `data/vocab_cache/` | Cached schema.org embeddings |
