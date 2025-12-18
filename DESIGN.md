# Wiki3 Knowledge Graph Pipeline Design

## Overview

The `wiki_to_kg_pipeline.ipynb` notebook implements a multi-stage pipeline for converting Wikipedia articles into RDF Knowledge Graphs. The pipeline uses intermediate Jupyter notebook files for each transformation stage, enabling human inspection, correction, and incremental reprocessing.

## Design Goals

1. **Transparency**: Each stage produces human-readable output that can be inspected and edited
2. **Provenance**: Full traceability from source content to final RDF triples
3. **Incremental Processing**: Only regenerate content when source material changes
4. **Interruptibility**: Pipeline can be stopped and resumed without losing progress
5. **Editability**: Humans (or other agents) can modify intermediate outputs

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
┌─────────────────┐
│ {article}_      │  Markdown cells with source text
│ chunks.ipynb    │  + context breadcrumbs + CID signatures
└────────┬────────┘
         │ LLM extraction
         ▼
┌─────────────────┐
│ {article}_      │  Markdown cells with factual statements
│ facts.ipynb     │  + CID signatures linking to source chunks
└────────┬────────┘
         │ LLM conversion
         ▼
┌─────────────────┐
│ {article}_      │  Raw cells with Turtle RDF
│ rdf.ipynb       │  + CID signatures linking to source facts
└────────┬────────┘
         │ Export
         ▼
┌─────────────────┐
│ {article}.ttl   │  Combined Turtle file with prefixes
└─────────────────┘
```

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

Long-running LLM calls are protected with SIGALRM-based timeouts:

```python
CELL_TIMEOUT_SECONDS = 300  # 5 minutes

@contextmanager
def timeout_context(seconds):
    # Uses signal.SIGALRM for timeout
```

Cells that timeout are marked with error content:
```
# Error: Timeout after 300s
```

## Configuration

```python
ARTICLE_TITLE = "Albert Einstein"
OUTPUT_DIR = "data"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 128

LLM_CONFIG = {
    "provider": "lm_studio",
    "model": "qwen/qwen3-coder-30b",
    "temperature": 1,
    "base_url": "http://host.docker.internal:1234/v1",
}
```

## Output Files

| File | Description |
|------|-------------|
| `data/{article}_chunks.ipynb` | Chunked source text with context |
| `data/{article}_facts.ipynb` | Extracted factual statements |
| `data/{article}_rdf.ipynb` | RDF triples in Turtle format |
| `data/{article}.ttl` | Combined Turtle file for import |
| `data/entity_registry.json` | Shared entity registry |

## Workflow

### Initial Run

1. Configure article title and LLM settings
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

The pipeline includes an embedding-based vocabulary matcher (`schema_matcher.py`) to help the LLM select appropriate schema.org terms instead of inventing custom `wiki3:` predicates.

### Components

- **`VocabTerm`**: Represents a class or property from an RDF vocabulary
- **`SchemaVocabulary`**: Manages a vocabulary with embedding-based search
- **`SchemaMatcher`**: Multi-vocabulary matcher with LLM-callable interface

### Embedding Model

Recommended embedding models for LM Studio:
- `nomic-ai/nomic-embed-text-v1.5` - 768 dimensions, good quality
- `BAAI/bge-small-en-v1.5` - 384 dimensions, faster

### API Functions

```python
# Find matching RDF classes
matcher.find_class("a famous scientist") -> [{"uri": "...", "prefix": "schema:Person", ...}]

# Find matching RDF properties  
matcher.find_property("the date someone was born", subject_type="Person")

# Find all components of a triple
matcher.find_triple_components(
    subject_desc="Albert Einstein, a physicist",
    predicate_desc="was born in",
    object_desc="the city of Ulm"
)
```

### Setup

Run `schema_setup.ipynb` to:
1. Fetch schema.org vocabulary from web
2. Build embedding index for all terms
3. Save index to `data/vocab_cache/` for reuse

## Future Considerations

- Entity extraction during facts stage (currently seeded manually)
- Cross-article entity linking
- Validation of generated RDF syntax
- SHACL shape validation
- Multi-article batch processing
- Additional vocabularies (Dublin Core, FOAF, etc.)
