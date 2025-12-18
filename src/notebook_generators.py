"""Functions to generate output notebooks."""

import json
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_raw_cell

from .cid import compute_cid, make_signature
from .entity_registry import EntityRegistry
from .prompts import RDF_GENERATION_PROMPT, RDF_PREFIXES


def generate_chunks_notebook(
    chunks: list,
    provenance: dict,
    registry: EntityRegistry,
    chunk_size: int,
    chunk_overlap: int,
    output_path: str
) -> str:
    """Generate a notebook with chunked source text and context metadata.
    
    Each chunk cell is followed by a signature raw cell with its CID.
    """
    nb = new_notebook()
    
    # Cell 0: Provenance markdown
    provenance_yaml = f"""# Chunked Text: {provenance['article_title']}

## Provenance

```yaml
source_url: {provenance['source_url']}
article_title: {provenance['article_title']}
fetched_at: {provenance['fetched_at']}
content_length: {provenance['content_length']}
license: {provenance['license']}
license_url: {provenance['license_url']}
attribution: {provenance['attribution']}
chunk_size: {chunk_size}
chunk_overlap: {chunk_overlap}
total_chunks: {len(chunks)}
generated_by: wiki_to_kg_pipeline.ipynb
generated_at: {datetime.now().isoformat()}
```

## Processing Instructions

Each chunk below contains source text with contextual metadata. The context line (before the separator) provides:
- **Context**: Hierarchical breadcrumb showing article > section path
- **Chunk**: Position in sequence

The text below the `---` separator is the unchanged source content.
Each chunk is followed by a signature cell containing its Content ID (CID).
"""
    nb.cells.append(new_markdown_cell(provenance_yaml))
    
    # Cell 1: Entity registry (raw cell)
    nb.cells.append(new_raw_cell(registry.to_json()))
    
    # Chunk cells with signatures
    # Compute CID of raw source for provenance
    source_cid = compute_cid(provenance['source_url'] + str(provenance['content_length']))
    
    for chunk in chunks:
        # Content cell
        chunk_content = f"""**Context:** {chunk.breadcrumb}
**Chunk:** {chunk.chunk_index + 1} of {chunk.total_chunks}

---

{chunk.content}
"""
        nb.cells.append(new_markdown_cell(chunk_content))
        
        # Signature cell
        chunk_cid = compute_cid(chunk_content)
        signature = make_signature(
            cell_num=chunk.chunk_index + 1,
            cell_type="chunk",
            cid=chunk_cid,
            from_cid=source_cid
        )
        nb.cells.append(new_raw_cell(json.dumps(signature)))
    
    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    return output_path


def generate_facts_notebook(
    chunks: list,
    provenance: dict,
    registry: EntityRegistry,
    llm_config: dict,
    source_notebook: str,
    output_path: str
) -> str:
    """Generate a notebook structure for extracted facts.
    
    Creates header cells only - actual fact content is populated by the pipeline
    during the facts extraction phase.
    """
    nb = new_notebook()
    
    # Cell 0: Provenance markdown
    provenance_yaml = f"""# Extracted Facts: {provenance['article_title']}

## Provenance

```yaml
source_notebook: {source_notebook}
source_url: {provenance['source_url']}
article_title: {provenance['article_title']}
license: {provenance['license']}
license_url: {provenance['license_url']}
extraction_model: {llm_config['model']}
extraction_timestamp: {datetime.now().isoformat()}
total_chunks: {len(chunks)}
generated_by: wiki_to_kg_pipeline.ipynb
```

## Processing Instructions

This notebook contains factual statements extracted from source chunks.
Each facts cell is followed by a signature cell with CID provenance linking back to its source chunk.

To regenerate a specific cell: delete both the content cell and its signature, then re-run the pipeline.

Structure:
- **Context**: Section breadcrumb from source
- **Chunk**: Position in sequence
- **Facts**: Bulleted list of extracted factual statements
"""
    nb.cells.append(new_markdown_cell(provenance_yaml))
    
    # Cell 1: Entity registry (raw cell)
    nb.cells.append(new_raw_cell(registry.to_json()))
    
    # Write notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    return output_path


def generate_rdf_notebook_header(
    provenance: dict,
    registry: EntityRegistry,
    llm_config: dict
) -> nbformat.NotebookNode:
    """Generate just the header cells for RDF notebook."""
    nb = new_notebook()
    
    formatted_prefixes = RDF_PREFIXES.format(source_url=provenance['source_url'])
    provenance_md = f"""# RDF Triples: {provenance['article_title']}

## Provenance

```yaml
source_url: {provenance['source_url']}
article_title: {provenance['article_title']}
license: {provenance['license']}
license_url: {provenance['license_url']}
source_notebook: facts.ipynb
generated_by: wiki_to_kg_pipeline.ipynb
generated_at: {datetime.now().isoformat()}
llm_provider: {llm_config['provider']}
llm_model: {llm_config['model']}
llm_temperature: {llm_config['temperature']}
rdf_format: Turtle
```

## RDF Prefixes

```turtle
{formatted_prefixes}
```

## Processing Instructions

This notebook contains RDF triples in Turtle format, one cell per source statement.
Each content cell is followed by a signature cell with CID provenance.

To regenerate a specific cell: delete both the content cell and its signature, then re-run the pipeline.

## Prompt Template

```
{RDF_GENERATION_PROMPT}
```
"""
    nb.cells.append(new_markdown_cell(provenance_md))
    nb.cells.append(new_raw_cell(registry.to_json()))
    
    return nb
