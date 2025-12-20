"""
Provenance tracking for DSPy training pipeline.

Generates and manages Jupyter notebooks with CID-based provenance:
- statements.ipynb: Extracted statements per chunk
- classifications.ipynb: Per-statement GOOD/BAD classifications

Uses the same CID system as the original pipeline (src/cid.py).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_raw_cell

# Import from the original cid module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cid import compute_cid, make_signature, cid_to_uri, extract_signatures


# =============================================================================
# Statements Notebook
# =============================================================================

def generate_statements_notebook_header(
    provenance: Dict[str, Any],
    model_config: Dict[str, Any],
) -> nbformat.NotebookNode:
    """Generate header cells for statements.ipynb.
    
    Args:
        provenance: Dict with article_title, source_url, etc.
        model_config: Dict with model name, temperature, etc.
    
    Returns:
        NotebookNode with header cells (provenance + config)
    """
    nb = new_notebook()
    
    provenance_yaml = f"""# Extracted Statements: {provenance.get('article_title', 'Unknown')}

## Provenance

```yaml
source_notebook: chunks.ipynb
source_url: {provenance.get('source_url', 'N/A')}
article_title: {provenance.get('article_title', 'N/A')}
license: {provenance.get('license', 'CC BY-SA 4.0')}
extraction_model: {model_config.get('model', 'unknown')}
extraction_temperature: {model_config.get('temperature', 0.7)}
generated_by: ontological_engineer
generated_at: {datetime.now().isoformat()}
```

## Description

This notebook contains atomic statements extracted from Wikipedia chunks using DSPy.
Each statement is:
- Self-contained (understandable without original text)
- Preserves entity links: [Entity Name](/wiki/Entity_Name)
- Contains exactly one verifiable claim

Each content cell is followed by a CID signature linking back to its source chunk.
"""
    nb.cells.append(new_markdown_cell(provenance_yaml))
    
    # Config cell
    config = {
        "model": model_config.get("model"),
        "temperature": model_config.get("temperature"),
        "generated_at": datetime.now().isoformat(),
    }
    nb.cells.append(new_raw_cell(json.dumps(config, indent=2)))
    
    return nb


def append_statements_cell(
    nb: nbformat.NotebookNode,
    chunk_num: int,
    total_chunks: int,
    section_context: str,
    statements: List[str],
    chunk_cid: str,
) -> Tuple[str, Dict]:
    """Append a statements cell and signature to the notebook.
    
    Args:
        nb: Notebook to append to
        chunk_num: Chunk number (1-indexed)
        total_chunks: Total number of chunks
        section_context: Section breadcrumb
        statements: List of extracted statements
        chunk_cid: CID of the source chunk
    
    Returns:
        Tuple of (statements_cid, signature_dict)
    """
    # Format statements content
    statements_list = "\n".join(f"{i+1}. {stmt}" for i, stmt in enumerate(statements))
    
    content = f"""**Context:** {section_context}
**Chunk:** {chunk_num} of {total_chunks}
**Statements:** {len(statements)}

---

{statements_list}
"""
    nb.cells.append(new_markdown_cell(content))
    
    # Compute CID and create signature
    statements_cid = compute_cid(content)
    cell_num = len(nb.cells) - 1  # 0-indexed
    
    signature = make_signature(
        cell_num=cell_num,
        cell_type="statements",
        cid=statements_cid,
        from_cid=chunk_cid,
        repro_class="Data",
        label=f"statements:chunk_{chunk_num}",
        chunk_num=chunk_num,
    )
    nb.cells.append(new_raw_cell(json.dumps(signature, indent=2)))
    
    return statements_cid, signature


def save_statements_notebook(nb: nbformat.NotebookNode, output_path: Path) -> None:
    """Save the statements notebook to disk."""
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


def load_statements_from_notebook(notebook_path: Path) -> List[Dict[str, Any]]:
    """Load statements from a statements.ipynb file.
    
    Returns:
        List of dicts with keys: chunk_num, section_context, statements, cid, from_cid
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    results = []
    signatures = extract_signatures(nb)
    
    # Find statement cells (markdown cells followed by signature cells)
    i = 2  # Skip header cells
    while i < len(nb.cells) - 1:
        cell = nb.cells[i]
        next_cell = nb.cells[i + 1]
        
        if cell.cell_type == 'markdown' and next_cell.cell_type == 'raw':
            # Parse the markdown content
            content = cell.source
            try:
                sig = json.loads(next_cell.source)
                if sig.get('_type') == 'statements':
                    # Extract metadata from content
                    lines = content.split('\n')
                    section_context = ""
                    chunk_num = 0
                    statements = []
                    
                    in_statements = False
                    for line in lines:
                        if line.startswith('**Context:**'):
                            section_context = line.replace('**Context:**', '').strip()
                        elif line.startswith('**Chunk:**'):
                            chunk_part = line.replace('**Chunk:**', '').strip()
                            chunk_num = int(chunk_part.split(' of ')[0])
                        elif line == '---':
                            in_statements = True
                        elif in_statements and line.strip():
                            # Remove numbering prefix
                            stmt = line.strip()
                            if stmt and stmt[0].isdigit():
                                stmt = stmt.split('. ', 1)[-1]
                            if stmt:
                                statements.append(stmt)
                    
                    results.append({
                        'chunk_num': chunk_num,
                        'section_context': section_context,
                        'statements': statements,
                        'cid': sig.get('dcterms:identifier'),
                        'from_cid': sig.get('prov:wasDerivedFrom', {}).get('@id', '').replace('ipfs://', ''),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        
        i += 2  # Move to next content/signature pair
    
    return results


# =============================================================================
# Classifications Notebook
# =============================================================================

def generate_classifications_notebook_header(
    provenance: Dict[str, Any],
    model_config: Dict[str, Any],
) -> nbformat.NotebookNode:
    """Generate header cells for classifications.ipynb."""
    nb = new_notebook()
    
    provenance_yaml = f"""# Statement Classifications: {provenance.get('article_title', 'Unknown')}

## Provenance

```yaml
source_notebook: statements.ipynb
source_url: {provenance.get('source_url', 'N/A')}
article_title: {provenance.get('article_title', 'N/A')}
classifier_model: {model_config.get('model', 'unknown')}
classifier_temperature: {model_config.get('temperature', 0.7)}
generated_by: ontological_engineer.StatementClassifier
generated_at: {datetime.now().isoformat()}
```

## Description

This notebook contains per-statement quality classifications:
- **GOOD**: Atomic, accurate, links preserved
- **BAD**: Multiple claims, inaccurate, or missing links

Each classification cell is followed by a CID signature linking back to its source statements.
These classifications can be used for:
- Human review and correction
- Training data for judge improvement
- Quality metrics tracking
"""
    nb.cells.append(new_markdown_cell(provenance_yaml))
    
    # Config cell
    config = {
        "model": model_config.get("model"),
        "temperature": model_config.get("temperature"),
        "generated_at": datetime.now().isoformat(),
    }
    nb.cells.append(new_raw_cell(json.dumps(config, indent=2)))
    
    return nb


def append_classifications_cell(
    nb: nbformat.NotebookNode,
    chunk_num: int,
    total_chunks: int,
    section_context: str,
    classifications: List[Any],  # List of StatementClassification
    missing_facts: str,
    score: float,
    statements_cid: str,
) -> Tuple[str, Dict]:
    """Append a classifications cell and signature to the notebook.
    
    Args:
        nb: Notebook to append to
        chunk_num: Chunk number (1-indexed)
        total_chunks: Total number of chunks
        section_context: Section breadcrumb
        classifications: List of StatementClassification objects
        missing_facts: Missing facts string from classifier
        score: Overall score (fraction of GOOD)
        statements_cid: CID of the source statements
    
    Returns:
        Tuple of (classifications_cid, signature_dict)
    """
    good_count = sum(1 for c in classifications if c.is_good)
    total = len(classifications)
    
    # Build table rows
    table_rows = []
    for c in classifications:
        emoji = "✅" if c.is_good else "❌"
        # Truncate statement for table display
        stmt_short = c.statement[:60] + "..." if len(c.statement) > 60 else c.statement
        # Escape pipe characters in statement
        stmt_short = stmt_short.replace('|', '\\|')
        reason_short = c.reason[:40] + "..." if len(c.reason) > 40 else c.reason
        table_rows.append(f"| {c.index} | {emoji} {c.classification} | {stmt_short} | {reason_short} |")
    
    table = "\n".join(table_rows)
    
    content = f"""**Context:** {section_context}
**Chunk:** {chunk_num} of {total_chunks}
**Score:** {score:.1%} ({good_count}/{total} GOOD)

---

| # | Classification | Statement | Reason |
|---|----------------|-----------|--------|
{table}

**Missing facts:** {missing_facts if missing_facts else 'none'}
"""
    nb.cells.append(new_markdown_cell(content))
    
    # Compute CID and create signature
    classifications_cid = compute_cid(content)
    cell_num = len(nb.cells) - 1
    
    # Also store full classification data as JSON for machine processing
    classification_data = {
        "classifications": [
            {
                "index": c.index,
                "statement": c.statement,
                "classification": c.classification,
                "reason": c.reason,
            }
            for c in classifications
        ],
        "missing_facts": missing_facts,
        "score": score,
    }
    
    signature = make_signature(
        cell_num=cell_num,
        cell_type="classifications",
        cid=classifications_cid,
        from_cid=statements_cid,
        repro_class="Data",
        label=f"classifications:chunk_{chunk_num}",
        chunk_num=chunk_num,
    )
    # Add classification data to signature for easy retrieval
    signature["_classification_data"] = classification_data
    
    nb.cells.append(new_raw_cell(json.dumps(signature, indent=2)))
    
    return classifications_cid, signature


def save_classifications_notebook(nb: nbformat.NotebookNode, output_path: Path) -> None:
    """Save the classifications notebook to disk."""
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


def load_classifications_from_notebook(notebook_path: Path) -> List[Dict[str, Any]]:
    """Load classifications from a classifications.ipynb file.
    
    Returns:
        List of dicts with keys: chunk_num, section_context, classifications, score, cid, from_cid
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    results = []
    
    # Find classification cells
    i = 2  # Skip header cells
    while i < len(nb.cells) - 1:
        cell = nb.cells[i]
        next_cell = nb.cells[i + 1]
        
        if cell.cell_type == 'markdown' and next_cell.cell_type == 'raw':
            try:
                sig = json.loads(next_cell.source)
                if sig.get('_type') == 'classifications':
                    # Extract from signature's embedded data
                    classification_data = sig.get('_classification_data', {})
                    
                    # Parse chunk_num from label
                    label = sig.get('prov:label', '')
                    chunk_num = sig.get('_chunk_num', 0)
                    
                    results.append({
                        'chunk_num': chunk_num,
                        'classifications': classification_data.get('classifications', []),
                        'missing_facts': classification_data.get('missing_facts', ''),
                        'score': classification_data.get('score', 0.0),
                        'cid': sig.get('dcterms:identifier'),
                        'from_cid': sig.get('prov:wasDerivedFrom', {}).get('@id', '').replace('ipfs://', ''),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        
        i += 2
    
    return results


# =============================================================================
# Utility Functions
# =============================================================================

def get_processed_chunk_cids(notebook_path: Path) -> set:
    """Get set of chunk CIDs that have already been processed.
    
    Useful for incremental processing - skip chunks whose CID
    appears in the from_cid of existing content.
    """
    if not notebook_path.exists():
        return set()
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    signatures = extract_signatures(nb)
    
    from_cids = set()
    for sig in signatures.values():
        from_ref = sig.get('prov:wasDerivedFrom', {})
        if isinstance(from_ref, dict):
            cid = from_ref.get('@id', '').replace('ipfs://', '')
        else:
            cid = str(from_ref).replace('ipfs://', '')
        if cid:
            from_cids.add(cid)
    
    return from_cids


def create_output_directory(article_title: str, base_dir: Path = None) -> Path:
    """Create timestamped output directory for an article.
    
    Args:
        article_title: Wikipedia article title
        base_dir: Base data directory (default: data/)
    
    Returns:
        Path to the created directory
    """
    if base_dir is None:
        base_dir = Path("data")
    
    # Slugify article title
    slug = article_title.lower().replace(' ', '_').replace('/', '_')
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir
