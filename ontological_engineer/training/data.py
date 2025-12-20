"""Wikipedia data fetching with CID-based caching and provenance.

All fetched data is cached in Jupyter notebooks with CID signatures for:
- Reproducibility (same content = same CID)
- Provenance tracking (what data came from where)
- Incremental processing (skip already-fetched pages)

See DEVELOPMENT_PRACTICES.md for patterns.
"""

import json
import re
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from urllib.parse import quote

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_raw_cell

from ontological_engineer.cid import compute_cid, make_signature, extract_signatures


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WikipediaPage:
    """A Wikipedia page with metadata."""
    title: str
    views: int
    content: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    fetched_at: Optional[str] = None


@dataclass
class WikipediaChunk:
    """A chunk of Wikipedia text with context."""
    text: str
    section_context: str  # e.g., "Albert Einstein > Early life"
    chunk_num: int
    total_chunks: int
    page_title: str


# =============================================================================
# Pageviews API (top pages)
# =============================================================================

USER_AGENT = "Wiki3.ai OntologicalEngineer/0.1 (https://github.com/wiki3-ai/ontological-engineer)"


def fetch_top_pages(
    project: str = "en.wikipedia",
    access: str = "all-access",
    year: Optional[int] = None,
    month: Optional[int] = None,
    limit: int = 1000,
) -> List[WikipediaPage]:
    """Fetch top viewed Wikipedia pages for a given month.
    
    Uses the Wikimedia Pageviews API:
    https://wikimedia.org/api/rest_v1/
    
    Args:
        project: Wiki project (e.g., "en.wikipedia")
        access: Access type ("all-access", "desktop", "mobile-web")
        year: Year to fetch (default: last month)
        month: Month to fetch (default: last month)
        limit: Maximum number of pages to return
        
    Returns:
        List of WikipediaPage objects sorted by views (descending)
    """
    if year is None or month is None:
        last_month = datetime.now() - timedelta(days=30)
        year = last_month.year
        month = last_month.month
    
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/{access}/{year}/{month:02d}/all-days"
    headers = {"User-Agent": USER_AGENT}
    
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    
    # Extract articles (skip special pages)
    pages = []
    for item in data["items"][0]["articles"]:
        title = item["article"]
        views = item["views"]
        
        # Skip special pages, main page, and disambiguation
        if title.startswith(("Special:", "Wikipedia:", "File:", "Template:", "Category:", "Portal:")):
            continue
        if title in ("Main_Page", "-"):
            continue
        if "(disambiguation)" in title:
            continue
        
        pages.append(WikipediaPage(
            title=title.replace("_", " "),
            views=views,
        ))
        
        if len(pages) >= limit:
            break
    
    return pages


# =============================================================================
# Content Fetching
# =============================================================================

def fetch_page_content(title: str) -> Optional[str]:
    """Fetch the plain text content of a Wikipedia article.
    
    Args:
        title: Wikipedia article title
        
    Returns:
        Plain text content or None if not found
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                return None
            return page_data.get("extract", "")
    except Exception as e:
        print(f"  Error fetching {title}: {e}")
        return None


def chunk_article(
    title: str,
    text: str,
    max_chunk_size: int = 1500,
    min_chunk_size: int = 100,
) -> List[WikipediaChunk]:
    """Split article into chunks by section.
    
    Args:
        title: Article title
        text: Plain text content
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters (smaller chunks are skipped)
        
    Returns:
        List of WikipediaChunk objects
    """
    if not text:
        return []
    
    chunks = []
    
    # Split by section headers (== Header ==)
    sections = re.split(r'\n(==+\s+.+?\s+==+)\n', text)
    
    current_section = title
    current_text = ""
    
    for part in sections:
        # Check if this is a header
        if re.match(r'==+\s+.+?\s+==+', part):
            # Save previous chunk if it has content
            if current_text.strip() and len(current_text.strip()) >= min_chunk_size:
                chunks.append({
                    "text": current_text.strip(),
                    "section": current_section,
                })
            
            # Update section name
            current_section = part.strip('= \n')
            current_text = ""
        else:
            current_text += part
            
            # If chunk gets too large, split it
            while len(current_text) > max_chunk_size:
                # Find a good break point (paragraph)
                break_point = current_text.rfind('\n\n', 0, max_chunk_size)
                if break_point == -1:
                    break_point = current_text.rfind('. ', 0, max_chunk_size)
                if break_point == -1:
                    # No good break point - force split at max_chunk_size
                    break_point = max_chunk_size
                
                # Ensure we make progress (avoid infinite loop)
                if break_point <= 0:
                    break_point = max_chunk_size
                
                chunk_text = current_text[:break_point].strip()
                if chunk_text and len(chunk_text) >= min_chunk_size:
                    chunks.append({
                        "text": chunk_text,
                        "section": current_section,
                    })
                
                # Move past the break point - ensure we always advance
                remaining = current_text[break_point:].lstrip()
                if len(remaining) >= len(current_text):
                    # Safety: if we're not making progress, force advance
                    current_text = current_text[max_chunk_size:]
                else:
                    current_text = remaining
    
    # Don't forget the last chunk
    if current_text.strip() and len(current_text.strip()) >= min_chunk_size:
        chunks.append({
            "text": current_text.strip(),
            "section": current_section,
        })
    
    # Convert to WikipediaChunk objects
    return [
        WikipediaChunk(
            text=c["text"],
            section_context=f"{title} > {c['section']}",
            chunk_num=i,
            total_chunks=len(chunks),
            page_title=title,
        )
        for i, c in enumerate(chunks)
    ]


# =============================================================================
# Notebook-Based Caching with CID Provenance
# =============================================================================

def generate_sample_notebook_header(
    sampling_method: str,
    sample_size: int,
    source_url: str = "https://wikimedia.org/api/rest_v1/",
    alpha: float = 0.5,
) -> nbformat.NotebookNode:
    """Generate header for wikipedia_sample.ipynb.
    
    Args:
        sampling_method: Description of sampling method
        sample_size: Target number of pages
        source_url: API source URL
        alpha: Power-law exponent used
        
    Returns:
        NotebookNode with header cell
    """
    nb = new_notebook()
    
    header = f"""# Wikipedia Sample

## Provenance

```yaml
source: {source_url}
sampling_method: {sampling_method}
sample_size: {sample_size}
alpha: {alpha}
generated_by: ontological_engineer
generated_at: {datetime.now().isoformat()}
```

## Description

This notebook contains a sample of Wikipedia pages selected for training.
Each page entry includes title, view count, and a CID signature for provenance.
"""
    nb.cells.append(new_markdown_cell(header))
    
    return nb


def append_sample_page_cell(
    nb: nbformat.NotebookNode,
    page: WikipediaPage,
    index: int,
    total: int,
) -> Tuple[str, Dict]:
    """Append a page entry cell with CID signature.
    
    Args:
        nb: Notebook to append to
        page: WikipediaPage to add
        index: Page index in sample
        total: Total pages in sample
        
    Returns:
        Tuple of (page_cid, signature_dict)
    """
    content = json.dumps({
        "title": page.title,
        "views": page.views,
        "index": index,
        "total": total,
    }, indent=2)
    
    nb.cells.append(new_raw_cell(content))
    
    # Compute CID and signature
    page_cid = compute_cid(content)
    cell_num = len(nb.cells) - 1
    
    signature = make_signature(
        cell_num=cell_num,
        cell_type="sample_page",
        cid=page_cid,
        from_cid="",  # No source CID for sampled pages
        repro_class="InputData",
        label=f"sample:{page.title}",
    )
    nb.cells.append(new_raw_cell(json.dumps(signature, indent=2)))
    
    return page_cid, signature


def generate_chunks_notebook_header(
    page_title: str,
    source_url: str,
    model_config: Optional[Dict] = None,
) -> nbformat.NotebookNode:
    """Generate header for chunks.ipynb.
    
    Args:
        page_title: Wikipedia article title
        source_url: Wikipedia URL for the page
        model_config: Optional model configuration
        
    Returns:
        NotebookNode with header cells
    """
    nb = new_notebook()
    
    header = f"""# Chunks: {page_title}

## Provenance

```yaml
source_url: {source_url}
article_title: {page_title}
license: CC BY-SA 4.0
chunking_strategy: section-based with max_size=1500
generated_by: ontological_engineer
generated_at: {datetime.now().isoformat()}
```

## Description

This notebook contains text chunks from the Wikipedia article.
Each chunk preserves section context via breadcrumb metadata.
Each content cell is followed by a CID signature for provenance tracking.
"""
    nb.cells.append(new_markdown_cell(header))
    
    return nb


def append_chunk_cell(
    nb: nbformat.NotebookNode,
    chunk: WikipediaChunk,
    source_cid: Optional[str] = None,
) -> Tuple[str, Dict]:
    """Append a chunk cell with CID signature.
    
    Args:
        nb: Notebook to append to
        chunk: WikipediaChunk to add
        source_cid: Optional CID of the source (e.g., page content)
        
    Returns:
        Tuple of (chunk_cid, signature_dict)
    """
    content = f"""**Context:** {chunk.section_context}
**Chunk:** {chunk.chunk_num + 1} of {chunk.total_chunks}

---

{chunk.text}
"""
    nb.cells.append(new_markdown_cell(content))
    
    # Compute CID and signature
    chunk_cid = compute_cid(content)
    cell_num = len(nb.cells) - 1
    
    signature = make_signature(
        cell_num=cell_num,
        cell_type="chunk",
        cid=chunk_cid,
        from_cid=source_cid or "",
        repro_class="Data",
        label=f"chunk:{chunk.chunk_num}",
        chunk_num=chunk.chunk_num,
    )
    nb.cells.append(new_raw_cell(json.dumps(signature, indent=2)))
    
    return chunk_cid, signature


def load_sample_from_notebook(notebook_path: Path) -> List[WikipediaPage]:
    """Load Wikipedia sample from a notebook.
    
    Args:
        notebook_path: Path to wikipedia_sample.ipynb
        
    Returns:
        List of WikipediaPage objects
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    pages = []
    
    # Skip header cell, parse raw cells in pairs (content + signature)
    i = 1  # Skip header
    while i < len(nb.cells) - 1:
        cell = nb.cells[i]
        next_cell = nb.cells[i + 1]
        
        if cell.cell_type == 'raw' and next_cell.cell_type == 'raw':
            try:
                data = json.loads(cell.source)
                sig = json.loads(next_cell.source)
                
                if sig.get('_type') == 'sample_page' and 'title' in data:
                    pages.append(WikipediaPage(
                        title=data['title'],
                        views=data.get('views', 0),
                    ))
            except (json.JSONDecodeError, KeyError):
                pass
        
        i += 2
    
    return pages


def load_chunks_from_notebook(notebook_path: Path) -> List[Dict[str, Any]]:
    """Load chunks from a chunks.ipynb file.
    
    Args:
        notebook_path: Path to chunks.ipynb
        
    Returns:
        List of dicts with keys: chunk_num, section_context, text, cid
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    results = []
    
    # Skip header cell
    i = 1
    while i < len(nb.cells) - 1:
        cell = nb.cells[i]
        next_cell = nb.cells[i + 1]
        
        if cell.cell_type == 'markdown' and next_cell.cell_type == 'raw':
            try:
                sig = json.loads(next_cell.source)
                if sig.get('_type') == 'chunk':
                    # Parse markdown content
                    content = cell.source
                    lines = content.split('\n')
                    
                    section_context = ""
                    chunk_num = 0
                    text = ""
                    
                    in_text = False
                    for line in lines:
                        if line.startswith('**Context:**'):
                            section_context = line.replace('**Context:**', '').strip()
                        elif line.startswith('**Chunk:**'):
                            chunk_part = line.replace('**Chunk:**', '').strip()
                            chunk_num = int(chunk_part.split(' of ')[0]) - 1  # Convert to 0-indexed
                        elif line == '---':
                            in_text = True
                        elif in_text:
                            text += line + '\n'
                    
                    results.append({
                        'chunk_num': chunk_num,
                        'section_context': section_context,
                        'text': text.strip(),
                        'cid': sig.get('dcterms:identifier', ''),
                    })
            except (json.JSONDecodeError, KeyError):
                pass
        
        i += 2
    
    return results


def get_processed_page_titles(notebook_path: Path) -> set:
    """Get titles of pages already in a sample notebook.
    
    Args:
        notebook_path: Path to wikipedia_sample.ipynb
        
    Returns:
        Set of page titles already processed
    """
    if not notebook_path.exists():
        return set()
    
    pages = load_sample_from_notebook(notebook_path)
    return {p.title for p in pages}


def save_notebook(nb: nbformat.NotebookNode, output_path: Path) -> None:
    """Save a notebook to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)


# =============================================================================
# Convenience Functions for Training Pipeline
# =============================================================================

def fetch_and_cache_pages(
    pages: List[WikipediaPage],
    output_dir: Path,
    max_pages: Optional[int] = None,
) -> Tuple[nbformat.NotebookNode, List[WikipediaChunk]]:
    """Fetch page content and cache as chunks notebook.
    
    Args:
        pages: List of pages to fetch
        output_dir: Directory to save chunks notebooks
        max_pages: Maximum pages to process (None = all)
        
    Returns:
        Tuple of (sample_notebook, all_chunks)
    """
    from tqdm import tqdm
    
    all_chunks = []
    pages_to_process = pages[:max_pages] if max_pages else pages
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for page in tqdm(pages_to_process, desc="Fetching pages"):
        content = fetch_page_content(page.title)
        if not content:
            continue
        
        page.content = content
        page.fetched_at = datetime.now().isoformat()
        
        chunks = chunk_article(page.title, content)
        if not chunks:
            continue
        
        # Generate chunks notebook for this page
        page_slug = page.title.lower().replace(' ', '_').replace('/', '_')
        chunks_nb = generate_chunks_notebook_header(
            page_title=page.title,
            source_url=f"https://en.wikipedia.org/wiki/{quote(page.title.replace(' ', '_'))}",
        )
        
        # Compute a CID for the full page content
        page_cid = compute_cid(content)
        
        for chunk in chunks:
            chunk_cid, _ = append_chunk_cell(chunks_nb, chunk, source_cid=page_cid)
        
        # Save the chunks notebook
        chunks_path = output_dir / f"{page_slug}_chunks.ipynb"
        save_notebook(chunks_nb, chunks_path)
        
        all_chunks.extend(chunks)
    
    return all_chunks


def process_wikipedia_sample(
    pages: List[WikipediaPage],
    output_dir: Path,
    max_pages: Optional[int] = None,
    min_chunk_length: int = 60,
) -> Tuple[List[WikipediaChunk], int]:
    """Process Wikipedia pages: fetch, chunk, and save with CID provenance.
    
    This is the main entry point for processing a Wikipedia sample.
    Handles incremental processing - skips pages that already have chunks.
    
    Args:
        pages: List of WikipediaPage objects to process
        output_dir: Directory to save per-page chunks notebooks
        max_pages: Maximum pages to process (default: all)
        min_chunk_length: Minimum chunk text length to keep
        
    Returns:
        Tuple of (list of all WikipediaChunks, pages_processed count)
    """
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_chunks = []
    pages_processed = 0
    pages_to_process = pages[:max_pages] if max_pages else pages
    
    for page in tqdm(pages_to_process, desc="Processing pages"):
        # Check if already processed
        page_slug = page.title.lower().replace(' ', '_').replace('/', '_')
        chunks_path = output_dir / f"{page_slug}_chunks.ipynb"
        
        if chunks_path.exists():
            # Load from existing notebook (skip re-fetching)
            existing_chunks = load_chunks_from_notebook(chunks_path)
            for c in existing_chunks:
                training_chunks.append(WikipediaChunk(
                    text=c['text'],
                    section_context=c['section_context'],
                    chunk_num=c['chunk_num'],
                    total_chunks=len(existing_chunks),
                    page_title=page.title,
                ))
            pages_processed += 1
            continue
        
        # Fetch and chunk
        content = fetch_page_content(page.title)
        if not content:
            continue
        
        chunks = chunk_article(page.title, content)
        chunks = [c for c in chunks if len(c.text) >= min_chunk_length]
        if not chunks:
            continue
        
        # Create chunks notebook with CID provenance
        nb = generate_chunks_notebook_header(
            page_title=page.title,
            source_url=f"https://en.wikipedia.org/wiki/{quote(page.title.replace(' ', '_'))}",
        )
        
        # Compute a CID for the full page content (source provenance)
        page_cid = compute_cid(content)
        
        # Add each chunk with CID signature
        for chunk in chunks:
            chunk_cid, _ = append_chunk_cell(nb, chunk, source_cid=page_cid)
        
        # Save the chunks notebook
        save_notebook(nb, chunks_path)
        
        training_chunks.extend(chunks)
        pages_processed += 1
    
    return training_chunks, pages_processed
