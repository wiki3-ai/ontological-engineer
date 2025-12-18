"""General utilities for the pipeline."""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime


def log_progress(msg: str, end="\n"):
    """Print with immediate flush for real-time progress."""
    print(msg, end=end, flush=True)


@dataclass
class ContextualChunk:
    """A text chunk with contextual metadata."""
    content: str
    chunk_index: int
    total_chunks: int
    breadcrumb: str
    section_title: str
    char_start: int
    char_end: int


def setup_output_directory(
    output_dir: str,
    article_title: str,
    continue_from_run: str = None
) -> tuple[str, str]:
    """Set up the output directory structure.
    
    Returns:
        tuple of (run_output_dir, run_timestamp)
    """
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    article_slug = article_title.lower().replace(' ', '_')
    run_output_dir = os.path.join(output_dir, article_slug, run_timestamp)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Copy files from previous run if specified
    if continue_from_run:
        if os.path.isdir(continue_from_run):
            print(f"Continuing from previous run: {continue_from_run}")
            copied_files = []
            for filename in os.listdir(continue_from_run):
                src = os.path.join(continue_from_run, filename)
                dst = os.path.join(run_output_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    copied_files.append(filename)
            print(f"  Copied {len(copied_files)} files: {', '.join(copied_files)}")
            print(f"  Existing cells with matching CIDs will be skipped")
        else:
            print(f"WARNING: Previous run not found: {continue_from_run}")
            print(f"  Starting fresh run instead")
    
    return run_output_dir, run_timestamp


def create_contextual_chunks(
    raw_chunks: list[str],
    raw_content: str,
    sections: list[dict],
    article_title: str,
    get_section_context_fn
) -> list[ContextualChunk]:
    """Add context metadata to raw text chunks."""
    contextual_chunks = []
    current_pos = 0
    
    for i, chunk_text in enumerate(raw_chunks):
        # Find position in original content
        chunk_start = raw_content.find(chunk_text, current_pos)
        if chunk_start == -1:
            chunk_start = current_pos  # Fallback
        chunk_end = chunk_start + len(chunk_text)
        
        # Get section context
        context = get_section_context_fn(chunk_start, sections, article_title)
        
        contextual_chunks.append(ContextualChunk(
            content=chunk_text,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            breadcrumb=context["breadcrumb"],
            section_title=context["section_title"],
            char_start=chunk_start,
            char_end=chunk_end,
        ))
        
        current_pos = chunk_start + 1
    
    return contextual_chunks
