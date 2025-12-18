"""Content ID (CID) utilities for provenance tracking."""

import hashlib
import json
from typing import Optional


def compute_cid(content: str) -> str:
    """Compute SHA256 content ID for a string."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def make_signature(cell_num: int, cell_type: str, cid: str, from_cid: str) -> dict:
    """Create a signature dict for a generated cell."""
    return {
        "cell": cell_num,
        "type": cell_type,
        "cid": cid,
        "from_cid": from_cid,
    }


def parse_signature(raw_content: str) -> Optional[dict]:
    """Parse a signature from raw cell content. Returns None if not a valid signature."""
    try:
        data = json.loads(raw_content.strip())
        if all(k in data for k in ("cell", "type", "cid", "from_cid")):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def extract_signatures(notebook) -> dict:
    """Extract all signatures from a notebook, keyed by cell number."""
    signatures = {}
    for cell in notebook.cells:
        if cell.cell_type == 'raw':
            sig = parse_signature(cell.source)
            if sig:
                signatures[sig["cell"]] = sig
    return signatures


def extract_statement_signatures(notebook) -> dict:
    """Extract signatures keyed by 'chunk_stmt' format for statement-level tracking."""
    signatures = {}
    for cell in notebook.cells:
        if cell.cell_type == 'raw':
            sig = parse_signature(cell.source)
            if sig:
                # Support both old (cell number) and new (chunk_stmt) keys
                key = sig.get("stmt_key", sig["cell"])
                signatures[key] = sig
    return signatures
