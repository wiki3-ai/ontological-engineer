"""Content ID (CID) utilities for provenance tracking.

Uses IPFS CIDs (Content Identifiers) computed via multiformats library.
Provenance follows the REPRODUCE-ME ontology (extension of PROV-O).

CID Format: CIDv1 with 'raw' codec and SHA2-256 hash.
URI Format: ipfs://<CID>

Signatures are JSON-LD documents using PROV-O/REPRODUCE-ME vocabulary.

References:
- REPRODUCE-ME: https://w3id.org/reproduceme
- IPFS CID: https://docs.ipfs.tech/concepts/content-addressing/
- PROV-O: https://www.w3.org/TR/prov-o/
"""

import hashlib
import json
from typing import Optional, Union

from multiformats import CID

# Namespace prefixes for REPRODUCE-ME provenance
REPRO_NS = "https://w3id.org/reproduceme#"
PROV_NS = "http://www.w3.org/ns/prov#"
PPLAN_NS = "http://purl.org/net/p-plan#"
DCTERMS_NS = "http://purl.org/dc/terms/"

# JSON-LD context for signatures
JSONLD_CONTEXT = {
    "prov": PROV_NS,
    "repro": REPRO_NS,
    "dcterms": DCTERMS_NS,
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "prov:wasDerivedFrom": {"@type": "@id"},
    "prov:wasGeneratedBy": {"@type": "@id"},
}


def compute_cid(content: Union[str, bytes]) -> str:
    """Compute IPFS CID for content.
    
    Args:
        content: String or bytes to hash.
        
    Returns:
        CIDv1 string (base32 encoded, e.g., 'bafkreig...')
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    sha256_digest = hashlib.sha256(content).digest()
    # Create CIDv1 with raw codec, base32 encoding, sha2-256 hash
    # The digest is passed as a tuple of (multihash_codec, raw_digest)
    cid = CID("base32", 1, "raw", ("sha2-256", sha256_digest))
    return str(cid)


def cid_to_uri(cid: str) -> str:
    """Convert a CID string to an ipfs:// URI."""
    return f"ipfs://{cid}"


def uri_to_cid(uri: str) -> str:
    """Extract CID from an ipfs:// URI."""
    if uri.startswith("ipfs://"):
        return uri[7:]
    return uri


def make_signature(
    cell_num: int,
    cell_type: str,
    cid: str,
    from_cid: str,
    repro_class: str = "Data",
    label: str = None,
    stmt_key: str = None,
    chunk_num: int = None,
    stmt_idx: int = None,
) -> dict:
    """Create a JSON-LD signature for a generated cell using PROV-O/REPRODUCE-ME vocabulary.
    
    Args:
        cell_num: Cell number in the notebook.
        cell_type: Type of content (e.g., 'source', 'chunk', 'fact', 'rdf').
        cid: IPFS CID of this cell's content.
        from_cid: IPFS CID of the source content this was derived from.
        repro_class: REPRODUCE-ME class name (Data, InputData, OutputData, etc.)
        label: Human-readable label for the entity.
        stmt_key: Optional statement key for RDF cells (e.g., "3_2").
        chunk_num: Optional chunk number for RDF cells.
        stmt_idx: Optional statement index for RDF cells.
        
    Returns:
        JSON-LD signature dict with PROV-O provenance.
    """
    # Ensure repro_class has the repro: prefix
    if repro_class and not repro_class.startswith("repro:"):
        repro_class = f"repro:{repro_class}"
    
    sig = {
        "@context": JSONLD_CONTEXT,
        "@id": cid_to_uri(cid),
        "@type": ["prov:Entity", repro_class],
        "dcterms:identifier": cid,
        "prov:label": label or f"{cell_type}:{cell_num}",
        # Pipeline metadata (not PROV-O, but useful for processing)
        "_cell": cell_num,
        "_type": cell_type,
    }
    
    # Add prov:wasDerivedFrom link
    if from_cid:
        sig["prov:wasDerivedFrom"] = {"@id": cid_to_uri(from_cid)}
    
    # Add RDF-specific metadata
    if stmt_key:
        sig["_stmt_key"] = stmt_key
    if chunk_num is not None:
        sig["_chunk_num"] = chunk_num
    if stmt_idx is not None:
        sig["_stmt_idx"] = stmt_idx
    
    return sig


def parse_signature(raw_content: str) -> Optional[dict]:
    """Parse a JSON-LD signature from raw cell content.
    
    Supports both new JSON-LD format and legacy format for backwards compatibility.
    Returns a normalized dict with both JSON-LD properties and convenience accessors.
    """
    try:
        data = json.loads(raw_content.strip())
        
        # Check for JSON-LD format (has @id)
        if "@id" in data:
            # Extract CID from @id (ipfs:// URI)
            cid = uri_to_cid(data["@id"])
            
            # Extract from_cid from prov:wasDerivedFrom
            from_cid = None
            derived_from = data.get("prov:wasDerivedFrom")
            if derived_from:
                if isinstance(derived_from, dict):
                    from_cid = uri_to_cid(derived_from.get("@id", ""))
                elif isinstance(derived_from, str):
                    from_cid = uri_to_cid(derived_from)
            
            # Return normalized structure with both JSON-LD and legacy accessors
            return {
                # JSON-LD properties (preserved)
                "@context": data.get("@context"),
                "@id": data["@id"],
                "@type": data.get("@type"),
                "dcterms:identifier": data.get("dcterms:identifier", cid),
                "prov:label": data.get("prov:label"),
                "prov:wasDerivedFrom": data.get("prov:wasDerivedFrom"),
                # Legacy accessors (for backwards compatibility)
                "cell": data.get("_cell"),
                "type": data.get("_type"),
                "cid": cid,
                "from_cid": from_cid,
                "label": data.get("prov:label"),
                "stmt_key": data.get("_stmt_key"),
                "chunk_num": data.get("_chunk_num"),
                "stmt_idx": data.get("_stmt_idx"),
            }
        
        # Legacy format support (has "cell" and "cid")
        if all(k in data for k in ("cell", "cid")):
            # Add legacy accessors if missing
            data.setdefault("type", data.get("_type", "unknown"))
            data.setdefault("from_cid", None)
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


def generate_provenance_ttl(signatures: list[dict], notebook_label: str = None) -> str:
    """Generate REPRODUCE-ME compliant provenance in Turtle format from JSON-LD signatures.
    
    Args:
        signatures: List of JSON-LD signature dicts from the pipeline.
        notebook_label: Optional label for the notebook entity.
        
    Returns:
        Turtle-formatted provenance triples.
    """
    lines = [
        "# Provenance metadata (REPRODUCE-ME / PROV-O)",
        "@prefix prov: <http://www.w3.org/ns/prov#> .",
        "@prefix repro: <https://w3id.org/reproduceme#> .",
        "@prefix pplan: <http://purl.org/net/p-plan#> .",
        "@prefix dcterms: <http://purl.org/dc/terms/> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "",
    ]
    
    for sig in signatures:
        # Get URI - support both JSON-LD (@id) and legacy (cid) formats
        uri = sig.get("@id") or cid_to_uri(sig.get("cid", ""))
        if not uri:
            continue
            
        # Get types - support both JSON-LD (@type) and legacy (type/repro_class) formats
        types = sig.get("@type")
        if types:
            if isinstance(types, list):
                type_str = ", ".join(types)
            else:
                type_str = types
        else:
            cell_type = sig.get("type", "data")
            type_to_class = {
                "source": "repro:InputData",
                "chunk": "repro:Data",
                "fact": "repro:Data",
                "rdf": "repro:OutputData",
            }
            repro_class = type_to_class.get(cell_type, "repro:Data")
            type_str = f"prov:Entity, {repro_class}"
        
        # Get identifier
        identifier = sig.get("dcterms:identifier") or sig.get("cid", uri_to_cid(uri))
        
        # Get label
        label = sig.get("prov:label") or sig.get("label", f"entity:{identifier[:12]}")
        
        # Get derivation source
        derived_from = sig.get("prov:wasDerivedFrom")
        from_uri = None
        if derived_from:
            if isinstance(derived_from, dict):
                from_uri = derived_from.get("@id")
            elif isinstance(derived_from, str):
                from_uri = derived_from
        elif sig.get("from_cid"):
            from_uri = cid_to_uri(sig["from_cid"])
        
        lines.append(f"<{uri}>")
        lines.append(f'    a {type_str} ;')
        lines.append(f'    dcterms:identifier "{identifier}" ;')
        lines.append(f'    prov:label "{label}" ;')
        
        if from_uri:
            lines.append(f'    prov:wasDerivedFrom <{from_uri}> ;')
        
        # Remove trailing semicolon from last property
        if lines[-1].endswith(" ;"):
            lines[-1] = lines[-1][:-2] + " ."
        else:
            lines.append("    .")
        lines.append("")
    
    return "\n".join(lines)


def collect_pipeline_signatures(output_dir: str) -> list[dict]:
    """Collect all signatures from the pipeline notebooks.
    
    Args:
        output_dir: Directory containing source.ipynb, chunks.ipynb, facts.ipynb, rdf.ipynb
        
    Returns:
        List of signature dicts for provenance generation.
    """
    import os
    import nbformat
    
    signatures = []
    notebook_files = ['source.ipynb', 'chunks.ipynb', 'facts.ipynb', 'rdf.ipynb']
    
    for nb_file in notebook_files:
        nb_path = os.path.join(output_dir, nb_file)
        if not os.path.exists(nb_path):
            continue
        
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == 'raw':
                sig = parse_signature(cell.source)
                if sig:
                    signatures.append(sig)
    
    return signatures
