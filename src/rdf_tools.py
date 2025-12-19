"""RDF generation tools for LLM tool calling."""

import re
from typing import List
from langchain_core.tools import tool


def create_rdf_tools(schema_matcher):
    """Create RDF tools bound to a schema matcher instance.
    
    Returns a tuple of (tools_list, emitted_triples_collector, reset_function)
    """
    # Collector for emitted triples
    emitted_triples: List[dict] = []
    
    def reset_triples():
        """Reset the triples collector."""
        emitted_triples.clear()
    
    def get_triples() -> List[dict]:
        """Get a copy of collected triples."""
        return list(emitted_triples)
    
    @tool
    def find_rdf_class(description: str) -> str:
        """Find the best schema.org class/type for an entity based on a natural language description.
        
        Use this when you need to determine the rdf:type of an entity.
        Example: find_rdf_class("a person who does scientific research")
        
        Args:
            description: Natural language description of the entity type
        
        Returns:
            Top matching schema.org classes with URIs and descriptions
        """
        results = schema_matcher.find_class(description, top_k=5)
        if not results:
            return "No matches found. Use a generic type like schema:Thing"
        
        lines = ["Top matching classes:"]
        for r in results:
            lines.append(f"  {r['prefix']} ({r['score']:.2f})")
            lines.append(f"    URI: {r['uri']}")
            if r['description']:
                lines.append(f"    Description: {r['description'][:100]}")
        return "\n".join(lines)

    @tool
    def find_rdf_property(description: str, subject_type: str = "", object_type: str = "") -> str:
        """Find the best schema.org property/predicate for a relationship.
        
        Use this when you need to find the right predicate for a triple.
        Example: find_rdf_property("the date when someone was born", subject_type="Person")
        
        Args:
            description: Natural language description of the relationship
            subject_type: Optional - the type of the subject (e.g., "Person", "Organization")  
            object_type: Optional - the type of the object/value (e.g., "Date", "Place")
        
        Returns:
            Top matching schema.org properties with URIs, domains, and ranges
        """
        results = schema_matcher.find_property(
            description, 
            subject_type=subject_type or None,
            object_type=object_type or None,
            top_k=5
        )
        if not results:
            return "No matches found. Consider using rdfs:label or a descriptive URI fragment."
        
        lines = ["Top matching properties:"]
        for r in results:
            lines.append(f"  {r['prefix']} ({r['score']:.2f})")
            lines.append(f"    URI: {r['uri']}")
            if r['domain']:
                lines.append(f"    Domain: {r['domain']}")
            if r['range']:
                lines.append(f"    Range: {r['range']}")
            if r['description']:
                lines.append(f"    Description: {r['description'][:80]}")
        return "\n".join(lines)

    def normalize_statement_id(sid) -> str:
        """Normalize statement ID - extract just the number, be permissive."""
        if sid is None:
            return "unknown"
        sid_str = str(sid).strip()
        # Remove common wrappers: brackets, quotes, whitespace
        sid_str = sid_str.strip('[]()"\' ')
        # Extract leading digits if present
        match = re.match(r'^(\d+)', sid_str)
        if match:
            return match.group(1)
        return sid_str if sid_str else "unknown"
    
    @tool
    def emit_triple(statement_id: str, subject: str, predicate: str, object_value: str) -> str:
        """Emit a single RDF triple. Use this to output each triple you generate.
        
        Args:
            statement_id: The ID of the statement this triple comes from (e.g., "1", "2", "3")
            subject: The subject URI (e.g., "<https://example.org#entity>") or prefixed (e.g., "schema:Person")
            predicate: The predicate URI or prefixed term (e.g., "schema:birthDate", "rdf:type")
            object_value: The object - a URI, prefixed term, or literal with datatype 
                          (e.g., '"1879-03-14"^^xsd:date', '"Albert Einstein"@en', '<https://...>')
        
        Returns:
            Confirmation message or error with guidance
        """
        # Validate and provide helpful feedback
        issues = []
        if not statement_id:
            issues.append("statement_id is required (e.g., '1', '2')")
        if not subject:
            issues.append("subject is required (e.g., '<#entity_id>' or 'schema:Person')")
        if not predicate:
            issues.append("predicate is required (e.g., 'schema:birthDate' or 'rdf:type')")
        if not object_value:
            issues.append("object_value is required (e.g., '\"value\"' or '<#uri>')")
        
        if issues:
            return f"INVALID - please fix and retry: {'; '.join(issues)}"
        
        norm_id = normalize_statement_id(statement_id)
        emitted_triples.append({
            "statement_id": norm_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
        })
        return f"OK: recorded triple for statement {norm_id}"

    @tool  
    def emit_triples(triples: List[dict]) -> str:
        """Emit multiple RDF triples at once. More efficient than calling emit_triple repeatedly.
        
        Args:
            triples: List of triple dictionaries, each with keys:
                     - statement_id: The ID of the statement this triple comes from (e.g., "1", "2")
                     - subject: The subject URI or prefixed term
                     - predicate: The predicate URI or prefixed term  
                     - object: The object URI, prefixed term, or literal
                     
        Example:
            emit_triples([
                {"statement_id": "1", "subject": "<#person_einstein>", "predicate": "rdf:type", "object": "schema:Person"},
                {"statement_id": "1", "subject": "<#person_einstein>", "predicate": "schema:name", "object": '"Albert Einstein"'},
                {"statement_id": "2", "subject": "<#person_einstein>", "predicate": "schema:birthDate", "object": '"1879-03-14"^^xsd:date'}
            ])
        
        Returns:
            Confirmation with count of triples recorded
        """
        count = 0
        issues = []
        for i, t in enumerate(triples):
            if not isinstance(t, dict):
                issues.append(f"item {i}: expected dict, got {type(t).__name__}")
                continue
            
            # Be flexible with key names
            stmt_id = t.get("statement_id") or t.get("statementId") or t.get("id") or t.get("stmt_id")
            subject = t.get("subject") or t.get("s")
            predicate = t.get("predicate") or t.get("p") or t.get("property")
            obj = t.get("object") or t.get("o") or t.get("value") or t.get("object_value")
            
            missing = []
            if not subject:
                missing.append("subject")
            if not predicate:
                missing.append("predicate")
            if not obj:
                missing.append("object")
            
            if missing:
                issues.append(f"item {i}: missing {missing}, got keys: {list(t.keys())}")
                continue
                
            emitted_triples.append({
                "statement_id": normalize_statement_id(stmt_id),
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            })
            count += 1
        
        if count == 0 and issues:
            return f"ERROR: 0 triples recorded. Issues: {'; '.join(issues[:5])}. Required keys: statement_id, subject, predicate, object"
        elif issues:
            return f"PARTIAL: recorded {count} triples, skipped {len(issues)}: {'; '.join(issues[:3])}{'...' if len(issues) > 3 else ''}"
        return f"OK: recorded {count} triples"
    
    tools = [find_rdf_class, find_rdf_property, emit_triple, emit_triples]
    
    return tools, get_triples, reset_triples


def parse_statements(facts_text: str) -> List[str]:
    """Parse bulleted statements from facts text."""
    statements = []
    for line in facts_text.split('\n'):
        line = line.strip()
        # Match bullet points: -, *, •, or numbered (1., 2., etc.)
        if line.startswith(('-', '*', '•')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
            # Remove bullet/number prefix
            if line[0].isdigit():
                line = line.split(' ', 1)[1] if ' ' in line else line[2:]
            else:
                line = line[1:].strip()
            if line:
                statements.append(line)
    return statements


def triples_to_turtle(triples: List[dict]) -> str:
    """Convert collected triples to Turtle format."""
    if not triples:
        return "# No triples emitted"
    
    lines = []
    for t in triples:
        lines.append(f"{t['subject']} {t['predicate']} {t['object']} .")
    return "\n".join(lines)
