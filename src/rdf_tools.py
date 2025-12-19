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
        
        TIPS FOR BETTER QUERIES:
        - Include the nature of the thing: "a quote someone said" not just "quote"
        - Include context: "a written work that is a quotation attributed to a person"
        - For events: "an event where something happened"
        - For creative works: "a creative work such as a book, article, or quote"
        
        Example queries:
        - "a person who does scientific research" → schema:Person, schema:Researcher
        - "a spoken or written quotation attributed to someone" → schema:Quotation
        - "an organization that employs people" → schema:Organization
        - "an award or honor given to someone" → schema:Award
        
        Args:
            description: Natural language description of the entity type
        
        Returns:
            Top matching schema.org classes with URIs, descriptions, and usage hints.
            Scores range 0-1. Only use matches with score >= 0.5.
            If no good match (all scores < 0.5), use schema:Thing or skip the type assertion.
        """
        results = schema_matcher.find_class(description, top_k=5)
        if not results:
            return "No matches found. Use schema:Thing as a fallback or skip the type assertion."
        
        # Filter to decent matches
        good_matches = [r for r in results if r['score'] >= 0.5]
        weak_matches = [r for r in results if r['score'] < 0.5]
        
        lines = []
        if good_matches:
            lines.append("Good matches (score >= 0.5):")
            for r in good_matches:
                lines.append(f"\n  {r['prefix']} ({r['score']:.2f})")
                lines.append(f"    URI: {r['uri']}")
                if r['description']:
                    # Show more of the description - it contains usage guidance
                    lines.append(f"    Description: {r['description'][:200]}")
                if r.get('usage_hint'):
                    lines.append(f"    HOW TO USE: {r['usage_hint']}")
                if r.get('parent'):
                    lines.append(f"    Parent class: schema:{r['parent']}")
        else:
            lines.append("WARNING: No good matches found (all scores < 0.5).")
            lines.append("Consider using schema:Thing or skipping this type assertion.")
        
        if weak_matches and not good_matches:
            lines.append("\nWeak matches (not recommended):")
            for r in weak_matches[:2]:
                lines.append(f"  {r['prefix']} ({r['score']:.2f}) - LOW CONFIDENCE")
        
        return "\n".join(lines)

    @tool
    def find_rdf_property(description: str, subject_type: str = "", object_type: str = "") -> str:
        """Find the best schema.org property/predicate for a relationship.
        
        Use this when you need to find the right predicate for a triple.
        
        TIPS FOR BETTER QUERIES:
        - Describe the relationship, not just a word: "a quote that someone said" not "said"
        - Include what's being related: "the text content of a quotation"
        - For attribution: "who authored or spoke something"
        - Always include subject_type if known for better matches
        
        Example queries:
        - "the text content of a quotation or creative work" → schema:text
        - "the person who created or authored something" → schema:author, schema:creator
        - "the person who said a quote" → schema:spokenByCharacter
        - "the date when someone was born" (subject_type="Person") → schema:birthDate
        
        Args:
            description: Natural language description of the relationship
            subject_type: Optional - the type of the subject (e.g., "Person", "Organization", "Quotation")  
            object_type: Optional - the type of the object/value (e.g., "Date", "Place", "Text")
        
        Returns:
            Top matching schema.org properties with URIs, domains, ranges, and usage hints.
            Scores range 0-1. Only use matches with score >= 0.5.
            If no good match, DO NOT emit a triple - skip the statement or use rdfs:comment.
        """
        results = schema_matcher.find_property(
            description, 
            subject_type=subject_type or None,
            object_type=object_type or None,
            top_k=5
        )
        if not results:
            return "No matches found. DO NOT emit a made-up triple. Skip this statement or use rdfs:comment to preserve the information as text."
        
        # Filter to decent matches
        good_matches = [r for r in results if r['score'] >= 0.5]
        weak_matches = [r for r in results if r['score'] < 0.5]
        
        lines = []
        if good_matches:
            lines.append("Good matches (score >= 0.5):")
            for r in good_matches:
                lines.append(f"\n  {r['prefix']} ({r['score']:.2f})")
                lines.append(f"    URI: {r['uri']}")
                if r['domain']:
                    lines.append(f"    Domain (subject type): {r['domain']}")
                if r['range']:
                    lines.append(f"    Range (object type): {r['range']}")
                if r['description']:
                    # Show more description - it contains important usage guidance
                    lines.append(f"    Description: {r['description'][:150]}")
                if r.get('usage_hint'):
                    lines.append(f"    HOW TO USE: {r['usage_hint']}")
        else:
            lines.append("WARNING: No good matches found (all scores < 0.5).")
            lines.append("DO NOT emit a random/unrelated triple.")
            lines.append("Options: 1) Skip this statement, 2) Use rdfs:comment to store as text")
        
        if weak_matches and not good_matches:
            lines.append("\nWeak matches (DO NOT USE):")
            for r in weak_matches[:2]:
                lines.append(f"  {r['prefix']} ({r['score']:.2f}) - TOO LOW, DO NOT USE")
        
        return "\n".join(lines)

    def normalize_statement_id(sid) -> str:
        """Normalize statement ID - extract the full ID, be permissive with types."""
        if sid is None:
            return "unknown"
        # Convert to string first (handles float like 3.4 -> "3.4")
        sid_str = str(sid).strip()
        # Remove common wrappers: brackets, quotes, whitespace
        sid_str = sid_str.strip('[]()"\' ')
        # Keep the full ID including dots (e.g., "3.1", "3.4")
        return sid_str if sid_str else "unknown"
    
    @tool
    def emit_triple(statement_id, subject: str, predicate: str, object_value: str) -> str:
        """Emit a single RDF triple. Use this to output each triple you generate.
        
        Args:
            statement_id: The ID of the statement this triple comes from (e.g., "3.1", "3.2"). Can be string or number.
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
