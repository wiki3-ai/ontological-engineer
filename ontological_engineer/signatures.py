"""
DSPy Signatures for Ontological Engineer.

This module defines all the DSPy signatures used in the pipeline:
- Statement extraction from Wikipedia chunks
- Quality judgment for statements and triples
- Schema term selection
- RDF generation
"""

from typing import List
import dspy


# =============================================================================
# Stage 1: Statement Extraction
# =============================================================================

class ExtractStatements(dspy.Signature):
    """Extract atomic, verifiable statements from Wikipedia text.
    
    Each statement must:
    - Be self-contained (understandable without the original text)
    - Preserve markdown links: [Entity Name](/wiki/Entity_Name)
    - Contain exactly one verifiable claim
    - Not editorialize or interpret beyond what's stated
    
    Example input chunk:
        "Albert Einstein was born in Ulm, in the Kingdom of Württemberg 
        in the German Empire, on 14 March 1879."
        
    Example output statements:
        - "[Albert Einstein](/wiki/Albert_Einstein) was born in [Ulm](/wiki/Ulm)."
        - "[Albert Einstein](/wiki/Albert_Einstein) was born on 14 March 1879."
        - "[Ulm](/wiki/Ulm) was in the [Kingdom of Württemberg](/wiki/Kingdom_of_Württemberg)."
    """
    
    chunk_text: str = dspy.InputField(
        desc="Wikipedia article chunk with markdown links preserved"
    )
    section_context: str = dspy.InputField(
        desc="Breadcrumb showing location: Article > Section > Subsection"
    )
    
    statements: List[str] = dspy.OutputField(
        desc="List of atomic statements, each preserving [Entity](/wiki/...) links"
    )


class JudgeStatementQuality(dspy.Signature):
    """Judge the quality of extracted statements from a Wikipedia chunk.
    
    Evaluate on four dimensions:
    1. Completeness: Are all key facts from the chunk captured?
    2. Atomicity: Is each statement truly atomic (one verifiable claim)?
    3. Accuracy: Do statements faithfully represent the source without adding info?
    4. Link preservation: Are [Entity](/wiki/...) links preserved correctly?
    """
    
    chunk_text: str = dspy.InputField(
        desc="Original Wikipedia chunk"
    )
    section_context: str = dspy.InputField(
        desc="Section breadcrumb for context"
    )
    statements: List[str] = dspy.InputField(
        desc="Extracted statements to evaluate"
    )
    
    completeness: float = dspy.OutputField(
        desc="Score 0-1: Are all key facts from the chunk captured?"
    )
    atomicity: float = dspy.OutputField(
        desc="Score 0-1: Is each statement truly atomic (one claim)?"
    )
    accuracy: float = dspy.OutputField(
        desc="Score 0-1: Do statements faithfully represent the source?"
    )
    link_preservation: float = dspy.OutputField(
        desc="Score 0-1: Are [Entity](/wiki/...) links preserved correctly?"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the scores given"
    )


# =============================================================================
# Stage 2: Schema Context
# =============================================================================

class SelectSchemaTerms(dspy.Signature):
    """Select relevant schema terms for converting statements to RDF.
    
    Given a set of statements and candidate schema terms (classes and properties),
    select which ones are most appropriate for representing the statements as RDF.
    Also identify if custom annotations are needed (temporal, reification, etc.).
    """
    
    statements: List[str] = dspy.InputField(
        desc="Statements to be converted to RDF"
    )
    candidate_classes: str = dspy.InputField(
        desc="Available classes with URIs and descriptions (JSON format)"
    )
    candidate_properties: str = dspy.InputField(
        desc="Available properties with URIs, domains, ranges (JSON format)"
    )
    
    selected_classes: List[str] = dspy.OutputField(
        desc="URIs of classes needed for these statements"
    )
    selected_properties: List[str] = dspy.OutputField(
        desc="URIs of properties needed for these statements"
    )
    custom_annotation_needs: str = dspy.OutputField(
        desc="Description of any custom annotations needed (temporal, roles, etc.)"
    )


# =============================================================================
# Stage 3: RDF Generation
# =============================================================================

class GenerateRDF(dspy.Signature):
    """Generate RDF triples from statements using provided schema context.
    
    Rules:
    - Convert [Entity](/wiki/Entity) links to <https://en.wikipedia.org/wiki/Entity>
    - Use provided prefixes and property definitions
    - Apply temporal/reification patterns where dates are involved
    - Output valid Turtle syntax
    - Use entity_registry for known entity URIs
    
    Example statement:
        "[Albert Einstein](/wiki/Albert_Einstein) was born on 14 March 1879."
        
    Example output:
        wiki:Albert_Einstein schema:birthDate "1879-03-14"^^xsd:date .
    """
    
    statements: List[str] = dspy.InputField(
        desc="Statements to convert to RDF"
    )
    schema_context: str = dspy.InputField(
        desc="Prefixes, class definitions, property definitions, and usage examples"
    )
    entity_registry: str = dspy.InputField(
        desc="JSON mapping of entity names to their URIs"
    )
    
    turtle_triples: str = dspy.OutputField(
        desc="Valid Turtle RDF representing the statements"
    )


class JudgeTripleQuality(dspy.Signature):
    """Judge the quality of generated RDF triples.
    
    Evaluate:
    1. Syntax validity: Is the Turtle syntax correct?
    2. URI correctness: Are URIs properly formed?
    3. Schema conformance: Are appropriate schema terms used?
    4. Completeness: Are all statements represented in the RDF?
    """
    
    statements: List[str] = dspy.InputField(
        desc="Source statements that were converted"
    )
    turtle_triples: str = dspy.InputField(
        desc="Generated RDF in Turtle format"
    )
    schema_context: str = dspy.InputField(
        desc="Schema terms that were available"
    )
    
    syntax_valid: bool = dspy.OutputField(
        desc="Is the Turtle syntax valid?"
    )
    uris_correct: bool = dspy.OutputField(
        desc="Are URIs properly formed (no spaces, encoded correctly)?"
    )
    schema_conformance: float = dspy.OutputField(
        desc="Score 0-1: Are appropriate schema terms used?"
    )
    completeness: float = dspy.OutputField(
        desc="Score 0-1: Are all statements represented in the RDF?"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the evaluation"
    )
