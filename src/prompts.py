"""Prompt templates for LLM interactions."""

FACTS_EXTRACTION_PROMPT = """You are an expert at extracting factual information from text.

Given text from a Wikipedia article, extract standalone factual statements. Each output sentence must be fully interpretable in isolation - it will be processed independently for RDF conversion.

Requirements for every output sentence:
- Explicitly name all entities (people, organizations, events, objects, times, places) in the same sentence
- Never rely on earlier text, other statements, or omitted material for identification
- Do not use any expression whose correct interpretation depends on context:
  * NO pronouns (he, she, it, they, who, which, that, himself, etc.)
  * NO demonstratives (this, that, these, those, such, the former, the latter)
  * NO definite descriptions that refer to something from context (e.g., "the man," "the company") 
  * NO verb phrase anaphora (do so, do it, did too, did the same)
  * NO cross-sentence connectors (therefore, however, in this case, for this reason)
- Each sentence must contain all the lexical material needed to understand what it asserts
- Be verifiable from the source text
- Avoid opinions, interpretations, or hedged language

The text comes from: {source_url}
Section context: {breadcrumb}

Known entities (use these exact names for consistency):
{known_entities}

---
{chunk_text}
---

Extract self-contained factual statements as a bulleted list:
"""

RDF_GENERATION_PROMPT = """You are an expert at converting factual statements to RDF triples in Turtle format.

Convert the following factual statements to RDF using schema.org vocabulary where possible.

Source: {source_url}
Section: {breadcrumb}

Use these prefixes:
{prefixes}

Entity registry (use these URIs):
{entity_registry}

Guidelines:
- Use schema.org properties (schema:birthDate, schema:birthPlace, schema:worksFor, etc.)
- For relationships not in schema.org, use wiki3: prefix
- Include rdfs:label for entities
- Use xsd datatypes for dates and numbers
- Entity URIs should use the source URL as base with fragment identifiers

---
{facts}
---

Generate Turtle RDF:
"""

RDF_STATEMENT_SYSTEM_PROMPT = """You are an expert at converting factual statements to RDF triples.

You have access to these tools:

LOOKUP TOOLS (use first to find correct vocabulary):
- find_rdf_class: Find the best schema.org class/type for an entity
- find_rdf_property: Find the best predicate for a relationship

OUTPUT TOOLS (use to emit your triples - MUST include statement_id):
- emit_triple: Output a single triple with statement_id
- emit_triples: Output multiple triples at once (more efficient)

WORKFLOW:
1. For each statement, use find_rdf_class and find_rdf_property to look up appropriate schema.org terms
2. Then use emit_triple or emit_triples to output RDF triples, INCLUDING the statement_id for each
3. Process ALL statements before giving your final summary

IMPORTANT:
- Do NOT write Turtle syntax in your response - use the emit tools instead
- ALWAYS include the statement_id when emitting triples (e.g., "1", "2", "3")
- Use schema.org terms you looked up (e.g., schema:Person, schema:birthDate)
- For URIs, use angle brackets: <https://...> or fragment references: <#entity_id>
- For literals, use quotes with optional datatype: "value"^^xsd:date or "text"@en
- For prefixed terms as objects, just use the prefix: schema:Person
- Each statement is self-contained - extract all facts from each one"""

RDF_STATEMENT_HUMAN_PROMPT = """Convert these factual statements to RDF triples.

Source: {source_url}
Section context: {breadcrumb}

Entity Registry (use these URIs for known entities):
{entity_registry}

Statements to convert (each has a unique ID you must include when emitting triples):
{statements}

For EACH statement:
1. Look up appropriate schema.org classes and properties
2. Emit triples using emit_triple or emit_triples, ALWAYS including the statement_id

Process all statements, then provide a brief summary."""

RDF_PREFIXES = """@prefix schema: <https://schema.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix wiki3: <https://wiki3.ai/vocab/> .
@base <{source_url}> .
"""
