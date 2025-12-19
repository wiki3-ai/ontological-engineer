"""Prompt templates for LLM interactions."""

FACTS_EXTRACTION_PROMPT = """You are an expert at extracting factual information from text while preserving entity references.

Given text from a Wikipedia article (with markdown links to entity pages), extract standalone factual statements. Each output sentence must be fully interpretable in isolation - it will be processed independently for RDF conversion.

CRITICAL: Preserve all markdown links from the source text. Links like [Albert Einstein](/wiki/Albert_Einstein) contain the entity's Wikipedia URI which is essential for RDF generation.

Requirements for every output sentence:
- PRESERVE markdown links: Copy [link text](url) exactly as they appear in the source
- For entities without links in the source, add a link if you know the Wikipedia page: [Entity Name](/wiki/Entity_Name)
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

Example input:
"[Albert Einstein](/wiki/Albert_Einstein) was born in [Ulm](/wiki/Ulm) in the [German Empire](/wiki/German_Empire) on 14 March 1879."

Example output:
- [Albert Einstein](/wiki/Albert_Einstein) was born on 14 March 1879.
- [Albert Einstein](/wiki/Albert_Einstein) was born in [Ulm](/wiki/Ulm).
- [Ulm](/wiki/Ulm) was part of the [German Empire](/wiki/German_Empire) in 1879.

Source URL: {source_url}
Section context: {breadcrumb}

Known entities (use these exact names and URIs for consistency):
{known_entities}

---
{chunk_text}
---

Extract self-contained factual statements as a bulleted list, preserving all markdown links:
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

The statements contain markdown links like [Entity Name](/wiki/Entity_Name) that provide Wikipedia URIs for entities.
Convert these to proper RDF URIs using: <https://en.wikipedia.org/wiki/Entity_Name>

You have access to these tools:

LOOKUP TOOLS (use first to find correct vocabulary):
- find_rdf_class: Find the best schema.org class/type for an entity
- find_rdf_property: Find the best predicate for a relationship

OUTPUT TOOLS (use to emit your triples - MUST include statement_id):
- emit_triple: Output a single triple with statement_id
- emit_triples: Output multiple triples at once (more efficient)

WORKFLOW:
1. For each statement, extract entities from markdown links: [Label](/wiki/Path) → <https://en.wikipedia.org/wiki/Path>
2. Use find_rdf_class and find_rdf_property to look up appropriate schema.org terms
3. Use emit_triple or emit_triples to output RDF triples, INCLUDING the statement_id

URI RULES:
- For entities with Wikipedia links like [Albert Einstein](/wiki/Albert_Einstein):
  Use: <https://en.wikipedia.org/wiki/Albert_Einstein>
- For entities in the Entity Registry: Use their provided URI
- For new entities not in either: Use fragment URIs like <#entity_name>
- NEVER invent Wikidata URIs (wd:Q...) unless explicitly provided

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

IMPORTANT: Statements contain markdown links like [Entity](/wiki/Path). Convert these to Wikipedia URIs:
[Albert Einstein](/wiki/Albert_Einstein) → <https://en.wikipedia.org/wiki/Albert_Einstein>

For EACH statement:
1. Extract entity URIs from markdown links
2. Look up appropriate schema.org classes and properties
3. Emit triples using emit_triple or emit_triples, ALWAYS including the statement_id

Process all statements, then provide a brief summary."""

RDF_PREFIXES = """@prefix schema: <https://schema.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix wiki3: <https://wiki3.ai/vocab/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix repro: <https://w3id.org/reproduceme#> .
@prefix pplan: <http://purl.org/net/p-plan#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@base <{source_url}> .
"""
