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
1. For each statement, identify what kind of fact it is (quote, employment, education, award, relationship, etc.)
2. Use find_rdf_class and find_rdf_property with DESCRIPTIVE queries to find appropriate schema.org terms
3. ONLY use schema matches with score >= 0.5. If no good match, skip or use rdfs:comment.
4. Use emit_triple or emit_triples to output RDF triples, INCLUDING the statement_id

QUERYING THE SCHEMA TOOLS:
Use descriptive natural language, not just keywords. Include:
- What the thing IS (for classes) or what relationship it represents (for properties)
- Context about domain/range when known

GOOD QUERIES:
- find_rdf_class("a spoken or written quotation attributed to someone") → Quotation
- find_rdf_class("an organization that educates students like a university") → EducationalOrganization
- find_rdf_property("the text content of a quote", subject_type="Quotation") → text
- find_rdf_property("who said or authored a quotation", subject_type="Quotation") → creator, spokenByCharacter
- find_rdf_property("schools someone attended as a student", subject_type="Person") → alumniOf

BAD QUERIES (too vague):
- find_rdf_class("quote") - missing context
- find_rdf_property("said") - doesn't explain the relationship
- find_rdf_property("school") - is it attended? works at? type of?

URI RULES:
- For entities with Wikipedia links like [Albert Einstein](/wiki/Albert_Einstein):
  Use: <https://en.wikipedia.org/wiki/Albert_Einstein>
- For entities in the Entity Registry: Use their provided URI
- For new entities not in either: Use fragment URIs like <#entity_name>
- NEVER invent Wikidata URIs (wd:Q...) unless explicitly provided

BLANK NODE NAMING:
Use the statement ID in blank node names to ensure uniqueness across chunks:
  - For statement [3.5], use: _:s3_5_role1, _:s3_5_role2, etc.
  - NOT just _:pos1, _:pos2 (these collide across statements)

MODELING PATTERNS:

1. QUOTATIONS (when someone said/stated/objected something):
   Create a Quotation instance with text and attribution:
   _:s1_quote1 a schema:Quotation .
   _:s1_quote1 schema:text "The actual quote text" .
   _:s1_quote1 schema:creator <wiki:Person_Who_Said_It> .
   
2. TEMPORAL/QUALIFIED RELATIONSHIPS (dates on relationships):
   Use an intermediate Role node (schema.org Role pattern):
   
   Example - Employment: "Einstein worked at University of Zurich from 1909 to 1911"
     <wiki:Albert_Einstein> schema:worksFor _:s1_role1 .
     _:s1_role1 a schema:OrganizationRole .
     _:s1_role1 schema:worksFor <wiki:University_of_Zurich> .
     _:s1_role1 schema:startDate "1909"^^xsd:gYear .
     _:s1_role1 schema:endDate "1911"^^xsd:gYear .

   Example - Awards: "Einstein received the Nobel Prize in 1921"
     <wiki:Albert_Einstein> schema:award _:s2_award1 .
     _:s2_award1 a schema:Role .
     _:s2_award1 schema:award <wiki:Nobel_Prize_in_Physics> .
     _:s2_award1 schema:startDate "1921"^^xsd:gYear .

   Example - Education: "Einstein attended University of Zurich from 1896 to 1900"
     <wiki:Albert_Einstein> schema:alumniOf _:s3_edu1 .
     _:s3_edu1 a schema:OrganizationRole .
     _:s3_edu1 schema:alumniOf <wiki:University_of_Zurich> .
     _:s3_edu1 schema:startDate "1896"^^xsd:gYear .
     _:s3_edu1 schema:endDate "1900"^^xsd:gYear .

3. SIMPLE FACTS (no dates/qualifiers needed):
   Direct triples without Role:
   <wiki:Einstein> schema:birthDate "1879-03-14"^^xsd:date .
   <wiki:Einstein> schema:birthPlace <wiki:Ulm> .

WHAT NOT TO DO:
1. DO NOT use schema:description for quotes, opinions, or narrative facts. 
   schema:description is ONLY for describing what an entity IS, not what happened to it.
   WRONG: <Einstein> schema:description "God does not play dice" .
   RIGHT: Create a schema:Quotation with schema:text for the quote.
   
2. DO NOT emit unrelated triples when no good schema match exists.
   If a statement is about "being sick in bed", don't emit schema:birthDate.
   If no good match (score >= 0.5), either:
   - Skip the statement entirely
   - Use rdfs:comment to preserve the text: <Einstein> rdfs:comment "Statement text here" .

3. DO NOT conflate past and present states.
   "Einstein renounced his German citizenship in 1896" means he LOST it.
   Use endDate on a citizenship Role, not startDate.

4. DO NOT extract facts not stated in the statement.
   Only model what the specific statement says, not general knowledge about the subject.

IMPORTANT:
- Do NOT write Turtle syntax in your response - use the emit tools instead
- ALWAYS include the statement_id when emitting triples
- Include statement_id in blank node names: _:s{{id}}_role1
- Only emit triples for facts that CAN be properly modeled in schema.org
- It is better to skip a statement than emit incorrect triples"""

RDF_STATEMENT_HUMAN_PROMPT = """Convert these factual statements to RDF triples.

Source: {source_url}
Section context: {breadcrumb}

Entity Registry (use these URIs for known entities):
{entity_registry}

Statements to convert (each has a unique ID like [3.5] - include this in blank node names):
{statements}

CRITICAL RULES:

1. URI EXTRACTION: Convert markdown links to Wikipedia URIs:
   [Albert Einstein](/wiki/Albert_Einstein) → <https://en.wikipedia.org/wiki/Albert_Einstein>

2. BLANK NODE NAMING: Include statement ID to avoid collisions:
   For statement [3.5], use: _:s3_5_role1 (NOT just _:role1)

3. SCHEMA MATCHING: Only use matches with score >= 0.5.
   If no good match exists, either skip the statement or use rdfs:comment.

4. TEMPORAL RELATIONSHIPS: Use Role pattern for dated relationships:
   - Employment, education, awards, membership, citizenship with dates → Role pattern
   - Simple facts without dates → direct triple

5. DO NOT:
   - Use schema:description for quotes or narrative (only for entity descriptions)
   - Emit unrelated triples when no good match exists
   - Extract facts not stated in the statement

Process each statement. If you cannot properly model it, skip it or use rdfs:comment."""

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
