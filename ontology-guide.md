# Wiki3.ai RDF-Star Ontology Guide (Revised)
**Self-Documenting Knowledge Extraction with W3C-Recommended Vocabularies**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Ontology Stack](#ontology-stack)
4. [Complete Example: Albert Einstein](#complete-example-albert-einstein)
5. [SPARQL-Star Queries](#sparql-star-queries)
6. [Integration with DSPy](#integration-with-dspy)
7. [Tools & Deployment](#tools--deployment)

---

## Executive Summary

Wiki3.ai extracts **self-documenting RDF-Star knowledge graphs** from Wikipedia. Every fact includes:

- **Content metadata** (confidence, source span, extraction quality)
- **System metadata** (which LLM? DSPy version? Agent who triggered? Timestamp?)
- **Computational provenance** (algorithm chain, intermediate steps, execution time)
- **Reproducibility** (exact component versions, dependencies, ontology versions)

**Key principle:** The extraction system describes *itself* in the same RDF framework it uses to describe Wikipedia content.

**Namespace Convention:** This guide uses standard W3C namespace prefixes. Examples use placeholder URIs like `ex:Albert_Einstein` for illustration only—actual deployment should use dereferenceable URIs (e.g., Wikidata QIDs or wiki3.ai namespace URIs).

---

## Architecture Overview

```
Wikipedia Article Text
        ↓
   [DSPy Extraction Pipeline]
        ├── Entity Linking (ReFinED)
        ├── Relation Extraction (REBEL)
        ├── Triple Generation
        └── GEPA Optimization
        ↓
   RDF-Star Knowledge Graph
        ├── Content triples (schema:Person, schema:Event, org:Membership)
        ├── RDF-Star annotations (confidence, source, provenance)
        ├── System metadata (DOAP, SPDX, PROV-O, SWO)
        └── Agent interaction log (FOAF + custom interfaces)
        ↓
   [Publication]
        ├── Turtle serialization → IPFS
        ├── JSON-LD → web interface
        └── SPARQL-Star queries
```

---

## Ontology Stack

### **Tier 1: Content Extraction** (Knowledge about Wikipedia)

#### 1.1 Core RDF/Semantic Web Standards

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **RDF** | `rdf:` http://www.w3.org/1999/02/22-rdf-syntax-ns# | Recommendation | RDF data model | `rdf:type`, `rdf:Property`, `rdf:Statement` |
| **RDFS** | `rdfs:` http://www.w3.org/2000/01/rdf-schema# | Recommendation | Basic typing & hierarchy | `rdfs:Class`, `rdfs:label`, `rdfs:comment`, `rdfs:domain`, `rdfs:range` |
| **OWL 2** | `owl:` http://www.w3.org/2002/07/owl# | Recommendation | Ontology description, reasoning | `owl:sameAs`, `owl:equivalentClass`, `owl:FunctionalProperty` |
| **RDF-Star** | (part of RDF 1.2) | Working Draft | Annotations on triples | `<< s p o >> {&#124; annotations &#124;}` |

**Example:**
```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:Albert_Einstein a schema:Person ;
  rdfs:label "Albert Einstein"@en ;
  schema:birthDate "1879-03-14"^^xsd:date .

# RDF-Star annotation with confidence
<< ex:Albert_Einstein schema:birthDate "1879-03-14"^^xsd:date >> {|
  wiki3:confidence 0.95 ;
  wiki3:sourceSpan "born on March 14, 1879 in Ulm" ;
|} .
```

#### 1.2 People & Organizations

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **FOAF** | `foaf:` http://xmlns.com/foaf/0.1/ | Community standard | Person, Agent, basic relationships | `foaf:Person`, `foaf:name`, `foaf:givenName`, `foaf:familyName`, `foaf:knows`, `foaf:Organization` |
| **schema.org** | `schema:` http://schema.org/ | De facto web standard | Rich vocabulary for people, orgs, events | `schema:Person`, `schema:Organization`, `schema:Event`, `schema:birthDate`, `schema:worksFor` |
| **W3C ORG** | `org:` http://www.w3.org/ns/org# | Recommendation | Organizational structures, roles, memberships | `org:Organization`, `org:Membership`, `org:Role`, `org:member`, `org:memberOf`, `org:hasMembership` |

**Example:**
```turtle
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix schema: <http://schema.org/> .
@prefix org: <http://www.w3.org/ns/org#> .

ex:Albert_Einstein a foaf:Person, schema:Person ;
  foaf:name "Albert Einstein"@en ;
  schema:givenName "Albert" ;
  schema:familyName "Einstein" ;
  foaf:homepage <https://en.wikipedia.org/wiki/Albert_Einstein> ;
  schema:nationality ex:Germany ;
.

ex:University_of_Bern a foaf:Organization, schema:EducationalOrganization ;
  foaf:name "University of Bern"@en ;
  schema:name "University of Bern"@en ;
.
```

#### 1.3 Events & Activities

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **schema.org** | `schema:` | De facto standard | General events, actions | `schema:Event`, `schema:Action`, `schema:startDate`, `schema:endDate`, `schema:location`, `schema:organizer`, `schema:participant` |
| **LODE** | `lode:` http://linkedevents.org/ontology/ | Community standard | Event ontology interlingua | `lode:Event`, `lode:atPlace`, `lode:atTime`, `lode:involved` |
| **SEM** | `sem:` http://semanticweb.cs.vu.nl/2009/11/sem/ | Academic standard | Simple Event Model | `sem:Event`, `sem:hasPlace`, `sem:hasTimeStamp`, `sem:hasActor`, `sem:eventProperty` |

**Example:**
```turtle
@prefix schema: <http://schema.org/> .
@prefix lode: <http://linkedevents.org/ontology/> .
@prefix time: <http://www.w3.org/2006/time#> .

ex:einstein_employment_bern a schema:Event, lode:Event ;
  rdfs:label "Einstein's lectureship at University of Bern"@en ;
  schema:startDate "1905"^^xsd:gYear ;
  schema:endDate "1909"^^xsd:gYear ;
  lode:involved ex:Albert_Einstein ;
  lode:involved ex:University_of_Bern ;
  schema:location ex:Bern ;
  schema:description "Lecturer in Theoretical Physics"@en ;
.

# Using ORG Ontology for formal membership
ex:membership_einstein_bern a org:Membership ;
  org:member ex:Albert_Einstein ;
  org:organization ex:University_of_Bern ;
  org:role ex:role_lecturer ;
  org:memberDuring [
    a time:Interval ;
    time:hasBeginning [ time:inXSDDate "1905"^^xsd:gYear ] ;
    time:hasEnd [ time:inXSDDate "1909"^^xsd:gYear ] ;
  ] ;
.

ex:role_lecturer a org:Role ;
  rdfs:label "Lecturer in Theoretical Physics"@en ;
.
```

#### 1.4 Temporal Relationships

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **OWL-Time** | `time:` http://www.w3.org/2006/time# | Recommendation | Temporal intervals, Allen's algebra | `time:Interval`, `time:Instant`, `time:hasBeginning`, `time:hasEnd`, `time:before`, `time:after`, `time:during` |

**Example:**
```turtle
@prefix time: <http://www.w3.org/2006/time#> .

ex:einstein_period_germany a time:Interval ;
  time:hasBeginning [
    a time:Instant ;
    time:inXSDDate "1879"^^xsd:gYear ;
  ] ;
  time:hasEnd [
    a time:Instant ;
    time:inXSDDate "1933"^^xsd:gYear ;
  ] ;
.

ex:weimar_republic a time:Interval ;
  time:hasBeginning [ time:inXSDDate "1919"^^xsd:gYear ] ;
  time:hasEnd [ time:inXSDDate "1933"^^xsd:gYear ] ;
.

# Einstein's time in Germany overlaps with Weimar Republic
ex:einstein_period_germany time:intervalDuring ex:weimar_republic .
```

#### 1.5 Provenance & Metadata

| Ontology | Namespace | W3C Status | Purpose | Key Properties |
|----------|-----------|------------|---------|-----------------|
| **Dublin Core** | `dcterms:` http://purl.org/dc/terms/ | ISO 15836 standard | Document & general metadata | `dcterms:title`, `dcterms:creator`, `dcterms:date`, `dcterms:source`, `dcterms:language`, `dcterms:subject` |

**Example:**
```turtle
@prefix dcterms: <http://purl.org/dc/terms/> .

ex:wikipedia_albert_einstein_article a schema:WebPage ;
  dcterms:title "Albert Einstein - Wikipedia"@en ;
  dcterms:language "en" ;
  dcterms:source <https://en.wikipedia.org/wiki/Albert_Einstein> ;
  dcterms:date "2025-12-19" ;
  dcterms:subject "Physics", "Biography" ;
.
```

---

### **Tier 2: System & Software** (Knowledge about wiki3.ai)

#### 2.1 Project & Software Description

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **DOAP** | `doap:` http://usefulinc.com/ns/doap# | Community standard (used by Apache, GitHub) | Software project metadata | `doap:Project`, `doap:name`, `doap:GitRepository`, `doap:programming-language`, `doap:license`, `doap:release` |

**Example:**
```turtle
@prefix doap: <http://usefulinc.com/ns/doap#> .

wiki3:Project a doap:Project ;
  doap:name "wiki3.ai" ;
  doap:description "Web3-native knowledge platform extracting RDF from Wikipedia" ;
  doap:homepage <https://wiki3.ai> ;
  doap:programming-language "Python", "TypeScript", "Common Lisp" ;
  doap:license <https://opensource.org/licenses/MIT> ;
  doap:repository [
    a doap:GitRepository ;
    doap:location <https://github.com/wiki3-ai/ontological-engineer> ;
  ] ;
.
```

#### 2.2 Software Bill of Materials (SBOM)

| Standard | Namespace | Status | Purpose | Key Classes/Properties |
|----------|-----------|--------|---------|------------------------|
| **SPDX 3.0** | `spdx:` https://spdx.org/rdf/terms/ | ISO/IEC 5962:2021 | Supply chain, dependencies, licenses | `spdx:Package`, `spdx:dependsOn`, `spdx:license`, `spdx:externalRef` |

**Example:**
```turtle
@prefix spdx: <https://spdx.org/rdf/terms/> .

wiki3:Package_DSPy a spdx:Package ;
  spdx:name "dspy-ai" ;
  spdx:versionInfo "2.4.3" ;
  spdx:downloadLocation <https://pypi.org/project/dspy-ai/2.4.3/> ;
  spdx:licenseConcluded "MIT" ;
  spdx:description "DSPy: Pythonic framework for optimizing language model programs" ;
.

wiki3:wiki3_environment a spdx:Package ;
  spdx:name "wiki3.ai" ;
  spdx:versionInfo "0.1.0" ;
  spdx:dependsOn wiki3:Package_DSPy ;
  spdx:dependsOn wiki3:Package_RDFLib ;
  spdx:dependsOn wiki3:Package_LangChain ;
.
```

#### 2.3 Computational Provenance

| Ontology | Namespace | W3C Status | Purpose | Key Classes/Properties |
|----------|-----------|------------|---------|------------------------|
| **PROV-O** | `prov:` http://www.w3.org/ns/prov# | Recommendation | Entity, Activity, Agent, relationships | `prov:Entity`, `prov:Activity`, `prov:Agent`, `prov:wasGeneratedBy`, `prov:used`, `prov:wasAssociatedWith`, `prov:startedAtTime`, `prov:endedAtTime` |

**Example:**
```turtle
@prefix prov: <http://www.w3.org/ns/prov#> .

# Extraction activity
wiki3:extraction_einstein_20251219 a prov:Activity ;
  rdfs:label "Extraction of Albert Einstein Wikipedia article" ;
  prov:startedAtTime "2025-12-19T09:30:00Z"^^xsd:dateTime ;
  prov:endedAtTime "2025-12-19T09:45:32Z"^^xsd:dateTime ;
  prov:used ex:wikipedia_albert_einstein_article ;
  prov:wasAssociatedWith wiki3:dspy_optimizer ;
  prov:wasAssociatedWith wiki3:llm_gpt4o ;
.

# Output entity
ex:rdf_output_einstein a prov:Entity ;
  rdfs:label "RDF-Star knowledge graph for Albert Einstein" ;
  prov:wasGeneratedBy wiki3:extraction_einstein_20251219 ;
  prov:wasAttributedTo wiki3:llm_gpt4o ;
  wiki3:format "application/rdf+turtle" ;
  wiki3:numTriples 4238 ;
  wiki3:avgConfidence 0.887 ;
.

# Agents
wiki3:dspy_optimizer a prov:Agent, prov:SoftwareAgent ;
  rdfs:label "DSPy GEPA Optimizer v2.4.3" ;
  wiki3:version "2.4.3" ;
.

wiki3:llm_gpt4o a prov:Agent, prov:SoftwareAgent ;
  rdfs:label "OpenAI GPT-4o" ;
  wiki3:modelVersion "2025-11" ;
.
```

---

### **Tier 3: Agents & Interfaces** (External systems)

#### 3.1 Agents & Actors

| Ontology | Namespace | Purpose | Key Classes/Properties |
|----------|-----------|---------|------------------------|
| **FOAF** | `foaf:` | Agent, Person, Organization | `foaf:Agent`, `foaf:Person`, `foaf:mbox`, `foaf:homepage` |

**Example:**
```turtle
@prefix foaf: <http://xmlns.com/foaf/0.1/> .

wiki3:user_web a foaf:Agent ;
  rdfs:label "Wikipedia article submitter (Web UI)" ;
  foaf:name "Human user" ;
  wiki3:hasInterface wiki3:web_interface ;
.

wiki3:radicle_agent a foaf:Agent ;
  rdfs:label "Radicle Patch Monitor" ;
  foaf:name "wiki3.ai Radicle Node" ;
  wiki3:hasInterface wiki3:radicle_interface ;
.

wiki3:ipfs_agent a foaf:Agent ;
  rdfs:label "IPFS Content Distribution Node" ;
  foaf:name "wiki3.ai IPFS Provider" ;
  wiki3:hasInterface wiki3:ipfs_interface ;
.
```

---

## Complete Example: Albert Einstein

Full RDF-Star knowledge graph with content + system metadata:

```turtle
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix foaf: <http://xmlns.com/foaf/0.1/> .
@prefix schema: <http://schema.org/> .
@prefix org: <http://www.w3.org/ns/org#> .
@prefix time: <http://www.w3.org/2006/time#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix prov: <http://www.w3.org/ns/prov#> .
@prefix doap: <http://usefulinc.com/ns/doap#> .
@prefix spdx: <https://spdx.org/rdf/terms/> .
@prefix lode: <http://linkedevents.org/ontology/> .

# Define wiki3 namespace for system-specific terms
@prefix wiki3: <https://wiki3.ai/ns/> .
@prefix ex: <https://wiki3.ai/entity/> .

# ===== CONTENT: ALBERT EINSTEIN =====

ex:Albert_Einstein a foaf:Person, schema:Person ;
  foaf:name "Albert Einstein"@en ;
  schema:givenName "Albert" ;
  schema:familyName "Einstein" ;
  foaf:homepage <https://en.wikipedia.org/wiki/Albert_Einstein> ;
  schema:nationality ex:Germany ;
  owl:sameAs <http://www.wikidata.org/entity/Q937> ;
  owl:sameAs <http://dbpedia.org/resource/Albert_Einstein> ;
.

# Birth event using schema.org
ex:einstein_birth a schema:Event ;
  schema:name "Birth of Albert Einstein"@en ;
  schema:startDate "1879-03-14"^^xsd:date ;
  schema:location ex:Ulm ;
  schema:actor ex:Albert_Einstein ;
.

# RDF-Star annotation on birth date
<< ex:einstein_birth schema:startDate "1879-03-14"^^xsd:date >> {|
  wiki3:confidence 0.98 ;
  wiki3:sourceSpan "Albert Einstein was born in Ulm, on 14 March 1879" ;
  wiki3:sourceSection "early_life_and_education" ;
  prov:wasGeneratedBy wiki3:extraction_einstein_20251219 ;
  prov:wasAttributedTo wiki3:llm_gpt4o ;
|} .

# Employment using ORG Ontology
ex:membership_einstein_bern a org:Membership ;
  org:member ex:Albert_Einstein ;
  org:organization ex:University_of_Bern ;
  org:role ex:role_lecturer_physics ;
  org:memberDuring [
    a time:Interval ;
    time:hasBeginning [ time:inXSDDate "1905"^^xsd:gYear ] ;
    time:hasEnd [ time:inXSDDate "1909"^^xsd:gYear ] ;
  ] ;
.

ex:role_lecturer_physics a org:Role ;
  rdfs:label "Lecturer in Theoretical Physics"@en ;
.

ex:University_of_Bern a foaf:Organization, schema:EducationalOrganization, org:Organization ;
  foaf:name "University of Bern"@en ;
  schema:name "University of Bern"@en ;
.

# RDF-Star annotation on membership
<< ex:membership_einstein_bern org:role ex:role_lecturer_physics >> {|
  wiki3:confidence 0.92 ;
  wiki3:sourceSpan "In 1905, Einstein was appointed lecturer at the University of Bern" ;
  prov:wasGeneratedBy wiki3:extraction_einstein_20251219 ;
|} .

# Nobel Prize as schema.org Award
ex:einstein_nobel_1921 a schema:Award ;
  schema:name "Nobel Prize in Physics"@en ;
  schema:recipient ex:Albert_Einstein ;
  schema:datePublished "1921" ;
  schema:description "For his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect"@en ;
.

<< ex:einstein_nobel_1921 schema:datePublished "1921" >> {|
  wiki3:confidence 0.99 ;
  wiki3:sourceSpan "In 1921, Einstein was awarded the Nobel Prize in Physics" ;
  prov:wasGeneratedBy wiki3:extraction_einstein_20251219 ;
|} .

# ===== SYSTEM METADATA =====

# Source document
ex:wikipedia_albert_einstein_article a schema:WebPage ;
  dcterms:title "Albert Einstein - Wikipedia"@en ;
  dcterms:source <https://en.wikipedia.org/wiki/Albert_Einstein> ;
  dcterms:retrieved "2025-12-19T09:25:00Z"^^xsd:dateTime ;
  wiki3:articleLength 48372 ;
  wiki3:wikiRevisionId "1234567890" ;
.

# Extraction activity (PROV-O)
wiki3:extraction_einstein_20251219 a prov:Activity ;
  rdfs:label "Extraction of Albert Einstein Wikipedia article" ;
  prov:startedAtTime "2025-12-19T09:30:00Z"^^xsd:dateTime ;
  prov:endedAtTime "2025-12-19T09:45:32Z"^^xsd:dateTime ;
  prov:used ex:wikipedia_albert_einstein_article ;
  prov:wasAssociatedWith wiki3:dspy_optimizer_v2_4_3 ;
  prov:wasAssociatedWith wiki3:llm_gpt4o ;
.

# Qualified association with LLM details (RDF-Star)
<< wiki3:extraction_einstein_20251219 prov:wasAssociatedWith wiki3:llm_gpt4o >> {|
  prov:qualifiedAssociation [
    a prov:Association ;
    prov:agent wiki3:llm_gpt4o ;
    prov:hadRole wiki3:role_extractor ;
    wiki3:temperature 0.7 ;
    wiki3:maxTokens 2048 ;
  ] ;
|} .

# Output RDF
ex:rdf_output_einstein a prov:Entity ;
  rdfs:label "RDF-Star Knowledge Graph for Albert Einstein" ;
  prov:wasGeneratedBy wiki3:extraction_einstein_20251219 ;
  prov:wasAttributedTo wiki3:llm_gpt4o ;
  prov:wasAttributedTo wiki3:dspy_optimizer_v2_4_3 ;
  wiki3:format "application/rdf+turtle" ;
  wiki3:sizeBytes 487392 ;
  wiki3:numTriples 4238 ;
  wiki3:numEntities 186 ;
  wiki3:avgConfidence 0.887 ;
.

# Software agents
wiki3:dspy_optimizer_v2_4_3 a prov:Agent, prov:SoftwareAgent ;
  rdfs:label "DSPy GEPA Optimizer v2.4.3" ;
  wiki3:version "2.4.3" ;
  wiki3:componentType "Prompt Optimizer" ;
.

wiki3:llm_gpt4o a prov:Agent, prov:SoftwareAgent ;
  rdfs:label "OpenAI GPT-4o" ;
  wiki3:modelVersion "2025-11" ;
  wiki3:provider "OpenAI" ;
.

# Project metadata (DOAP)
wiki3:Project a doap:Project ;
  doap:name "wiki3.ai" ;
  doap:description "Web3 knowledge platform extracting RDF from Wikipedia" ;
  doap:homepage <https://wiki3.ai> ;
  doap:programming-language "Python", "TypeScript" ;
  doap:license <https://opensource.org/licenses/MIT> ;
  doap:repository [
    a doap:GitRepository ;
    doap:location <https://github.com/wiki3-ai/ontological-engineer> ;
  ] ;
.

# SBOM (SPDX)
wiki3:package_dspy a spdx:Package ;
  spdx:name "dspy-ai" ;
  spdx:versionInfo "2.4.3" ;
  spdx:downloadLocation <https://pypi.org/project/dspy-ai/2.4.3/> ;
  spdx:licenseConcluded "MIT" ;
.

wiki3:wiki3_environment a spdx:Package ;
  spdx:name "wiki3.ai" ;
  spdx:versionInfo "0.1.0" ;
  spdx:dependsOn wiki3:package_dspy ;
.

# Temporal timeline (OWL-Time)
wiki3:extraction_timeline_ae a time:Interval ;
  time:hasBeginning [ time:inXSDDateTime "2025-12-19T09:30:00Z"^^xsd:dateTime ] ;
  time:hasEnd [ time:inXSDDateTime "2025-12-19T09:45:32Z"^^xsd:dateTime ] ;
.

# User agent
wiki3:user_web a foaf:Agent ;
  rdfs:label "User via Web Interface" ;
  wiki3:hasInterface wiki3:web_interface ;
.

wiki3:web_interface a wiki3:HTTPInterface ;
  wiki3:endpoint <https://api.wiki3.ai/extract> ;
  wiki3:acceptsFormat "application/json" ;
  wiki3:returnsFormat "application/rdf+turtle" ;
.
```

---

## SPARQL-Star Queries

### Query 1: Find all high-confidence facts with provenance

```sparql
PREFIX wiki3: <https://wiki3.ai/ns/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX schema: <http://schema.org/>

SELECT ?subject ?predicate ?object ?confidence ?agent
WHERE {
  << ?subject ?predicate ?object >> {|
    wiki3:confidence ?confidence ;
    prov:wasAttributedTo ?agent ;
  |} .
  
  FILTER (?confidence >= 0.90)
}
ORDER BY DESC(?confidence)
```

### Query 2: What software versions were used?

```sparql
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX wiki3: <https://wiki3.ai/ns/>

SELECT ?agent ?version (COUNT(*) AS ?num_extractions)
WHERE {
  ?extraction a prov:Activity ;
    prov:wasAssociatedWith ?agent .
  ?agent wiki3:version ?version .
}
GROUP BY ?agent ?version
```

### Query 3: Find organizational memberships with time periods

```sparql
PREFIX org: <http://www.w3.org/ns/org#>
PREFIX time: <http://www.w3.org/2006/time#>
PREFIX schema: <http://schema.org/>

SELECT ?person ?org ?role ?start ?end
WHERE {
  ?membership org:member ?person ;
    org:organization ?org ;
    org:role ?role ;
    org:memberDuring ?interval .
  
  ?interval time:hasBeginning [ time:inXSDDate ?start ] ;
    time:hasEnd [ time:inXSDDate ?end ] .
}
```

---

## Integration with DSPy

### Python Example: DSPy → RDF-Star

```python
import dspy
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, FOAF, XSD
import datetime

class BiographyExtractor:
    def __init__(self):
        self.graph = Graph()
        self._setup_namespaces()
    
    def _setup_namespaces(self):
        """Define standard W3C namespaces."""
        self.WIKI3 = Namespace("https://wiki3.ai/ns/")
        self.EX = Namespace("https://wiki3.ai/entity/")
        self.SCHEMA = Namespace("http://schema.org/")
        self.ORG = Namespace("http://www.w3.org/ns/org#")
        self.PROV = Namespace("http://www.w3.org/ns/prov#")
        self.DCTERMS = Namespace("http://purl.org/dc/terms/")
        self.TIME = Namespace("http://www.w3.org/2006/time#")
        
        self.graph.bind("wiki3", self.WIKI3)
        self.graph.bind("ex", self.EX)
        self.graph.bind("schema", self.SCHEMA)
        self.graph.bind("org", self.ORG)
        self.graph.bind("prov", self.PROV)
        self.graph.bind("dcterms", self.DCTERMS)
        self.graph.bind("time", self.TIME)
        self.graph.bind("foaf", FOAF)
    
    def extract(self, article_text):
        """Extract facts and add to RDF graph."""
        # Use DSPy to extract structured facts
        # (actual DSPy code omitted for brevity)
        
        person_uri = self.EX.Albert_Einstein
        self.graph.add((person_uri, RDF.type, FOAF.Person))
        self.graph.add((person_uri, RDF.type, self.SCHEMA.Person))
        self.graph.add((person_uri, FOAF.name, Literal("Albert Einstein", lang="en")))
        
        # Add birth event
        birth_uri = self.EX.einstein_birth
        self.graph.add((birth_uri, RDF.type, self.SCHEMA.Event))
        self.graph.add((birth_uri, self.SCHEMA.startDate, 
                       Literal("1879-03-14", datatype=XSD.date)))
        
        # Add provenance
        extraction_uri = self.WIKI3["extraction_" + datetime.datetime.now().isoformat()]
        self.graph.add((extraction_uri, RDF.type, self.PROV.Activity))
        self.graph.add((extraction_uri, self.PROV.startedAtTime, 
                       Literal(datetime.datetime.now().isoformat(), datatype=XSD.dateTime)))
        
        return self.graph
    
    def to_turtle(self):
        return self.graph.serialize(format="turtle")

# Usage
extractor = BiographyExtractor()
article_text = "Albert Einstein was born on March 14, 1879..."
rdf_graph = extractor.extract(article_text)
print(rdf_graph.to_turtle())
```

---

## Tools & Deployment

### Python Libraries

```bash
# RDF handling (W3C compliant)
pip install rdflib==7.0.0       # RDF-Star support
pip install pyld                 # JSON-LD
pip install sparqlwrapper        # SPARQL queries

# DSPy & LLM
pip install dspy-ai==2.4.3
pip install langchain==0.2.0

# Models
pip install transformers
pip install torch

# SBOM
pip install spdx-tools

# Validation
pip install pyshacl
```

### Docker Compose

```yaml
version: '3.8'

services:
  jupyter:
    image: wiki3/jupyter-dspy:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  graphdb:
    image: ontotext/graphdb:11.0
    ports:
      - "7200:7200"
    volumes:
      - graphdb-data:/opt/graphdb/data

  ipfs:
    image: ipfs/kubo:latest
    ports:
      - "5001:5001"
      - "8080:8080"
    volumes:
      - ipfs-data:/data/ipfs

volumes:
  graphdb-data:
  ipfs-data:
```

---

## Summary

**W3C-Recommended Ontology Stack:**

**Content (Tier 1):**  
- Core: RDF, RDFS, OWL, RDF-Star  
- People/Orgs: FOAF, schema.org, W3C ORG  
- Events: schema.org, LODE, SEM  
- Time: OWL-Time  
- Metadata: Dublin Core  

**System (Tier 2):**  
- Project: DOAP  
- Dependencies: SPDX 3.0  
- Provenance: PROV-O  

**Agents (Tier 3):**  
- FOAF + Custom interface descriptions

**Key differences from previous version:**
- Uses W3C ORG Ontology (not bio: vocab which is not W3C standard)
- Uses schema.org (widely adopted, de facto web standard)
- All namespace prefixes clearly defined
- No custom "x:" namespace in examples—uses wiki3: for system-specific terms
- Consistent use of standard vocabularies throughout
