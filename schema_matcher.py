"""
Schema.org Vocabulary Matcher using Embeddings

This module provides tools for finding the best schema.org (and other RDF vocabulary)
matches for natural language descriptions of entities and relationships.

It uses text embeddings to find semantically similar terms from the vocabulary.
"""

import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional
import numpy as np

# We'll use sentence-transformers or the OpenAI-compatible embedding API
# For LM Studio, you can load models like:
# - nomic-ai/nomic-embed-text-v1.5
# - BAAI/bge-small-en-v1.5
# - sentence-transformers/all-MiniLM-L6-v2


@dataclass
class VocabTerm:
    """A term from an RDF vocabulary."""
    uri: str
    label: str
    description: str
    term_type: str  # "class" or "property"
    domain: Optional[str] = None  # For properties: what class it applies to
    range: Optional[str] = None   # For properties: what type of value
    parent: Optional[str] = None  # Parent class/property
    usage_hint: Optional[str] = None  # How to use this term in RDF
    example_keywords: Optional[str] = None  # Keywords that should match this term
    
    def to_search_text(self) -> str:
        """Generate text for embedding search."""
        parts = [self.label]
        if self.description:
            parts.append(self.description)
        if self.domain:
            parts.append(f"applies to {self.domain}")
        if self.range:
            parts.append(f"value is {self.range}")
        if self.example_keywords:
            parts.append(self.example_keywords)
        return " | ".join(parts)


class SchemaVocabulary:
    """Manages a vocabulary with embedding-based search."""
    
    def __init__(self, name: str, prefix: str, base_uri: str):
        self.name = name
        self.prefix = prefix
        self.base_uri = base_uri
        self.terms: dict[str, VocabTerm] = {}
        self.embeddings: Optional[np.ndarray] = None
        self.term_ids: list[str] = []  # Maps embedding index to term URI
    
    def add_term(self, term: VocabTerm):
        """Add a term to the vocabulary."""
        self.terms[term.uri] = term
    
    def get_classes(self) -> list[VocabTerm]:
        """Get all class terms."""
        return [t for t in self.terms.values() if t.term_type == "class"]
    
    def get_properties(self) -> list[VocabTerm]:
        """Get all property terms."""
        return [t for t in self.terms.values() if t.term_type == "property"]
    
    def build_index(self, embed_fn):
        """Build embedding index for all terms.
        
        Args:
            embed_fn: Function that takes list[str] and returns np.ndarray of embeddings
        """
        self.term_ids = list(self.terms.keys())
        texts = [self.terms[uri].to_search_text() for uri in self.term_ids]
        self.embeddings = embed_fn(texts)
    
    def search(self, query: str, embed_fn, top_k: int = 5, 
               term_type: Optional[str] = None) -> list[tuple[VocabTerm, float]]:
        """Search for matching terms.
        
        Args:
            query: Natural language description to match
            embed_fn: Function that embeds a single string
            top_k: Number of results to return
            term_type: Filter to "class" or "property"
        
        Returns:
            List of (term, similarity_score) tuples
        """
        if self.embeddings is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Embed the query
        query_embedding = embed_fn([query])[0]
        
        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Filter by type if specified
        if term_type:
            mask = np.array([self.terms[uri].term_type == term_type for uri in self.term_ids])
            similarities = np.where(mask, similarities, -1)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Skip filtered out terms
                term = self.terms[self.term_ids[idx]]
                results.append((term, float(similarities[idx])))
        
        return results
    
    def save(self, path: str):
        """Save vocabulary and embeddings to disk."""
        data = {
            "name": self.name,
            "prefix": self.prefix,
            "base_uri": self.base_uri,
            "terms": {uri: {
                "uri": t.uri,
                "label": t.label,
                "description": t.description,
                "term_type": t.term_type,
                "domain": t.domain,
                "range": t.range,
                "parent": t.parent,
                "usage_hint": t.usage_hint,
                "example_keywords": t.example_keywords,
            } for uri, t in self.terms.items()},
            "term_ids": self.term_ids,
        }
        with open(path + ".json", "w") as f:
            json.dump(data, f, indent=2)
        
        if self.embeddings is not None:
            np.save(path + ".npy", self.embeddings)
    
    @classmethod
    def load(cls, path: str) -> 'SchemaVocabulary':
        """Load vocabulary and embeddings from disk."""
        with open(path + ".json", "r") as f:
            data = json.load(f)
        
        vocab = cls(data["name"], data["prefix"], data["base_uri"])
        for uri, t in data["terms"].items():
            vocab.terms[uri] = VocabTerm(**t)
        vocab.term_ids = data["term_ids"]
        
        npy_path = path + ".npy"
        if os.path.exists(npy_path):
            vocab.embeddings = np.load(npy_path)
        
        return vocab


class SchemaMatcher:
    """Multi-vocabulary schema matcher with LLM-callable interface."""
    
    def __init__(self, embed_base_url: str = "http://host.docker.internal:1234/v1",
                 embed_model: str = "s3dev-ai/text-embedding-nomic-embed-text-v1.5"):
        self.vocabularies: dict[str, SchemaVocabulary] = {}
        self.embed_base_url = embed_base_url
        self.embed_model = embed_model
        self._embed_client = None
    
    def _get_embed_client(self):
        """Lazy-load embedding client."""
        if self._embed_client is None:
            from openai import OpenAI
            self._embed_client = OpenAI(
                base_url=self.embed_base_url,
                api_key="lm-studio"
            )
        return self._embed_client
    
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts using the configured model."""
        client = self._get_embed_client()
        response = client.embeddings.create(
            model=self.embed_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def add_vocabulary(self, vocab: SchemaVocabulary):
        """Add a vocabulary to the matcher."""
        self.vocabularies[vocab.prefix] = vocab
    
    def build_all_indexes(self):
        """Build embedding indexes for all vocabularies."""
        for vocab in self.vocabularies.values():
            print(f"Building index for {vocab.name}...")
            vocab.build_index(self.embed)
            print(f"  Indexed {len(vocab.terms)} terms")
    
    def find_class(self, description: str, top_k: int = 5) -> list[dict]:
        """Find matching RDF classes for a natural language description.
        
        Use this to find the right class/type for an entity.
        Example: find_class("a person who creates scientific theories")
        """
        results = []
        for vocab in self.vocabularies.values():
            matches = vocab.search(description, self.embed, top_k=top_k, term_type="class")
            for term, score in matches:
                results.append({
                    "uri": term.uri,
                    "prefix": f"{vocab.prefix}:{term.label}",
                    "label": term.label,
                    "description": term.description,
                    "parent": term.parent,
                    "usage_hint": term.usage_hint,
                    "score": score,
                    "vocabulary": vocab.name,
                })
        
        # Sort by score and return top_k overall
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def find_property(self, description: str, subject_type: Optional[str] = None,
                      object_type: Optional[str] = None, top_k: int = 5) -> list[dict]:
        """Find matching RDF properties for a natural language description.
        
        Use this to find the right predicate for a relationship.
        Example: find_property("the date someone was born", subject_type="Person")
        """
        # Enhance query with type context
        query_parts = [description]
        if subject_type:
            query_parts.append(f"applies to {subject_type}")
        if object_type:
            query_parts.append(f"value is {object_type}")
        query = " | ".join(query_parts)
        
        results = []
        for vocab in self.vocabularies.values():
            matches = vocab.search(query, self.embed, top_k=top_k, term_type="property")
            for term, score in matches:
                results.append({
                    "uri": term.uri,
                    "prefix": f"{vocab.prefix}:{term.label}",
                    "label": term.label,
                    "description": term.description,
                    "domain": term.domain,
                    "range": term.range,
                    "usage_hint": term.usage_hint,
                    "score": score,
                    "vocabulary": vocab.name,
                })
        
        # Sort by score and return top_k overall
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def find_triple_components(self, subject_desc: str, predicate_desc: str, 
                                object_desc: str, top_k: int = 3) -> dict:
        """Find matching vocabulary terms for all parts of a triple.
        
        Use this when you have a complete fact and need to map all parts.
        Example: find_triple_components(
            subject_desc="Albert Einstein, a physicist",
            predicate_desc="was born in",
            object_desc="the city of Ulm"
        )
        """
        return {
            "subject_types": self.find_class(subject_desc, top_k=top_k),
            "predicates": self.find_property(predicate_desc, top_k=top_k),
            "object_types": self.find_class(object_desc, top_k=top_k),
        }
    
    def save(self, directory: str):
        """Save all vocabularies to a directory."""
        os.makedirs(directory, exist_ok=True)
        for prefix, vocab in self.vocabularies.items():
            vocab.save(os.path.join(directory, prefix))
        
        # Save config
        config = {
            "embed_model": self.embed_model,
            "vocabularies": list(self.vocabularies.keys()),
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, directory: str, embed_base_url: str = "http://host.docker.internal:1234/v1") -> 'SchemaMatcher':
        """Load matcher from directory."""
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
        
        matcher = cls(embed_base_url=embed_base_url, embed_model=config["embed_model"])
        for prefix in config["vocabularies"]:
            vocab = SchemaVocabulary.load(os.path.join(directory, prefix))
            matcher.vocabularies[prefix] = vocab
        
        return matcher


# Curated usage hints and keywords for important schema.org terms
# These help the LLM understand HOW to use each term, not just WHAT it means
SCHEMA_USAGE_HINTS = {
    # === CLASSES ===
    "Quotation": {
        "usage_hint": "Create a Quotation instance with schema:text for the quote text. Use schema:creator or schema:author for who said/wrote it. Use schema:spokenByCharacter if attributing within a creative work.",
        "example_keywords": "quote, quotation, said, stated, remarked, exclaimed, objected, famous saying, attributed words",
    },
    "Role": {
        "usage_hint": "Use as an intermediate node to add qualifiers (dates, role name) to a relationship. Pattern: subject -> property -> Role -> property -> object, with dates on the Role.",
        "example_keywords": "role, position, capacity, function, qualified relationship, temporal relationship",
    },
    "OrganizationRole": {
        "usage_hint": "Use for employment, membership, or affiliation with dates. Pattern: Person -> worksFor/memberOf -> OrganizationRole -> worksFor/memberOf -> Organization, add startDate/endDate to the role.",
        "example_keywords": "employment, job, position, membership, affiliation, worked at, employed by",
    },
    "EducationalOrganization": {
        "usage_hint": "Type for schools, universities, colleges. Use with schema:alumniOf (person was student) or schema:employee (person worked there).",
        "example_keywords": "school, university, college, institute, academy, educational institution",
    },
    "Award": {
        "usage_hint": "Type for awards, honors, prizes. Person receives award via schema:award property.",
        "example_keywords": "award, prize, honor, medal, recognition, Nobel, trophy",
    },
    "Event": {
        "usage_hint": "Type for events with dates/locations. Use schema:startDate, schema:endDate, schema:location.",
        "example_keywords": "event, conference, ceremony, meeting, occurrence, happened",
    },
    "CreativeWork": {
        "usage_hint": "Base type for books, articles, papers, etc. Use schema:author, schema:datePublished, schema:about.",
        "example_keywords": "work, publication, article, book, paper, writing, creative work",
    },
    
    # === PROPERTIES ===
    "text": {
        "usage_hint": "The textual content of a CreativeWork (including Quotation). Use this for the actual quote text in a Quotation.",
        "example_keywords": "text content, quote text, body, content, words, the actual text",
    },
    "spokenByCharacter": {
        "usage_hint": "The person/character who spoke the quotation. Domain is Quotation. For real people quoted, use this or schema:creator.",
        "example_keywords": "said by, spoken by, quoted, attributed to, who said",
    },
    "author": {
        "usage_hint": "The author/creator of content. Use for who wrote/said something. Works for CreativeWork, Quotation, etc.",
        "example_keywords": "author, writer, creator, who wrote, who said, by",
    },
    "creator": {
        "usage_hint": "The creator of a CreativeWork. Similar to author but more general.",
        "example_keywords": "creator, made by, created by, author",
    },
    "alumniOf": {
        "usage_hint": "Schools/organizations someone attended as a student. Use OrganizationRole pattern if dates are needed.",
        "example_keywords": "attended, studied at, graduated from, alumnus, student of, went to school",
    },
    "worksFor": {
        "usage_hint": "Organization someone works/worked for. Use OrganizationRole pattern for employment with dates.",
        "example_keywords": "works for, employed by, job at, position at, worked at",
    },
    "memberOf": {
        "usage_hint": "Organization someone is a member of. Use Role pattern if membership has dates.",
        "example_keywords": "member of, belongs to, part of, affiliated with",
    },
    "award": {
        "usage_hint": "An award someone received. Use Role pattern if the award has a date: Person -> award -> Role -> award -> Award, with startDate on Role.",
        "example_keywords": "received award, won, awarded, honored with, prize",
    },
    "birthDate": {
        "usage_hint": "Date of birth. Use xsd:date format. Only for actual birth, not other events.",
        "example_keywords": "born, birth date, date of birth, birthday",
    },
    "deathDate": {
        "usage_hint": "Date of death. Use xsd:date format.",
        "example_keywords": "died, death date, date of death, passed away",
    },
    "birthPlace": {
        "usage_hint": "Place where someone was born. Value should be a Place.",
        "example_keywords": "born in, birthplace, place of birth, native of",
    },
    "nationality": {
        "usage_hint": "Nationality of a person. Use for current/primary nationality. For historical citizenships with dates, consider Role pattern.",
        "example_keywords": "nationality, citizen of, national of",
    },
    "citizenship": {
        "usage_hint": "Country of citizenship. For citizenship that changed over time, use Role pattern with dates.",
        "example_keywords": "citizenship, citizen, naturalized, renounced citizenship",
    },
    "spouse": {
        "usage_hint": "Spouse of a person. For marriage dates, use Role pattern.",
        "example_keywords": "married to, spouse, husband, wife, partner",
    },
    "children": {
        "usage_hint": "Children of a person.",
        "example_keywords": "children, son, daughter, child, kids",
    },
    "parent": {
        "usage_hint": "Parent of a person.",
        "example_keywords": "parent, father, mother, parents",
    },
    "knows": {
        "usage_hint": "Person that this person knows. Use for general acquaintance relationships.",
        "example_keywords": "knows, friend, acquaintance, colleague",
    },
    "description": {
        "usage_hint": "A description of the item - what the entity IS. NOT for quotes, events, or narrative facts about the entity.",
        "example_keywords": "is described as, is a, definition, what it is",
    },
    "sameAs": {
        "usage_hint": "URL of a reference page that identifies the item (Wikipedia, Wikidata, official site).",
        "example_keywords": "same as, equivalent, identical to, also known as URL",
    },
    "about": {
        "usage_hint": "The subject matter of a CreativeWork.",
        "example_keywords": "about, concerning, regarding, subject matter, topic",
    },
}


def load_schema_org_vocabulary() -> SchemaVocabulary:
    """Load schema.org vocabulary from their JSON-LD definition."""
    import urllib.request
    
    vocab = SchemaVocabulary(
        name="Schema.org",
        prefix="schema",
        base_uri="https://schema.org/"
    )
    
    # Fetch schema.org definitions
    url = "https://schema.org/version/latest/schemaorg-current-https.jsonld"
    print(f"Fetching schema.org from {url}...")
    
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    
    # Parse the JSON-LD graph
    graph = data.get("@graph", [])
    print(f"Processing {len(graph)} schema.org definitions...")
    
    for item in graph:
        item_id = item.get("@id", "")
        if not item_id.startswith("schema:"):
            continue
        
        uri = f"https://schema.org/{item_id[7:]}"  # Remove "schema:" prefix
        label = item_id[7:]  # The local name
        
        # Get description
        description = ""
        comment = item.get("rdfs:comment", "")
        if isinstance(comment, dict):
            description = comment.get("@value", "")
        elif isinstance(comment, str):
            description = comment
        
        # Get curated usage hints if available
        hints = SCHEMA_USAGE_HINTS.get(label, {})
        usage_hint = hints.get("usage_hint")
        example_keywords = hints.get("example_keywords")
        
        # Determine type
        item_types = item.get("@type", [])
        if isinstance(item_types, str):
            item_types = [item_types]
        
        if "rdfs:Class" in item_types:
            term_type = "class"
            parent = None
            subclass = item.get("rdfs:subClassOf", {})
            if isinstance(subclass, dict):
                parent = subclass.get("@id", "").replace("schema:", "")
            elif isinstance(subclass, list) and subclass:
                parent = subclass[0].get("@id", "").replace("schema:", "")
            
            vocab.add_term(VocabTerm(
                uri=uri,
                label=label,
                description=description[:500],  # Truncate long descriptions
                term_type=term_type,
                parent=parent,
                usage_hint=usage_hint,
                example_keywords=example_keywords,
            ))
        
        elif "rdf:Property" in item_types:
            term_type = "property"
            
            # Get domain (what classes this applies to)
            domain = None
            domain_includes = item.get("schema:domainIncludes", [])
            if isinstance(domain_includes, dict):
                domain = domain_includes.get("@id", "").replace("schema:", "")
            elif isinstance(domain_includes, list) and domain_includes:
                domains = [d.get("@id", "").replace("schema:", "") for d in domain_includes if isinstance(d, dict)]
                domain = ", ".join(domains[:3])  # First 3
            
            # Get range (what type of values)
            range_type = None
            range_includes = item.get("schema:rangeIncludes", [])
            if isinstance(range_includes, dict):
                range_type = range_includes.get("@id", "").replace("schema:", "")
            elif isinstance(range_includes, list) and range_includes:
                ranges = [r.get("@id", "").replace("schema:", "") for r in range_includes if isinstance(r, dict)]
                range_type = ", ".join(ranges[:3])  # First 3
            
            vocab.add_term(VocabTerm(
                uri=uri,
                label=label,
                description=description[:500],
                term_type=term_type,
                domain=domain,
                range=range_type,
                usage_hint=usage_hint,
                example_keywords=example_keywords,
            ))
    
    print(f"Loaded {len(vocab.get_classes())} classes and {len(vocab.get_properties())} properties")
    return vocab


if __name__ == "__main__":
    # Test loading schema.org
    vocab = load_schema_org_vocabulary()
    print(f"\nSample classes:")
    for term in list(vocab.get_classes())[:5]:
        print(f"  {term.label}: {term.description[:60]}...")
    
    print(f"\nSample properties:")
    for term in list(vocab.get_properties())[:5]:
        print(f"  {term.label} ({term.domain} -> {term.range}): {term.description[:40]}...")
