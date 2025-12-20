"""
Schema Library.

Manages vocabulary/schema/ontology definitions with embedding-based search.
Provides fast retrieval of relevant classes and properties for RDF generation.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import numpy as np


class SchemaLibrary:
    """
    Manages vocabulary and ontology definitions for RDF generation.
    
    Features:
    - Load schemas from JSON files
    - Embedding-based semantic search for relevant terms
    - Prefix management for Turtle output
    - Usage examples and patterns
    """
    
    def __init__(self, vocab_cache_dir: Optional[Path] = None):
        """
        Initialize the schema library.
        
        Args:
            vocab_cache_dir: Directory containing cached schema files
        """
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.prefixes: Dict[str, str] = {}
        self.examples: Dict[str, List[str]] = {}
        self.patterns: Dict[str, str] = {}
        
        # Embeddings for semantic search
        self.embeddings: Optional[np.ndarray] = None
        self.uri_index: List[str] = []
        
        if vocab_cache_dir:
            self.load_from_cache(vocab_cache_dir)
    
    def load_from_cache(self, cache_dir: Path) -> None:
        """Load schema from cached files."""
        cache_dir = Path(cache_dir)
        
        # Load schema JSON
        schema_file = cache_dir / "schema.json"
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            self.classes = schema.get('classes', {})
            self.properties = schema.get('properties', {})
            self.prefixes = schema.get('prefixes', {})
            self.examples = schema.get('examples', {})
            self.patterns = schema.get('patterns', {})
        
        # Load embeddings
        embeddings_file = cache_dir / "schema.npy"
        if embeddings_file.exists():
            self.embeddings = np.load(embeddings_file)
        
        # Load config
        config_file = cache_dir / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.uri_index = config.get('uri_index', [])
    
    def search_relevant(
        self,
        statements: List[str],
        top_k: int = 20,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for relevant schema terms using embeddings.
        
        Args:
            statements: Statements to find relevant terms for
            top_k: Number of results to return
            
        Returns:
            Dict with 'classes' and 'properties' lists
        """
        if self.embeddings is None or not self.uri_index:
            # Fall back to returning all terms if no embeddings
            return {
                'classes': list(self.classes.values())[:top_k],
                'properties': list(self.properties.values())[:top_k],
            }
        
        # Get embedding for combined statements
        query_text = " ".join(statements)
        query_embedding = self._embed_text(query_text)
        
        if query_embedding is None:
            return {
                'classes': list(self.classes.values())[:top_k],
                'properties': list(self.properties.values())[:top_k],
            }
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Collect results
        classes = []
        properties = []
        
        for idx in top_indices:
            uri = self.uri_index[idx]
            if uri in self.classes:
                classes.append(self.classes[uri])
            elif uri in self.properties:
                properties.append(self.properties[uri])
        
        return {
            'classes': classes,
            'properties': properties,
        }
    
    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for text. Override for custom embedding model.
        
        Default implementation returns None (falls back to all terms).
        """
        # TODO: Implement with actual embedding model
        # Could use sentence-transformers, OpenAI embeddings, etc.
        return None
    
    def build_prefix_block(
        self,
        classes: List[str],
        properties: List[str],
    ) -> str:
        """
        Build Turtle prefix declarations for the selected terms.
        
        Args:
            classes: Selected class URIs
            properties: Selected property URIs
            
        Returns:
            Turtle prefix block string
        """
        needed_prefixes: Set[str] = set()
        
        # Find prefixes needed for selected terms
        all_uris = set(classes) | set(properties)
        for uri in all_uris:
            for prefix, namespace in self.prefixes.items():
                if uri.startswith(namespace):
                    needed_prefixes.add(prefix)
                    break
        
        # Always include common prefixes
        needed_prefixes.update(['rdf', 'rdfs', 'xsd', 'owl'])
        
        # Build prefix block
        lines = []
        for prefix in sorted(needed_prefixes):
            if prefix in self.prefixes:
                lines.append(f"@prefix {prefix}: <{self.prefixes[prefix]}> .")
        
        return "\n".join(lines)
    
    def format_classes(self, class_uris: List[str]) -> str:
        """Format class definitions for prompt."""
        lines = []
        for uri in class_uris:
            if uri in self.classes:
                cls = self.classes[uri]
                label = cls.get('label', uri.split('/')[-1])
                desc = cls.get('description', 'No description')
                lines.append(f"- {label} ({uri}): {desc}")
        return "\n".join(lines) if lines else "No specific classes needed."
    
    def format_properties(self, property_uris: List[str]) -> str:
        """Format property definitions for prompt."""
        lines = []
        for uri in property_uris:
            if uri in self.properties:
                prop = self.properties[uri]
                label = prop.get('label', uri.split('/')[-1])
                desc = prop.get('description', '')
                domain = prop.get('domain', 'Any')
                range_type = prop.get('range', 'Any')
                lines.append(f"- {label} ({uri})")
                if desc:
                    lines.append(f"  Description: {desc}")
                lines.append(f"  Domain: {domain}, Range: {range_type}")
        return "\n".join(lines) if lines else "No specific properties needed."
    
    def get_examples(
        self,
        class_uris: List[str],
        property_uris: List[str],
    ) -> str:
        """Get usage examples for selected terms."""
        examples = []
        
        for uri in class_uris + property_uris:
            if uri in self.examples:
                examples.extend(self.examples[uri])
        
        if not examples:
            return "Use standard RDF patterns."
        
        return "\n".join(f"```turtle\n{ex}\n```" for ex in examples[:5])
    
    def get_pattern(self, pattern_name: str) -> Optional[str]:
        """Get a named RDF pattern template."""
        return self.patterns.get(pattern_name)
    
    def add_class(
        self,
        uri: str,
        label: str,
        description: str = "",
        superclass: Optional[str] = None,
    ) -> None:
        """Add a class to the library."""
        self.classes[uri] = {
            'uri': uri,
            'label': label,
            'description': description,
            'superclass': superclass,
        }
    
    def add_property(
        self,
        uri: str,
        label: str,
        description: str = "",
        domain: Optional[str] = None,
        range_type: Optional[str] = None,
    ) -> None:
        """Add a property to the library."""
        self.properties[uri] = {
            'uri': uri,
            'label': label,
            'description': description,
            'domain': domain,
            'range': range_type,
        }
    
    def add_prefix(self, prefix: str, namespace: str) -> None:
        """Add a prefix mapping."""
        self.prefixes[prefix] = namespace
    
    def add_example(self, uri: str, example: str) -> None:
        """Add a usage example for a term."""
        if uri not in self.examples:
            self.examples[uri] = []
        self.examples[uri].append(example)
    
    def add_pattern(self, name: str, template: str) -> None:
        """Add a named pattern template."""
        self.patterns[name] = template
    
    def save_to_cache(self, cache_dir: Path) -> None:
        """Save schema to cache files."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        schema = {
            'classes': self.classes,
            'properties': self.properties,
            'prefixes': self.prefixes,
            'examples': self.examples,
            'patterns': self.patterns,
        }
        
        with open(cache_dir / "schema.json", 'w') as f:
            json.dump(schema, f, indent=2)
        
        if self.embeddings is not None:
            np.save(cache_dir / "schema.npy", self.embeddings)
        
        config = {
            'uri_index': self.uri_index,
        }
        with open(cache_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)


def create_default_schema_library() -> SchemaLibrary:
    """
    Create a schema library with common vocabularies.
    
    Includes: Schema.org, Dublin Core, FOAF, SKOS, and common patterns.
    """
    lib = SchemaLibrary()
    
    # Common prefixes
    lib.add_prefix('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
    lib.add_prefix('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
    lib.add_prefix('xsd', 'http://www.w3.org/2001/XMLSchema#')
    lib.add_prefix('owl', 'http://www.w3.org/2002/07/owl#')
    lib.add_prefix('schema', 'https://schema.org/')
    lib.add_prefix('dc', 'http://purl.org/dc/elements/1.1/')
    lib.add_prefix('dcterms', 'http://purl.org/dc/terms/')
    lib.add_prefix('foaf', 'http://xmlns.com/foaf/0.1/')
    lib.add_prefix('skos', 'http://www.w3.org/2004/02/skos/core#')
    lib.add_prefix('wiki', 'https://en.wikipedia.org/wiki/')
    
    # Schema.org classes
    lib.add_class('https://schema.org/Person', 'Person', 'A person')
    lib.add_class('https://schema.org/Organization', 'Organization', 'An organization')
    lib.add_class('https://schema.org/Place', 'Place', 'A physical location')
    lib.add_class('https://schema.org/Event', 'Event', 'An event')
    lib.add_class('https://schema.org/CreativeWork', 'CreativeWork', 'A creative work')
    
    # Schema.org properties
    lib.add_property('https://schema.org/name', 'name', 'The name of the item',
                     domain='Thing', range_type='Text')
    lib.add_property('https://schema.org/birthDate', 'birthDate', 'Date of birth',
                     domain='Person', range_type='Date')
    lib.add_property('https://schema.org/deathDate', 'deathDate', 'Date of death',
                     domain='Person', range_type='Date')
    lib.add_property('https://schema.org/birthPlace', 'birthPlace', 'Place of birth',
                     domain='Person', range_type='Place')
    lib.add_property('https://schema.org/nationality', 'nationality', 'Nationality',
                     domain='Person', range_type='Country')
    lib.add_property('https://schema.org/award', 'award', 'An award received',
                     domain='Person', range_type='Text')
    lib.add_property('https://schema.org/worksFor', 'worksFor', 'Organization worked for',
                     domain='Person', range_type='Organization')
    lib.add_property('https://schema.org/alumniOf', 'alumniOf', 'Educational institution attended',
                     domain='Person', range_type='Organization')
    lib.add_property('https://schema.org/spouse', 'spouse', 'Spouse',
                     domain='Person', range_type='Person')
    lib.add_property('https://schema.org/children', 'children', 'Children',
                     domain='Person', range_type='Person')
    
    # RDF-Star pattern for temporal annotations
    lib.add_pattern('temporal', '''
# Pattern: Temporal annotation using RDF-Star
<< :subject :predicate :object >> :validFrom "YYYY-MM-DD"^^xsd:date ;
                                   :validUntil "YYYY-MM-DD"^^xsd:date .
''')
    
    # Role reification pattern
    lib.add_pattern('role', '''
# Pattern: Role reification (e.g., person in role at organization)
:roleInstance a :RoleType ;
    :agent :person ;
    :organization :org ;
    :startDate "YYYY-MM-DD"^^xsd:date ;
    :endDate "YYYY-MM-DD"^^xsd:date .
''')
    
    return lib
