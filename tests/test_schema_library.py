"""Tests for schema library."""

import pytest
import json
import tempfile
from pathlib import Path

from ontological_engineer.schema_library import SchemaLibrary, create_default_schema_library


class TestSchemaLibrary:
    """Tests for SchemaLibrary class."""
    
    def test_initialization_empty(self):
        """Test empty initialization."""
        lib = SchemaLibrary()
        assert lib.classes == {}
        assert lib.properties == {}
        assert lib.prefixes == {}
    
    def test_add_class(self):
        """Test adding a class."""
        lib = SchemaLibrary()
        lib.add_class(
            uri='https://schema.org/Person',
            label='Person',
            description='A person',
        )
        
        assert 'https://schema.org/Person' in lib.classes
        cls = lib.classes['https://schema.org/Person']
        assert cls['label'] == 'Person'
        assert cls['description'] == 'A person'
    
    def test_add_property(self):
        """Test adding a property."""
        lib = SchemaLibrary()
        lib.add_property(
            uri='https://schema.org/name',
            label='name',
            description='The name',
            domain='Thing',
            range_type='Text',
        )
        
        assert 'https://schema.org/name' in lib.properties
        prop = lib.properties['https://schema.org/name']
        assert prop['label'] == 'name'
        assert prop['domain'] == 'Thing'
        assert prop['range'] == 'Text'
    
    def test_add_prefix(self):
        """Test adding a prefix."""
        lib = SchemaLibrary()
        lib.add_prefix('schema', 'https://schema.org/')
        
        assert lib.prefixes['schema'] == 'https://schema.org/'
    
    def test_add_example(self):
        """Test adding usage examples."""
        lib = SchemaLibrary()
        uri = 'https://schema.org/name'
        
        lib.add_example(uri, 'ex:thing schema:name "Example" .')
        lib.add_example(uri, 'ex:person schema:name "John" .')
        
        assert uri in lib.examples
        assert len(lib.examples[uri]) == 2
    
    def test_add_pattern(self):
        """Test adding named patterns."""
        lib = SchemaLibrary()
        lib.add_pattern('temporal', '<< :s :p :o >> :validFrom "2024" .')
        
        assert lib.get_pattern('temporal') == '<< :s :p :o >> :validFrom "2024" .'
        assert lib.get_pattern('nonexistent') is None
    
    def test_build_prefix_block(self):
        """Test building Turtle prefix declarations."""
        lib = SchemaLibrary()
        lib.add_prefix('schema', 'https://schema.org/')
        lib.add_prefix('foaf', 'http://xmlns.com/foaf/0.1/')
        lib.add_prefix('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
        
        block = lib.build_prefix_block(
            classes=['https://schema.org/Person'],
            properties=['http://xmlns.com/foaf/0.1/name'],
        )
        
        assert '@prefix' in block
        assert 'rdf:' in block  # Always included
    
    def test_format_classes_empty(self):
        """Test formatting empty class list."""
        lib = SchemaLibrary()
        result = lib.format_classes([])
        assert 'No specific classes' in result
    
    def test_format_classes_with_data(self):
        """Test formatting classes with data."""
        lib = SchemaLibrary()
        lib.add_class('https://schema.org/Person', 'Person', 'A person')
        
        result = lib.format_classes(['https://schema.org/Person'])
        assert 'Person' in result
        assert 'A person' in result
    
    def test_search_relevant_no_embeddings(self):
        """Test search falls back without embeddings."""
        lib = SchemaLibrary()
        lib.add_class('https://schema.org/Person', 'Person', 'A person')
        lib.add_property('https://schema.org/name', 'name', 'The name')
        
        result = lib.search_relevant(['Einstein was a physicist'], top_k=5)
        
        assert 'classes' in result
        assert 'properties' in result
    
    def test_save_and_load_cache(self):
        """Test saving and loading from cache."""
        lib = SchemaLibrary()
        lib.add_class('https://schema.org/Person', 'Person', 'A person')
        lib.add_property('https://schema.org/name', 'name', 'The name')
        lib.add_prefix('schema', 'https://schema.org/')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            lib.save_to_cache(cache_dir)
            
            # Verify files exist
            assert (cache_dir / 'schema.json').exists()
            assert (cache_dir / 'config.json').exists()
            
            # Load into new instance
            lib2 = SchemaLibrary(vocab_cache_dir=cache_dir)
            assert 'https://schema.org/Person' in lib2.classes
            assert 'https://schema.org/name' in lib2.properties
            assert lib2.prefixes['schema'] == 'https://schema.org/'


class TestCreateDefaultSchemaLibrary:
    """Tests for create_default_schema_library factory."""
    
    def test_creates_library(self):
        """Test factory creates a library."""
        lib = create_default_schema_library()
        assert isinstance(lib, SchemaLibrary)
    
    def test_has_common_prefixes(self):
        """Test library has common prefixes."""
        lib = create_default_schema_library()
        assert 'rdf' in lib.prefixes
        assert 'rdfs' in lib.prefixes
        assert 'xsd' in lib.prefixes
        assert 'schema' in lib.prefixes
        assert 'wiki' in lib.prefixes
    
    def test_has_schema_org_classes(self):
        """Test library has Schema.org classes."""
        lib = create_default_schema_library()
        assert 'https://schema.org/Person' in lib.classes
        assert 'https://schema.org/Organization' in lib.classes
        assert 'https://schema.org/Place' in lib.classes
    
    def test_has_schema_org_properties(self):
        """Test library has Schema.org properties."""
        lib = create_default_schema_library()
        assert 'https://schema.org/name' in lib.properties
        assert 'https://schema.org/birthDate' in lib.properties
        assert 'https://schema.org/birthPlace' in lib.properties
    
    def test_has_patterns(self):
        """Test library has RDF patterns."""
        lib = create_default_schema_library()
        assert lib.get_pattern('temporal') is not None
        assert lib.get_pattern('role') is not None
