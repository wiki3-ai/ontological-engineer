"""Tests for schema context builder module."""

import pytest
import dspy

from ontological_engineer.schema_context import SchemaContextBuilder
from ontological_engineer.schema_library import SchemaLibrary, create_default_schema_library


class TestSchemaContextBuilder:
    """Tests for SchemaContextBuilder module."""
    
    def test_initialization_without_library(self):
        """Test module can be instantiated without library."""
        builder = SchemaContextBuilder()
        assert builder is not None
        assert builder.schema_library is None
    
    def test_initialization_with_library(self):
        """Test module can be instantiated with library."""
        lib = create_default_schema_library()
        builder = SchemaContextBuilder(schema_library=lib)
        assert builder.schema_library is lib
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        builder = SchemaContextBuilder()
        assert isinstance(builder, dspy.Module)
    
    def test_forward_raises_without_library(self):
        """Test forward raises error when no library provided."""
        builder = SchemaContextBuilder()
        
        with pytest.raises(ValueError) as exc_info:
            builder.forward(statements=["Test statement"])
        
        assert "No schema_library" in str(exc_info.value)
    
    def test_format_as_prompt_context_empty(self):
        """Test formatting empty context."""
        builder = SchemaContextBuilder()
        
        context = {}
        result = builder.format_as_prompt_context(context)
        
        assert result == ""
    
    def test_format_as_prompt_context_with_prefixes(self):
        """Test formatting context with prefixes."""
        builder = SchemaContextBuilder()
        
        context = {
            'prefixes': '@prefix schema: <https://schema.org/> .',
        }
        result = builder.format_as_prompt_context(context)
        
        assert '## Prefixes' in result
        assert '@prefix schema' in result
    
    def test_format_as_prompt_context_with_classes(self):
        """Test formatting context with classes."""
        builder = SchemaContextBuilder()
        
        context = {
            'class_definitions': '- Person: A person',
        }
        result = builder.format_as_prompt_context(context)
        
        assert '## Classes' in result
        assert 'Person' in result
    
    def test_format_as_prompt_context_with_properties(self):
        """Test formatting context with properties."""
        builder = SchemaContextBuilder()
        
        context = {
            'property_definitions': '- name: The name of an item',
        }
        result = builder.format_as_prompt_context(context)
        
        assert '## Properties' in result
        assert 'name' in result
    
    def test_format_as_prompt_context_with_examples(self):
        """Test formatting context with examples."""
        builder = SchemaContextBuilder()
        
        context = {
            'usage_examples': '```turtle\nex:thing a schema:Thing .\n```',
        }
        result = builder.format_as_prompt_context(context)
        
        assert '## Examples' in result
        assert 'turtle' in result
    
    def test_format_as_prompt_context_with_notes(self):
        """Test formatting context with custom notes."""
        builder = SchemaContextBuilder()
        
        context = {
            'custom_annotation_needs': 'Use temporal annotations for dates.',
        }
        result = builder.format_as_prompt_context(context)
        
        assert '## Notes' in result
        assert 'temporal' in result
    
    def test_format_as_prompt_context_full(self):
        """Test formatting full context."""
        builder = SchemaContextBuilder()
        
        context = {
            'prefixes': '@prefix schema: <https://schema.org/> .',
            'class_definitions': '- Person: A person',
            'property_definitions': '- name: The name',
            'usage_examples': 'Example turtle code',
            'custom_annotation_needs': 'Use RDF-Star',
        }
        result = builder.format_as_prompt_context(context)
        
        # Check all sections are present and in order
        assert '## Prefixes' in result
        assert '## Classes' in result
        assert '## Properties' in result
        assert '## Examples' in result
        assert '## Notes' in result
        
        # Check sections are separated
        sections = result.split('\n\n')
        assert len(sections) == 5
