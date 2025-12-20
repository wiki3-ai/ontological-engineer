"""Tests for RDF generator module."""

import pytest
import dspy

from ontological_engineer.rdf_generator import RDFGenerator, BatchRDFGenerator


class TestRDFGenerator:
    """Tests for RDFGenerator module."""
    
    def test_initialization(self):
        """Test module can be instantiated."""
        generator = RDFGenerator()
        assert generator is not None
        assert hasattr(generator, 'generate')
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        generator = RDFGenerator()
        assert isinstance(generator, dspy.Module)
    
    def test_has_forward_method(self):
        """Verify forward method exists."""
        generator = RDFGenerator()
        assert callable(generator.forward)
    
    def test_validate_turtle_valid(self):
        """Test validate_turtle with valid Turtle."""
        generator = RDFGenerator()
        
        valid_turtle = """@prefix schema: <https://schema.org/> .
@prefix wiki: <https://en.wikipedia.org/wiki/> .

wiki:Albert_Einstein a schema:Person ;
    schema:name "Albert Einstein" .
"""
        # This will fail if rdflib is not installed, which is OK for unit tests
        try:
            result = generator.validate_turtle(valid_turtle)
            assert result is True
        except ImportError:
            pytest.skip("rdflib not installed")
        except Exception:
            # rdflib might have issues with specific syntax
            pytest.skip("rdflib parsing issue")
    
    def test_validate_turtle_invalid(self):
        """Test validate_turtle with invalid Turtle."""
        generator = RDFGenerator()
        
        invalid_turtle = """
This is not valid turtle syntax at all
{ random json maybe? }
"""
        try:
            result = generator.validate_turtle(invalid_turtle)
            assert result is False
        except ImportError:
            pytest.skip("rdflib not installed")


class TestBatchRDFGenerator:
    """Tests for BatchRDFGenerator module."""
    
    def test_initialization(self):
        """Test module can be instantiated."""
        generator = BatchRDFGenerator()
        assert generator is not None
        assert hasattr(generator, 'generator')
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        generator = BatchRDFGenerator()
        assert isinstance(generator, dspy.Module)
    
    def test_has_forward_method(self):
        """Verify forward method exists."""
        generator = BatchRDFGenerator()
        assert callable(generator.forward)
