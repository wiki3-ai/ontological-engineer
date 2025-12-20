"""Tests for statement extractor modules."""

import pytest
import dspy

from ontological_engineer.extractors import StatementExtractor, BatchStatementExtractor


class TestStatementExtractor:
    """Tests for StatementExtractor module."""
    
    def test_initialization(self):
        """Test module can be instantiated."""
        extractor = StatementExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'extract')
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        extractor = StatementExtractor()
        assert isinstance(extractor, dspy.Module)
    
    def test_has_forward_method(self):
        """Verify forward method exists with correct signature."""
        extractor = StatementExtractor()
        assert callable(extractor.forward)
    
    def test_parse_statements_from_list_markers(self):
        """Test parsing statements from various list formats."""
        extractor = StatementExtractor()
        
        # Test dash markers
        text = "- Statement one\n- Statement two"
        result = extractor._parse_statements(text)
        assert result == ["Statement one", "Statement two"]
        
        # Test asterisk markers
        text = "* Statement one\n* Statement two"
        result = extractor._parse_statements(text)
        assert result == ["Statement one", "Statement two"]
        
        # Test numbered markers
        text = "1. Statement one\n2. Statement two"
        result = extractor._parse_statements(text)
        assert result == ["Statement one", "Statement two"]
    
    def test_parse_statements_filters_empty(self):
        """Test that empty lines are filtered."""
        extractor = StatementExtractor()
        text = "- Statement one\n\n- Statement two\n"
        result = extractor._parse_statements(text)
        assert result == ["Statement one", "Statement two"]


class TestBatchStatementExtractor:
    """Tests for BatchStatementExtractor module."""
    
    def test_initialization(self):
        """Test module can be instantiated."""
        extractor = BatchStatementExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'extractor')
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        extractor = BatchStatementExtractor()
        assert isinstance(extractor, dspy.Module)
