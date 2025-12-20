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


class TestStatementExtractorErrorHandling:
    """Tests for error handling in StatementExtractor."""
    
    def test_graceful_handling_without_lm(self):
        """Test that extraction handles errors gracefully."""
        # Save current LM config
        import dspy
        old_lm = dspy.settings.lm
        
        try:
            # Remove LM configuration to force an error
            dspy.configure(lm=None)
            
            extractor = StatementExtractor()
            result = extractor(
                chunk_text="Some text",
                section_context="Test > Section",
            )
            
            # Should not crash - returns graceful result with empty statements
            assert result.statements == []
            # Should have error info
            assert hasattr(result, 'error') or (hasattr(result, 'reasoning') and 'Error' in result.reasoning)
        finally:
            # Restore LM config
            if old_lm:
                dspy.configure(lm=old_lm)
    
    def test_error_result_has_empty_statements(self):
        """Test that error results have empty statements list."""
        import dspy
        old_lm = dspy.settings.lm
        
        try:
            dspy.configure(lm=None)
            extractor = StatementExtractor()
            result = extractor(
                chunk_text="Some text",
                section_context="Test > Section",
            )
            
            assert result.statements == []
            assert isinstance(result.statements, list)
        finally:
            if old_lm:
                dspy.configure(lm=old_lm)
