"""Tests for DSPy signatures."""

import pytest
import dspy

from ontological_engineer.signatures import (
    ExtractStatements,
    JudgeStatementQuality,
    SelectSchemaTerms,
    GenerateRDF,
    JudgeTripleQuality,
)


class TestExtractStatements:
    """Tests for ExtractStatements signature."""
    
    def test_signature_has_required_input_fields(self):
        """Verify input fields are defined."""
        assert 'chunk_text' in ExtractStatements.model_fields
        assert 'section_context' in ExtractStatements.model_fields
    
    def test_signature_has_required_output_fields(self):
        """Verify output fields are defined."""
        assert 'statements' in ExtractStatements.model_fields
    
    def test_input_field_types(self):
        """Verify input field types are strings."""
        sig = ExtractStatements
        # Check that fields exist and have descriptions
        assert 'chunk_text' in sig.model_fields
        assert 'section_context' in sig.model_fields


class TestJudgeStatementQuality:
    """Tests for JudgeStatementQuality signature."""
    
    def test_signature_has_input_fields(self):
        """Verify input fields."""
        assert 'chunk_text' in JudgeStatementQuality.model_fields
        assert 'section_context' in JudgeStatementQuality.model_fields
        assert 'statements' in JudgeStatementQuality.model_fields
    
    def test_signature_has_output_fields(self):
        """Verify output fields for scoring."""
        assert 'completeness' in JudgeStatementQuality.model_fields
        assert 'atomicity' in JudgeStatementQuality.model_fields
        assert 'accuracy' in JudgeStatementQuality.model_fields
        assert 'link_preservation' in JudgeStatementQuality.model_fields
        assert 'reasoning' in JudgeStatementQuality.model_fields


class TestSelectSchemaTerms:
    """Tests for SelectSchemaTerms signature."""
    
    def test_signature_has_input_fields(self):
        """Verify input fields."""
        assert 'statements' in SelectSchemaTerms.model_fields
        assert 'candidate_classes' in SelectSchemaTerms.model_fields
        assert 'candidate_properties' in SelectSchemaTerms.model_fields
    
    def test_signature_has_output_fields(self):
        """Verify output fields."""
        assert 'selected_classes' in SelectSchemaTerms.model_fields
        assert 'selected_properties' in SelectSchemaTerms.model_fields
        assert 'custom_annotation_needs' in SelectSchemaTerms.model_fields


class TestGenerateRDF:
    """Tests for GenerateRDF signature."""
    
    def test_signature_has_input_fields(self):
        """Verify input fields."""
        assert 'statements' in GenerateRDF.model_fields
        assert 'schema_context' in GenerateRDF.model_fields
        assert 'entity_registry' in GenerateRDF.model_fields
    
    def test_signature_has_output_fields(self):
        """Verify output fields."""
        assert 'turtle_triples' in GenerateRDF.model_fields


class TestJudgeTripleQuality:
    """Tests for JudgeTripleQuality signature."""
    
    def test_signature_has_input_fields(self):
        """Verify input fields."""
        assert 'statements' in JudgeTripleQuality.model_fields
        assert 'turtle_triples' in JudgeTripleQuality.model_fields
        assert 'schema_context' in JudgeTripleQuality.model_fields
    
    def test_signature_has_output_fields(self):
        """Verify output fields."""
        assert 'syntax_valid' in JudgeTripleQuality.model_fields
        assert 'uris_correct' in JudgeTripleQuality.model_fields
        assert 'schema_conformance' in JudgeTripleQuality.model_fields
        assert 'completeness' in JudgeTripleQuality.model_fields
        assert 'reasoning' in JudgeTripleQuality.model_fields
