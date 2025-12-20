"""Tests for quality judge modules."""

import pytest
import dspy

from ontological_engineer.judges import (
    StatementQualityJudge,
    TripleQualityJudge,
    statement_quality_metric,
    triple_quality_metric,
)


class TestStatementQualityJudge:
    """Tests for StatementQualityJudge module."""
    
    def test_initialization_default_weights(self):
        """Test module initializes with default weights."""
        judge = StatementQualityJudge()
        assert judge is not None
        assert judge.weights['completeness'] == 0.25
        assert judge.weights['atomicity'] == 0.25
        assert judge.weights['accuracy'] == 0.25
        assert judge.weights['link_preservation'] == 0.25
    
    def test_initialization_custom_weights(self):
        """Test module accepts custom weights."""
        weights = {
            'completeness': 0.4,
            'atomicity': 0.3,
            'accuracy': 0.2,
            'link_preservation': 0.1,
        }
        judge = StatementQualityJudge(weights=weights)
        assert judge.weights == weights
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        judge = StatementQualityJudge()
        assert isinstance(judge, dspy.Module)
    
    def test_parse_score_from_float(self):
        """Test score parsing from float values."""
        judge = StatementQualityJudge()
        assert judge._parse_score(0.5) == 0.5
        assert judge._parse_score(1.0) == 1.0
        assert judge._parse_score(0.0) == 0.0
    
    def test_parse_score_clamps_values(self):
        """Test score parsing clamps to [0, 1] range."""
        judge = StatementQualityJudge()
        assert judge._parse_score(1.5) == 1.0
        assert judge._parse_score(-0.5) == 0.0
    
    def test_parse_score_from_string(self):
        """Test score parsing from string values."""
        judge = StatementQualityJudge()
        assert judge._parse_score("0.75") == 0.75
        assert judge._parse_score(" 0.5 ") == 0.5
    
    def test_parse_score_invalid_returns_default(self):
        """Test invalid values return 0.5 default."""
        judge = StatementQualityJudge()
        assert judge._parse_score("invalid") == 0.5
        assert judge._parse_score(None) == 0.5


class TestTripleQualityJudge:
    """Tests for TripleQualityJudge module."""
    
    def test_initialization_default_weights(self):
        """Test module initializes with default weights."""
        judge = TripleQualityJudge()
        assert judge is not None
        assert judge.weights['syntax_valid'] == 0.3
        assert judge.weights['uris_correct'] == 0.3
        assert judge.weights['schema_conformance'] == 0.2
        assert judge.weights['completeness'] == 0.2
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        judge = TripleQualityJudge()
        assert isinstance(judge, dspy.Module)
    
    def test_parse_bool_from_bool(self):
        """Test bool parsing from bool values."""
        judge = TripleQualityJudge()
        assert judge._parse_bool(True) is True
        assert judge._parse_bool(False) is False
    
    def test_parse_bool_from_string(self):
        """Test bool parsing from string values."""
        judge = TripleQualityJudge()
        assert judge._parse_bool("true") is True
        assert judge._parse_bool("True") is True
        assert judge._parse_bool("yes") is True
        assert judge._parse_bool("valid") is True
        assert judge._parse_bool("false") is False
        assert judge._parse_bool("no") is False


class TestMetricFunctions:
    """Tests for metric functions used in DSPy optimization."""
    
    def test_statement_quality_metric_signature(self):
        """Test metric function has correct signature."""
        import inspect
        sig = inspect.signature(statement_quality_metric)
        params = list(sig.parameters.keys())
        assert 'example' in params
        assert 'pred' in params
        assert 'trace' in params
    
    def test_triple_quality_metric_signature(self):
        """Test metric function has correct signature."""
        import inspect
        sig = inspect.signature(triple_quality_metric)
        params = list(sig.parameters.keys())
        assert 'example' in params
        assert 'pred' in params
        assert 'trace' in params
