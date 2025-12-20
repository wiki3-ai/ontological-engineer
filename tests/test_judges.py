"""Tests for quality judge modules."""

import pytest
import dspy

from ontological_engineer.judges import (
    StatementQualityJudge,
    StatementClassifier,
    StatementClassification,
    TripleQualityJudge,
    statement_quality_metric,
    statement_classification_metric,
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
    
    def test_statement_classification_metric_signature(self):
        """Test classification metric function has correct signature."""
        import inspect
        sig = inspect.signature(statement_classification_metric)
        params = list(sig.parameters.keys())
        assert 'example' in params
        assert 'pred' in params
        assert 'trace' in params


# =============================================================================
# StatementClassification Tests
# =============================================================================

class TestStatementClassification:
    """Tests for StatementClassification dataclass."""
    
    def test_creation(self):
        """Test creating a classification object."""
        c = StatementClassification(
            index=0,
            statement="Test statement",
            classification="GOOD",
            reason="Well formed"
        )
        assert c.index == 0
        assert c.statement == "Test statement"
        assert c.classification == "GOOD"
        assert c.reason == "Well formed"
    
    def test_is_good_true(self):
        """Test is_good property returns True for GOOD classification."""
        c = StatementClassification(
            index=0,
            statement="Test",
            classification="GOOD",
            reason="OK"
        )
        assert c.is_good is True
    
    def test_is_good_false(self):
        """Test is_good property returns False for BAD classification."""
        c = StatementClassification(
            index=0,
            statement="Test",
            classification="BAD",
            reason="Not OK"
        )
        assert c.is_good is False
    
    def test_is_good_case_insensitive(self):
        """Test is_good handles case variations."""
        c1 = StatementClassification(0, "Test", "good", "OK")
        c2 = StatementClassification(0, "Test", "Good", "OK")
        c3 = StatementClassification(0, "Test", "GOOD", "OK")
        assert c1.is_good is True
        assert c2.is_good is True
        assert c3.is_good is True


# =============================================================================
# StatementClassifier Tests
# =============================================================================

class TestStatementClassifier:
    """Tests for StatementClassifier module."""
    
    def test_initialization(self):
        """Test module initializes correctly."""
        classifier = StatementClassifier()
        assert classifier is not None
        assert hasattr(classifier, 'classifier')
    
    def test_is_dspy_module(self):
        """Verify it's a proper DSPy module."""
        classifier = StatementClassifier()
        assert isinstance(classifier, dspy.Module)
    
    def test_parse_classifications_good(self):
        """Test parsing GOOD classifications."""
        classifier = StatementClassifier()
        statements = ["Statement A", "Statement B"]
        raw_output = "0: GOOD - atomic and accurate\n1: GOOD - well formed"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 2
        assert result[0].index == 0
        assert result[0].classification == "GOOD"
        assert result[0].statement == "Statement A"
        assert "atomic" in result[0].reason
        assert result[1].index == 1
        assert result[1].classification == "GOOD"
    
    def test_parse_classifications_bad(self):
        """Test parsing BAD classifications."""
        classifier = StatementClassifier()
        statements = ["Statement A", "Statement B"]
        raw_output = "0: BAD - multiple claims\n1: BAD - missing links"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 2
        assert result[0].classification == "BAD"
        assert result[1].classification == "BAD"
        assert "multiple claims" in result[0].reason
    
    def test_parse_classifications_mixed(self):
        """Test parsing mixed GOOD/BAD classifications."""
        classifier = StatementClassifier()
        statements = ["Good one", "Bad one", "Another good"]
        raw_output = "0: GOOD - correct\n1: BAD - wrong\n2: GOOD - also correct"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 3
        assert result[0].is_good is True
        assert result[1].is_good is False
        assert result[2].is_good is True
    
    def test_parse_classifications_handles_missing(self):
        """Test that missing indices are marked as BAD."""
        classifier = StatementClassifier()
        statements = ["A", "B", "C"]
        # Only classify index 0 and 2, missing 1
        raw_output = "0: GOOD - ok\n2: GOOD - ok"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 3
        # Should be sorted by index
        assert result[0].index == 0
        assert result[0].is_good is True
        assert result[1].index == 1
        assert result[1].is_good is False  # Missing = BAD
        assert "Not explicitly classified" in result[1].reason
        assert result[2].index == 2
        assert result[2].is_good is True
    
    def test_parse_classifications_case_insensitive(self):
        """Test parsing handles case variations."""
        classifier = StatementClassifier()
        statements = ["A", "B"]
        raw_output = "0: good - ok\n1: Bad - not ok"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert result[0].classification == "GOOD"
        assert result[1].classification == "BAD"
    
    def test_parse_classifications_various_separators(self):
        """Test parsing handles different separator styles."""
        classifier = StatementClassifier()
        statements = ["A", "B", "C"]
        # Mix of separators: dash, en-dash, colon
        raw_output = "0: GOOD - reason one\n1: BAD – reason two\n2: GOOD: reason three"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 3
        assert result[0].is_good is True
        assert result[1].is_good is False
        assert result[2].is_good is True
    
    def test_parse_classifications_empty_statements(self):
        """Test parsing with empty statement list."""
        classifier = StatementClassifier()
        statements = []
        raw_output = ""
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert result == []
    
    def test_parse_classifications_blank_lines(self):
        """Test parsing ignores blank lines."""
        classifier = StatementClassifier()
        statements = ["A", "B"]
        raw_output = "0: GOOD - ok\n\n\n1: BAD - not ok\n"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        assert len(result) == 2
    
    def test_parse_classifications_out_of_range_index(self):
        """Test handling of out-of-range indices."""
        classifier = StatementClassifier()
        statements = ["A"]  # Only one statement
        raw_output = "0: GOOD - ok\n5: GOOD - out of range"
        
        result = classifier._parse_classifications(raw_output, statements)
        
        # Should have index 0 and index 5 (with empty statement)
        indices = {c.index for c in result}
        assert 0 in indices
        assert 5 in indices
        # Index 5 should have empty statement since out of range
        idx5 = next(c for c in result if c.index == 5)
        assert idx5.statement == ""
