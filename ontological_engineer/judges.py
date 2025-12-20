"""
Quality Judge Modules.

DSPy modules for evaluating the quality of extracted statements and generated RDF.
These judges provide reward signals for GRPO optimization.
"""

from typing import List, Optional, Dict, Any
import dspy

from ontological_engineer.signatures import JudgeStatementQuality, JudgeTripleQuality


class StatementQualityJudge(dspy.Module):
    """
    Judge the quality of extracted statements.
    
    Evaluates statements on four dimensions:
    - Completeness: Are all key facts captured?
    - Atomicity: Is each statement a single claim?
    - Accuracy: Do statements faithfully represent the source?
    - Link preservation: Are entity links preserved correctly?
    
    Returns a weighted score suitable for use as GRPO reward.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the judge.
        
        Args:
            weights: Dict mapping dimension names to weights.
                     Default: equal weights (0.25 each)
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeStatementQuality)
        
        self.weights = weights or {
            'completeness': 0.25,
            'atomicity': 0.25,
            'accuracy': 0.25,
            'link_preservation': 0.25,
        }
    
    def forward(
        self,
        chunk_text: str,
        section_context: str,
        statements: List[str],
    ) -> dspy.Prediction:
        """
        Judge the quality of extracted statements.
        
        Args:
            chunk_text: Original Wikipedia chunk
            section_context: Section breadcrumb
            statements: Extracted statements to evaluate
            
        Returns:
            Prediction with individual scores and weighted total
        """
        result = self.judge(
            chunk_text=chunk_text,
            section_context=section_context,
            statements=statements,
        )
        
        # Parse scores (handle both float and string returns)
        completeness = self._parse_score(result.completeness)
        atomicity = self._parse_score(result.atomicity)
        accuracy = self._parse_score(result.accuracy)
        link_preservation = self._parse_score(result.link_preservation)
        
        # Compute weighted score
        weighted_score = (
            self.weights['completeness'] * completeness +
            self.weights['atomicity'] * atomicity +
            self.weights['accuracy'] * accuracy +
            self.weights['link_preservation'] * link_preservation
        )
        
        return dspy.Prediction(
            completeness=completeness,
            atomicity=atomicity,
            accuracy=accuracy,
            link_preservation=link_preservation,
            weighted_score=weighted_score,
            reasoning=result.reasoning,
        )
    
    def _parse_score(self, value: Any) -> float:
        """Parse a score value to float in range [0, 1]."""
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        
        # Try to parse from string
        try:
            score = float(str(value).strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            # Default to middle score if parsing fails
            return 0.5


class TripleQualityJudge(dspy.Module):
    """
    Judge the quality of generated RDF triples.
    
    Evaluates triples on:
    - Syntax validity: Is the Turtle syntax correct?
    - URI correctness: Are URIs properly formed?
    - Schema conformance: Are appropriate terms used?
    - Completeness: Are all statements represented?
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the judge.
        
        Args:
            weights: Dict mapping dimension names to weights.
                     Default: syntax/URIs weighted higher as they're critical
        """
        super().__init__()
        self.judge = dspy.ChainOfThought(JudgeTripleQuality)
        
        self.weights = weights or {
            'syntax_valid': 0.3,
            'uris_correct': 0.3,
            'schema_conformance': 0.2,
            'completeness': 0.2,
        }
    
    def forward(
        self,
        statements: List[str],
        turtle_triples: str,
        schema_context: str,
    ) -> dspy.Prediction:
        """
        Judge the quality of generated RDF.
        
        Args:
            statements: Source statements that were converted
            turtle_triples: Generated RDF in Turtle format
            schema_context: Schema terms that were available
            
        Returns:
            Prediction with individual scores and weighted total
        """
        result = self.judge(
            statements=statements,
            turtle_triples=turtle_triples,
            schema_context=schema_context,
        )
        
        # Parse boolean and float scores
        syntax_valid = self._parse_bool(result.syntax_valid)
        uris_correct = self._parse_bool(result.uris_correct)
        schema_conformance = self._parse_score(result.schema_conformance)
        completeness = self._parse_score(result.completeness)
        
        # Compute weighted score (bools become 1.0 or 0.0)
        weighted_score = (
            self.weights['syntax_valid'] * (1.0 if syntax_valid else 0.0) +
            self.weights['uris_correct'] * (1.0 if uris_correct else 0.0) +
            self.weights['schema_conformance'] * schema_conformance +
            self.weights['completeness'] * completeness
        )
        
        return dspy.Prediction(
            syntax_valid=syntax_valid,
            uris_correct=uris_correct,
            schema_conformance=schema_conformance,
            completeness=completeness,
            weighted_score=weighted_score,
            reasoning=result.reasoning,
        )
    
    def _parse_bool(self, value: Any) -> bool:
        """Parse a boolean value."""
        if isinstance(value, bool):
            return value
        
        str_val = str(value).lower().strip()
        return str_val in ('true', 'yes', '1', 'valid')
    
    def _parse_score(self, value: Any) -> float:
        """Parse a score value to float in range [0, 1]."""
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        
        try:
            score = float(str(value).strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5


# =============================================================================
# Metric Functions for DSPy Evaluation / GRPO
# =============================================================================

def statement_quality_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
) -> float:
    """
    Compute quality metric for statement extraction.
    
    This function is used by DSPy optimizers (MIPROv2, GRPO) as the
    reward/metric function.
    
    Args:
        example: DSPy Example with inputs (chunk_text, section_context)
        pred: Prediction with extracted statements
        trace: Optional trace information (unused)
        
    Returns:
        Float score in range [0, 1]
    """
    judge = StatementQualityJudge()
    
    result = judge(
        chunk_text=example.chunk_text,
        section_context=example.section_context,
        statements=pred.statements,
    )
    
    return result.weighted_score


def triple_quality_metric(
    example: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
) -> float:
    """
    Compute quality metric for RDF generation.
    
    Args:
        example: DSPy Example with inputs (statements, schema_context)
        pred: Prediction with generated turtle_triples
        trace: Optional trace information (unused)
        
    Returns:
        Float score in range [0, 1]
    """
    judge = TripleQualityJudge()
    
    result = judge(
        statements=example.statements,
        turtle_triples=pred.turtle_triples,
        schema_context=example.schema_context,
    )
    
    return result.weighted_score
