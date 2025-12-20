"""
Statement Extractor Module.

DSPy module for extracting atomic, verifiable statements from Wikipedia chunks.
Uses ChainOfThought for reasoning through the extraction process.
"""

import logging
from typing import List, Optional
import dspy

from ontological_engineer.signatures import ExtractStatements


logger = logging.getLogger(__name__)


class StatementExtractor(dspy.Module):
    """
    Extract atomic statements from Wikipedia text chunks.
    
    This module uses ChainOfThought to reason through the extraction,
    ensuring each statement is self-contained, atomic, and preserves
    entity links in markdown format.
    
    Example usage:
        >>> extractor = StatementExtractor()
        >>> result = extractor(
        ...     chunk_text="Albert Einstein was born in Ulm on 14 March 1879.",
        ...     section_context="Albert Einstein > Early life"
        ... )
        >>> print(result.statements)
        ['[Albert Einstein](/wiki/Albert_Einstein) was born in [Ulm](/wiki/Ulm).',
         '[Albert Einstein](/wiki/Albert_Einstein) was born on 14 March 1879.']
    """
    
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractStatements)
    
    def forward(
        self,
        chunk_text: str,
        section_context: str,
    ) -> dspy.Prediction:
        """
        Extract statements from a Wikipedia chunk.
        
        Args:
            chunk_text: The Wikipedia article chunk with markdown links
            section_context: Breadcrumb showing location (Article > Section > ...)
            
        Returns:
            dspy.Prediction with 'statements' field containing list of strings
        """
        try:
            result = self.extract(
                chunk_text=chunk_text,
                section_context=section_context,
            )
        except Exception as e:
            # Handle LLM errors gracefully - return empty result with error info
            logger.warning(
                f"Extraction failed for {section_context[:50]}...: {type(e).__name__}: {e}"
            )
            return dspy.Prediction(
                statements=[],
                reasoning=f"Error during extraction: {type(e).__name__}: {e}",
                error=str(e),
            )
        
        # Ensure we return a list even if the model returns something else
        statements = result.statements
        if isinstance(statements, str):
            # Try to parse as list if it's a string representation
            statements = self._parse_statements(statements)
        
        return dspy.Prediction(
            statements=statements,
            reasoning=getattr(result, 'reasoning', None),
        )
    
    def _parse_statements(self, text: str) -> List[str]:
        """Parse statements from a string representation."""
        # Handle various formats the model might return
        lines = text.strip().split('\n')
        statements = []
        
        for line in lines:
            line = line.strip()
            # Remove list markers
            if line.startswith('- '):
                line = line[2:]
            elif line.startswith('* '):
                line = line[2:]
            elif len(line) > 2 and line[0].isdigit() and line[1] in '.):':
                line = line[2:].strip()
            
            if line:
                statements.append(line)
        
        return statements


class BatchStatementExtractor(dspy.Module):
    """
    Extract statements from multiple chunks efficiently.
    
    Useful for processing an entire article's worth of chunks.
    """
    
    def __init__(self):
        super().__init__()
        self.extractor = StatementExtractor()
    
    def forward(
        self,
        chunks: List[dict],
    ) -> List[dspy.Prediction]:
        """
        Extract statements from multiple chunks.
        
        Args:
            chunks: List of dicts with 'text' and 'section_context' keys
            
        Returns:
            List of Predictions, one per chunk
        """
        results = []
        for chunk in chunks:
            result = self.extractor(
                chunk_text=chunk['text'],
                section_context=chunk.get('section_context', ''),
            )
            results.append(result)
        return results
