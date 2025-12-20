"""
RDF Generator Module.

DSPy module for generating RDF triples from statements using schema context.
"""

from typing import List, Dict, Any, Optional
import dspy

from ontological_engineer.signatures import GenerateRDF


class RDFGenerator(dspy.Module):
    """
    Generate RDF triples from statements.
    
    This module converts extracted statements into valid Turtle RDF,
    using schema context prepared by SchemaContextBuilder.
    
    Features:
    - Converts [Entity](/wiki/...) links to Wikipedia URIs
    - Uses provided prefixes and property definitions
    - Applies temporal/reification patterns where appropriate
    - Outputs valid Turtle syntax
    """
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateRDF)
    
    def forward(
        self,
        statements: List[str],
        schema_context: str,
        entity_registry: str,
    ) -> dspy.Prediction:
        """
        Generate RDF from statements.
        
        Args:
            statements: Statements to convert to RDF
            schema_context: Formatted schema context (prefixes, definitions, examples)
            entity_registry: JSON string of known entity URIs
            
        Returns:
            Prediction with turtle_triples field
        """
        result = self.generate(
            statements=statements,
            schema_context=schema_context,
            entity_registry=entity_registry,
        )
        
        return dspy.Prediction(
            turtle_triples=result.turtle_triples,
            reasoning=getattr(result, 'reasoning', None),
        )
    
    def validate_turtle(self, turtle: str) -> bool:
        """
        Validate Turtle syntax using rdflib.
        
        Args:
            turtle: Turtle string to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            from rdflib import Graph
            g = Graph()
            g.parse(data=turtle, format='turtle')
            return True
        except Exception:
            return False


class BatchRDFGenerator(dspy.Module):
    """
    Generate RDF for multiple statement sets efficiently.
    """
    
    def __init__(self):
        super().__init__()
        self.generator = RDFGenerator()
    
    def forward(
        self,
        statement_batches: List[List[str]],
        schema_context: str,
        entity_registry: str,
    ) -> List[dspy.Prediction]:
        """
        Generate RDF for multiple batches of statements.
        
        Args:
            statement_batches: List of statement lists
            schema_context: Shared schema context
            entity_registry: Shared entity registry
            
        Returns:
            List of Predictions, one per batch
        """
        results = []
        for statements in statement_batches:
            result = self.generator(
                statements=statements,
                schema_context=schema_context,
                entity_registry=entity_registry,
            )
            results.append(result)
        return results
