"""
Schema Context Builder.

DSPy module for preparing schema context before RDF generation.
Uses embedding-based search followed by LLM selection to identify
relevant classes and properties.
"""

from typing import List, Dict, Any, Optional
import dspy

from ontological_engineer.signatures import SelectSchemaTerms


class SchemaContextBuilder(dspy.Module):
    """
    Build schema context for RDF generation.
    
    This module prepares all necessary schema information BEFORE
    the RDF generator runs, eliminating the need for tool calls
    during generation.
    
    Process:
    1. Embedding-based retrieval of candidate terms from schema library
    2. LLM selection of applicable classes and properties
    3. Format context for RDF generator prompt
    """
    
    def __init__(self, schema_library: Optional['SchemaLibrary'] = None):
        """
        Initialize the context builder.
        
        Args:
            schema_library: SchemaLibrary instance for term lookup.
                           If None, will need to be provided at forward() time.
        """
        super().__init__()
        self.schema_library = schema_library
        self.select_terms = dspy.ChainOfThought(SelectSchemaTerms)
    
    def forward(
        self,
        statements: List[str],
        schema_library: Optional['SchemaLibrary'] = None,
        top_k: int = 20,
    ) -> dspy.Prediction:
        """
        Build schema context for a set of statements.
        
        Args:
            statements: Statements to be converted to RDF
            schema_library: Optional override for schema library
            top_k: Number of candidate terms to retrieve
            
        Returns:
            Prediction with prefixes, definitions, examples, and patterns
        """
        lib = schema_library or self.schema_library
        if lib is None:
            raise ValueError("No schema_library provided")
        
        # Step 1: Retrieve candidate terms via embeddings
        candidates = lib.search_relevant(statements, top_k=top_k)
        
        # Step 2: LLM selects which terms are actually needed
        import json
        selection = self.select_terms(
            statements=statements,
            candidate_classes=json.dumps(candidates.get('classes', []), indent=2),
            candidate_properties=json.dumps(candidates.get('properties', []), indent=2),
        )
        
        # Step 3: Build context structure
        selected_classes = selection.selected_classes or []
        selected_properties = selection.selected_properties or []
        
        context = {
            'prefixes': lib.build_prefix_block(selected_classes, selected_properties),
            'class_definitions': lib.format_classes(selected_classes),
            'property_definitions': lib.format_properties(selected_properties),
            'usage_examples': lib.get_examples(selected_classes, selected_properties),
            'custom_annotation_needs': selection.custom_annotation_needs,
        }
        
        return dspy.Prediction(
            schema_context=context,
            selected_classes=selected_classes,
            selected_properties=selected_properties,
            custom_needs=selection.custom_annotation_needs,
        )
    
    def format_as_prompt_context(self, context: Dict[str, Any]) -> str:
        """
        Format schema context as text for the RDF generator prompt.
        
        Args:
            context: Schema context dict from forward()
            
        Returns:
            Formatted string for prompt inclusion
        """
        parts = []
        
        if context.get('prefixes'):
            parts.append("## Prefixes\n" + context['prefixes'])
        
        if context.get('class_definitions'):
            parts.append("## Classes\n" + context['class_definitions'])
        
        if context.get('property_definitions'):
            parts.append("## Properties\n" + context['property_definitions'])
        
        if context.get('usage_examples'):
            parts.append("## Examples\n" + context['usage_examples'])
        
        if context.get('custom_annotation_needs'):
            parts.append("## Notes\n" + context['custom_annotation_needs'])
        
        return "\n\n".join(parts)
