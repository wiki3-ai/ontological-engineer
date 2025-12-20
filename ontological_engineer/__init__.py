"""
Wiki3.ai Ontological Engineer

A DSPy-based pipeline for extracting knowledge graphs from Wikipedia articles.
Uses LLM judges and GRPO optimization for high-quality RDF generation.

Repository: https://github.com/wiki3-ai/ontological-engineer
"""

from ontological_engineer.config import (
    configure_lm,
    get_default_lm,
    get_preset_lm,
    get_arbor_lm,
)
from ontological_engineer.signatures import (
    ExtractStatements,
    JudgeStatementQuality,
    SelectSchemaTerms,
    GenerateRDF,
    JudgeTripleQuality,
)
from ontological_engineer.extractors import StatementExtractor, BatchStatementExtractor
from ontological_engineer.judges import (
    StatementQualityJudge,
    TripleQualityJudge,
    statement_quality_metric,
    triple_quality_metric,
)
from ontological_engineer.schema_context import SchemaContextBuilder
from ontological_engineer.schema_library import SchemaLibrary, create_default_schema_library
from ontological_engineer.rdf_generator import RDFGenerator, BatchRDFGenerator

__version__ = "0.1.0"
__all__ = [
    # Config
    "configure_lm",
    "get_default_lm",
    "get_preset_lm",
    "get_arbor_lm",
    # Signatures
    "ExtractStatements",
    "JudgeStatementQuality",
    "SelectSchemaTerms",
    "GenerateRDF",
    "JudgeTripleQuality",
    # Stage 1: Statement Extraction
    "StatementExtractor",
    "BatchStatementExtractor",
    "StatementQualityJudge",
    "statement_quality_metric",
    # Stage 2: Schema Context
    "SchemaContextBuilder",
    "SchemaLibrary",
    "create_default_schema_library",
    # Stage 3: RDF Generation
    "RDFGenerator",
    "BatchRDFGenerator",
    "TripleQualityJudge",
    "triple_quality_metric",
]
