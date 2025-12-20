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
    ClassifyStatements,
    SelectSchemaTerms,
    GenerateRDF,
    JudgeTripleQuality,
)
from ontological_engineer.extractors import StatementExtractor, BatchStatementExtractor
from ontological_engineer.judges import (
    StatementQualityJudge,
    StatementClassifier,
    StatementClassification,
    TripleQualityJudge,
    statement_quality_metric,
    statement_classification_metric,
    triple_quality_metric,
)
from ontological_engineer.schema_context import SchemaContextBuilder
from ontological_engineer.schema_library import SchemaLibrary, create_default_schema_library
from ontological_engineer.rdf_generator import RDFGenerator, BatchRDFGenerator
from ontological_engineer.provenance import (
    generate_statements_notebook_header,
    append_statements_cell,
    save_statements_notebook,
    load_statements_from_notebook,
    generate_classifications_notebook_header,
    append_classifications_cell,
    save_classifications_notebook,
    load_classifications_from_notebook,
    get_processed_chunk_cids,
    create_output_directory,
)
from ontological_engineer.training.data import (
    WikipediaPage,
    WikipediaChunk,
    fetch_top_pages,
    fetch_page_content,
    chunk_article,
    generate_sample_notebook_header,
    append_sample_page_cell,
    generate_chunks_notebook_header,
    append_chunk_cell,
    load_sample_from_notebook,
    load_chunks_from_notebook,
    get_processed_page_titles,
    save_notebook,
    fetch_and_cache_pages,
)

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
    "ClassifyStatements",
    "SelectSchemaTerms",
    "GenerateRDF",
    "JudgeTripleQuality",
    # Stage 1: Statement Extraction
    "StatementExtractor",
    "BatchStatementExtractor",
    "StatementQualityJudge",
    "StatementClassifier",
    "StatementClassification",
    "statement_quality_metric",
    "statement_classification_metric",
    # Stage 2: Schema Context
    "SchemaContextBuilder",
    "SchemaLibrary",
    "create_default_schema_library",
    # Stage 3: RDF Generation
    "RDFGenerator",
    "BatchRDFGenerator",
    "TripleQualityJudge",
    "triple_quality_metric",
    # Provenance
    "generate_statements_notebook_header",
    "append_statements_cell",
    "save_statements_notebook",
    "load_statements_from_notebook",
    "generate_classifications_notebook_header",
    "append_classifications_cell",
    "save_classifications_notebook",
    "load_classifications_from_notebook",
    "get_processed_chunk_cids",
    "create_output_directory",
    # Training Data
    "WikipediaPage",
    "WikipediaChunk",
    "fetch_top_pages",
    "fetch_page_content",
    "chunk_article",
    "generate_sample_notebook_header",
    "append_sample_page_cell",
    "generate_chunks_notebook_header",
    "append_chunk_cell",
    "load_sample_from_notebook",
    "load_chunks_from_notebook",
    "get_processed_page_titles",
    "save_notebook",
    "fetch_and_cache_pages",
]
