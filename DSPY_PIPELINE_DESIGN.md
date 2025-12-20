# Wiki3.ai Ontological Engineer - DSPy Pipeline Design

## Overview

The **Ontological Engineer (OE)** transforms Wikipedia articles into high-quality RDF knowledge graphs using DSPy and Arbor GRPO for reinforcement learning-based optimization.

**Repository**: https://github.com/wiki3-ai/ontological-engineer

### Goals

1. **Higher Quality**: Use LLM judges and RL optimization to improve extraction quality
2. **Faster Iteration**: DSPy's declarative approach enables rapid experimentation
3. **Model Flexibility**: Easy to swap between Qwen-30B, smaller models, etc.
4. **Human-in-the-Loop Ready**: Notebook-based outputs enable external feedback via Wiki3.ai

### What We're Keeping

- **Chunk preprocessing**: RecursiveCharacterTextSplitter with section breadcrumbs works well
- **Notebook-based provenance**: `.ipynb` files with CID signatures for reproducibility
- **Incremental processing**: Skip already-processed content based on content hashes

### What's Changing

| Component | Old Approach | New Approach |
|-----------|-------------|--------------|
| Statement Extraction | Manual LangChain prompt | DSPy Module + GRPO optimization |
| Quality Control | None (manual tuning) | Automated LLM judges |
| Schema Lookup | Tool calls during generation | Pre-computed context (DSPy module) |
| RDF Generation | Tool-based emit_triple() | Direct generation with schema context |
| Optimization | Manual prompt iteration | ArborGRPO / MIPROv2 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING (existing)                      │
│                                                                  │
│  Wikipedia Article                                               │
│       ↓                                                          │
│  fetch_wikipedia_with_links() → WikiArticle                     │
│       ↓                                                          │
│  RecursiveCharacterTextSplitter → chunks[]                      │
│       ↓                                                          │
│  create_contextual_chunks() → ContextualChunk[]                 │
│       (with section breadcrumbs, entity links)                  │
│                                                                  │
│  Output: chunks.ipynb                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 1: Statement Extraction                          │
│                                                                  │
│  DSPy Module: StatementExtractor                                 │
│    Input: chunk_text, section_context                           │
│    Output: statements[]                                         │
│                                                                  │
│  DSPy Judge: StatementQualityJudge                               │
│    Automated scoring → GRPO reward signal                       │
│                                                                  │
│  Optimizer: ArborGRPO                                            │
│    Trains extractor to maximize judge scores                    │
│                                                                  │
│  Output: statements.ipynb                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 2: Schema Context Preparation                    │
│                                                                  │
│  DSPy Module: SchemaContextBuilder                               │
│    Input: statements[], schema_library                          │
│    Output: prefixes, relevant_terms, usage_examples             │
│                                                                  │
│  Runs BEFORE RDF generation to build prompt context             │
│  No tools at generation time - all context pre-computed         │
│                                                                  │
│  Output: schema_context (in memory, or cached)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           STAGE 3: RDF Generation                                │
│                                                                  │
│  DSPy Module: RDFGenerator                                       │
│    Input: statements[], schema_context                          │
│    Output: turtle_triples                                       │
│                                                                  │
│  DSPy Judge: TripleQualityJudge                                  │
│    Validates syntax, URIs, schema conformance                   │
│                                                                  │
│  Can generate custom annotations per ontology-guide.md          │
│  (RDF-Star, Role reification, temporal patterns)                │
│                                                                  │
│  Output: rdf.ipynb, triples.ttl                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
wiki3-kg-project/
├── ontological_engineer/           # Main package
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # LM configuration, presets
│   ├── signatures.py               # All DSPy signatures
│   ├── extractors.py               # StatementExtractor module
│   ├── judges.py                   # Quality judge modules + metrics
│   ├── schema_context.py           # SchemaContextBuilder
│   ├── schema_library.py           # Vocab/ontology management
│   ├── rdf_generator.py            # RDFGenerator module
│   └── training/
│       ├── __init__.py
│       └── bootstrap.py            # Load training data from notebooks
├── notebooks/
│   ├── stage1_statements.ipynb     # Statement extraction experiments
│   ├── stage2_schema.ipynb         # Schema context experiments (TODO)
│   └── stage3_rdf.ipynb            # RDF generation experiments (TODO)
├── data/
│   ├── training/                   # Curated training sets
│   │   ├── statement_trainset.json
│   │   └── statement_devset.json
│   ├── albert_einstein/            # Example article outputs
│   └── vocab_cache/                # Cached schema embeddings
└── DSPY_PIPELINE_DESIGN.md         # This file
```

---

## Stage 1: Statement Extraction

### DSPy Signature

```python
class ExtractStatements(dspy.Signature):
    """Extract atomic, verifiable statements from Wikipedia text.
    
    Each statement must:
    - Be self-contained (understandable without the original text)
    - Preserve markdown links: [Entity Name](/wiki/Entity_Name)
    - Contain exactly one verifiable claim
    - Not editorialize or interpret beyond what's stated
    """
    
    chunk_text: str = dspy.InputField(desc="Wikipedia article chunk")
    section_context: str = dspy.InputField(desc="Breadcrumb: Article > Section > Subsection")
    
    statements: list[str] = dspy.OutputField(
        desc="List of atomic statements preserving entity links"
    )
```

### Quality Judge Signature

```python
class JudgeStatementQuality(dspy.Signature):
    """Judge the quality of extracted statements."""
    
    chunk_text: str = dspy.InputField(desc="Original Wikipedia chunk")
    section_context: str = dspy.InputField(desc="Section breadcrumb")
    statements: list[str] = dspy.InputField(desc="Extracted statements")
    
    completeness: float = dspy.OutputField(
        desc="0-1: Are all key facts from the chunk captured?"
    )
    atomicity: float = dspy.OutputField(
        desc="0-1: Is each statement truly atomic (one claim)?"
    )
    accuracy: float = dspy.OutputField(
        desc="0-1: Do statements faithfully represent the source?"
    )
    link_preservation: float = dspy.OutputField(
        desc="0-1: Are [Entity](/wiki/...) links preserved correctly?"
    )
    reasoning: str = dspy.OutputField(desc="Brief explanation of scores")
```

---

## Stage 2: Schema Context Preparation

### Purpose

Instead of tool calls during generation, we pre-compute relevant schema context:
- Which prefixes/namespaces are needed
- Which classes and properties are relevant
- Usage examples and patterns

### DSPy Signature

```python
class SelectSchemaTerms(dspy.Signature):
    """Select relevant schema terms for a set of statements."""
    
    statements: list[str] = dspy.InputField()
    candidate_classes: list[dict] = dspy.InputField(desc="Available classes with descriptions")
    candidate_properties: list[dict] = dspy.InputField(desc="Available properties with descriptions")
    
    selected_classes: list[str] = dspy.OutputField(desc="Classes needed for these statements")
    selected_properties: list[str] = dspy.OutputField(desc="Properties needed")
    custom_annotation_needs: str = dspy.OutputField(
        desc="Any custom annotations needed (temporal, reification, etc.)"
    )
```

---

## Stage 3: RDF Generation

### DSPy Signature

```python
class GenerateRDF(dspy.Signature):
    """Generate RDF triples from statements using provided schema context.
    
    Follow these rules:
    - Convert [Entity](/wiki/Entity) links to <https://en.wikipedia.org/wiki/Entity>
    - Use provided prefixes and property definitions
    - Apply temporal/reification patterns where dates are involved
    - Output valid Turtle syntax
    """
    
    statements: list[str] = dspy.InputField(desc="Statements to convert")
    schema_context: str = dspy.InputField(desc="Prefixes, definitions, examples")
    entity_registry: str = dspy.InputField(desc="Known entity URIs")
    
    turtle_triples: str = dspy.OutputField(desc="Valid Turtle RDF")
```

### Quality Judge Signature

```python
class JudgeTripleQuality(dspy.Signature):
    """Judge RDF triple quality."""
    
    statements: list[str] = dspy.InputField(desc="Source statements")
    turtle_triples: str = dspy.InputField(desc="Generated RDF")
    schema_context: str = dspy.InputField(desc="Available schema terms")
    
    syntax_valid: bool = dspy.OutputField(desc="Is the Turtle syntax valid?")
    uris_correct: bool = dspy.OutputField(desc="Are URIs properly formed?")
    schema_conformance: float = dspy.OutputField(
        desc="0-1: Are appropriate schema terms used?"
    )
    completeness: float = dspy.OutputField(
        desc="0-1: Are all statements represented?"
    )
    reasoning: str = dspy.OutputField()
```

---

## Training Strategy

### Phase 1: Bootstrap

1. **Curate training data** from existing `facts.ipynb` outputs
2. **Initial evaluation** with judge on devset to establish baseline
3. **MIPROv2 prompt optimization** (fast, no model training)

### Phase 2: GRPO Training

1. **ArborGRPO on StatementExtractor** using StatementQualityJudge as reward
2. **Evaluate and iterate** - check devset performance, review failures
3. **Try smaller models** - Qwen-7B, Qwen-3B for speed/quality tradeoffs

### Phase 3: RDF Generation

1. **Build SchemaLibrary** with patterns from ontology-guide.md
2. **Train SchemaContextBuilder** (may not need GRPO)
3. **ArborGRPO on RDFGenerator** with TripleQualityJudge

---

## Model Configuration

### Default: Qwen-30B (via LM Studio)

```python
lm = dspy.LM(
    model="openai/qwen/qwen3-coder-30b",
    api_base="http://host.docker.internal:1234/v1",
    api_key="lm-studio",
    temperature=0.7,
    max_tokens=2048,
)
```

### For GRPO Training (Arbor)

```python
arbor_server_info = arbor.init()

lm = dspy.LM(
    model=f"openai/arbor:{model_name}",
    provider=ArborProvider(),
    api_base=arbor_server_info["base_url"],
    api_key="arbor",
    temperature=1.0,  # Required for GRPO
    top_p=1.0,
    max_tokens=2048,
)
```

---

## Human Feedback Integration

Human feedback is **external to the training loop** but influences future training:

1. **Notebooks expose all intermediate data**
   - `statements.ipynb`: Extracted statements per chunk
   - `rdf.ipynb`: Generated triples with provenance
   
2. **Wiki3.ai interface allows**
   - Viewing any intermediate output
   - Marking agree/disagree on judge decisions
   - Providing corrections/improvements
   
3. **Feedback collection**
   - Store feedback alongside outputs
   - Periodically incorporate into training data

---

## Next Steps

1. [x] Create design document
2. [x] Create `ontological_engineer/` package structure
3. [x] Implement `StatementExtractor` and judge
4. [x] Create Stage 1 notebook for interactive development
5. [x] Bootstrap training data from existing outputs
6. [ ] Run MIPROv2 as quick first experiment
7. [ ] Set up Arbor for GRPO training
