# Wiki3.ai Ontological Engineer - DSPy Pipeline Design

> **See also**: [DEVELOPMENT_PRACTICES.md](DEVELOPMENT_PRACTICES.md) for coding conventions, common mistakes, and patterns.

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
├── tests/                          # pytest test suite
│   ├── __init__.py
│   ├── test_bootstrap.py           # Training data loading (13 tests)
│   ├── test_config.py              # LM configuration (12 tests)
│   ├── test_extractors.py          # StatementExtractor (7 tests)
│   ├── test_judges.py              # Quality judges (14 tests)
│   ├── test_rdf_generator.py       # RDF generation (8 tests)
│   ├── test_schema_context.py      # Schema context builder (11 tests)
│   ├── test_schema_library.py      # Schema library (16 tests)
│   └── test_signatures.py          # DSPy signatures (11 tests)
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

## Data Provenance & Persistence

The DSPy training pipeline integrates with the existing CID-based provenance system from the original pipeline. All intermediate data is persisted to Jupyter notebooks with cryptographic content identifiers (CIDs) and REPRODUCE-ME provenance metadata.

### Why Provenance Matters

1. **Reproducibility**: Every piece of data links back to its source
2. **Incremental Processing**: Skip already-processed content based on CID matches
3. **Auditability**: Full chain of custody from Wikipedia to RDF
4. **Human Feedback**: Labels link to specific CIDs, not ephemeral IDs

### Data Flow with CIDs

```
Wikipedia Article
    ↓  fetch + CID
┌─────────────────────────────────────────────────────────────────┐
│  data/{article_slug}/{timestamp}/source.ipynb                   │
│    - Raw article content                                        │
│    - CID: bafkrei... (source_cid)                              │
│    - from_cid: hash(wikipedia_url)                             │
└─────────────────────────────────────────────────────────────────┘
    ↓  chunk + CID per chunk
┌─────────────────────────────────────────────────────────────────┐
│  data/{article_slug}/{timestamp}/chunks.ipynb                   │
│    - Chunked content with section context                       │
│    - Each chunk has CID + from_cid → source_cid                │
└─────────────────────────────────────────────────────────────────┘
    ↓  extract statements + CID per chunk
┌─────────────────────────────────────────────────────────────────┐
│  data/{article_slug}/{timestamp}/statements.ipynb               │
│    - Extracted statements per chunk                             │
│    - Each statement set has CID + from_cid → chunk_cid         │
└─────────────────────────────────────────────────────────────────┘
    ↓  classify statements + CID per classification
┌─────────────────────────────────────────────────────────────────┐
│  data/{article_slug}/{timestamp}/classifications.ipynb          │
│    - Per-statement GOOD/BAD classifications                     │
│    - Each classification has CID + from_cid → statements_cid   │
└─────────────────────────────────────────────────────────────────┘
    ↓  generate RDF + CID per triple set
┌─────────────────────────────────────────────────────────────────┐
│  data/{article_slug}/{timestamp}/rdf.ipynb                      │
│    - RDF triples in Turtle format                               │
│    - Each triple set has CID + from_cid → statements_cid       │
└─────────────────────────────────────────────────────────────────┘
```

### CID Signature Format (JSON-LD)

Each content cell is followed by a signature in a raw cell:

```json
{
  "@context": {
    "prov": "http://www.w3.org/ns/prov#",
    "repro": "https://w3id.org/reproduceme#"
  },
  "@id": "ipfs://bafkreihdwdcefgh4dqkjv67uzcmw7o...",
  "@type": ["prov:Entity", "repro:Data"],
  "dcterms:identifier": "bafkreihdwdcefgh4dqkjv67uzcmw7o...",
  "prov:label": "statements:chunk_3",
  "prov:wasDerivedFrom": {"@id": "ipfs://bafkreiabc..."},
  "_cell": 5,
  "_type": "statements",
  "_chunk_num": 3
}
```

### New Notebook Types for Stage 1

#### statements.ipynb

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance metadata (YAML) |
| 1 | Raw | Entity registry (JSON) |
| 2 | Markdown | Statements for chunk 1 |
| 3 | Raw | Statements 1 CID signature (from_cid → chunk CID) |
| 4 | Markdown | Statements for chunk 2 |
| 5 | Raw | Statements 2 CID signature |
| ... | ... | ... |

**Statements cell format:**
```markdown
**Context:** Albert Einstein > Early life
**Chunk:** 1 of 63
**Statements:** 8

---

1. [Albert Einstein](/wiki/Albert_Einstein) was born on 14 March 1879.
2. [Albert Einstein](/wiki/Albert_Einstein) was born in [Ulm](/wiki/Ulm).
3. ...
```

#### classifications.ipynb

| Cell # | Type | Content |
|--------|------|---------|
| 0 | Markdown | Provenance metadata (YAML) |
| 1 | Raw | Classifier config (JSON) |
| 2 | Markdown | Classifications for chunk 1 statements |
| 3 | Raw | Classifications 1 CID signature (from_cid → statements CID) |
| ... | ... | ... |

**Classifications cell format:**
```markdown
**Context:** Albert Einstein > Early life
**Chunk:** 1 of 63
**Score:** 87.5% (7/8 GOOD)

---

| # | Classification | Statement | Reason |
|---|----------------|-----------|--------|
| 0 | ✅ GOOD | [Albert Einstein] was born on 14 March 1879. | atomic, accurate |
| 1 | ✅ GOOD | [Albert Einstein] was born in [Ulm]. | atomic, links preserved |
| 2 | ❌ BAD | Albert Einstein was a famous physicist... | missing entity link |
| ... | ... | ... | ... |

**Missing facts:** none
```

### Incremental Processing with CIDs

The training pipeline supports incremental updates:

1. **Load existing notebooks** and extract CID signatures
2. **For each source chunk**:
   - Compute what the from_cid would be
   - If matching signature exists → skip (already processed)
   - If no match → generate and append new content + signature
3. **Save notebook** after each chunk for resumability

```python
from src.cid import compute_cid, make_signature, extract_signatures

# Load existing signatures
existing_sigs = extract_signatures(statements_nb)
existing_from_cids = {sig['prov:wasDerivedFrom']['@id'] for sig in existing_sigs.values()}

# Check if chunk already processed
chunk_cid = compute_cid(chunk_content)
if f"ipfs://{chunk_cid}" in existing_from_cids:
    print(f"Chunk {i} already processed, skipping")
    continue

# Process and save
statements = extractor(chunk_text=..., section_context=...)
stmt_cid = compute_cid(statements_content)
signature = make_signature(
    cell_num=cell_num,
    cell_type="statements",
    cid=stmt_cid,
    from_cid=chunk_cid,
    repro_class="Data",
    label=f"statements:chunk_{i}"
)
```

### Human Feedback Integration

When humans provide feedback (via MLflow UI or direct annotation):

1. **Feedback links to CID**: The statement being judged has a CID
2. **Feedback stored with from_cid**: Links back to the statement CID
3. **Training data export**: Load all feedback for statements with matching CIDs
4. **Judge improvement**: Use feedback as DSPy training examples

This ensures feedback remains valid even if the pipeline reruns - feedback is attached to content, not positions.

---

## MLflow Observability & Human Feedback

We use [MLflow](https://mlflow.org/docs/latest/genai/) for tracing, evaluation, and human feedback collection. MLflow integrates natively with DSPy via `mlflow.dspy.autolog()`.

### Setup (Local Development)

1. **Install MLflow**:
   ```bash
   pip install "mlflow>=3.0"
   ```

2. **Start MLflow Server** (in a separate terminal):
   ```bash
   cd /workspaces/wiki3-kg-project
   mlflow server \
     --backend-store-uri sqlite:///mlflow.sqlite \
     --default-artifact-root ./mlflow-artifacts \
     --host 0.0.0.0 \
     --port 5000
   ```

3. **Open MLflow UI**: http://localhost:5000
   - In VS Code devcontainer: use port forwarding or `host.docker.internal`

### What MLflow Provides

| Feature | Description |
|---------|-------------|
| **Tracing** | See every LM call with inputs, outputs, latency |
| **Experiments** | Group runs by experiment (e.g., `wiki3-kg-stage1-statements`) |
| **Metrics** | Track quality scores across optimization runs |
| **Human Feedback** | Add assessments/labels directly in the UI |
| **Artifacts** | Store model checkpoints, evaluation datasets |

### Integration in Notebooks

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("wiki3-kg-stage1-statements")
mlflow.dspy.autolog()  # Auto-trace all DSPy calls

with mlflow.start_run(run_name="baseline_evaluation"):
    # Your DSPy code - automatically traced
    result = extractor(chunk_text=..., section_context=...)
    
    # Log custom metrics
    mlflow.log_metric("quality_score", score)
```

### Reviewing Predictions in MLflow UI

1. Navigate to the experiment
2. Click **Traces** tab to see all predictions
3. Click individual trace to inspect inputs/outputs
4. Use **Feedback** button to add human assessments
5. Export labeled data for judge improvement

### Human Feedback Workflow

1. **Run evaluation** → predictions logged to MLflow
2. **Review in UI** → inspect traces, add feedback labels
3. **Export annotations** → use MLflow client to retrieve
4. **Improve judge** → use labeled data as DSPy training examples

---

## Next Steps

1. [x] Create design document
2. [x] Create `ontological_engineer/` package structure
3. [x] Implement `StatementExtractor` and judge
4. [x] Create Stage 1 notebook for interactive development
5. [x] Bootstrap training data from existing outputs
6. [ ] Run MIPROv2 as quick first experiment
7. [ ] Set up Arbor for GRPO training
