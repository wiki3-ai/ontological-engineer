# Wiki3.ai Development Practices

> **Purpose**: Capture development patterns, conventions, and lessons learned to avoid repeating past mistakes.

---

## Quick Reference

### Project Structure

```
wiki3-kg-project/
├── ontological_engineer/       # Main DSPy package (importable)
│   ├── __init__.py             # Package exports
│   ├── config.py               # LM configuration
│   ├── signatures.py           # DSPy signatures (ExtractStatements, etc.)
│   ├── extractors.py           # DSPy modules (StatementExtractor)
│   ├── judges.py               # Judge modules + metrics + StatementClassifier
│   ├── provenance.py           # Notebook generation with CID provenance
│   └── training/
│       └── bootstrap.py        # Load training data from notebooks
├── src/                        # Utility modules (non-DSPy)
│   ├── cid.py                  # CID computation + JSON-LD signatures
│   ├── entity_registry.py      # Entity tracking
│   └── ...
├── tests/                      # pytest test suite
│   └── test_*.py               # All tests go here
├── notebooks/                  # Interactive development
│   └── stage1_statements.ipynb # Stage 1 experiments
└── data/                       # Output data
    └── training/               # Training datasets
```

---

## CID & Provenance Conventions

### JSON-LD Signature Format

**CRITICAL**: Signatures MUST use JSON-LD format with these required fields:

```python
from src.cid import make_signature

sig = make_signature(
    cell_num=1,
    cell_type="statements",
    cid="bafkreig...",           # Content hash
    from_cid="bafkreix...",      # Source content hash (provenance)
    repro_class="Data",
    label="statements:chunk_0"
)

# Produces:
{
    "@context": {...},
    "@id": "ipfs://bafkreig...",          # ← Required
    "@type": ["prov:Entity", "repro:Data"],
    "dcterms:identifier": "bafkreig...",
    "prov:label": "statements:chunk_0",
    "prov:wasDerivedFrom": {"@id": "ipfs://bafkreix..."},  # ← Required structure
    "_cell": 1,                            # ← Required metadata
    "_type": "statements",
}
```

### Reading Signatures

Use `parse_signature()` to handle both legacy and JSON-LD formats:

```python
from src.cid import parse_signature, extract_signatures

# Single signature
sig = parse_signature(raw_json_string)

# All signatures from a notebook
sigs = extract_signatures(notebook_obj)  # Returns dict[str, dict]
```

### Checking Already-Processed Content

```python
from src.cid import extract_signatures

existing_sigs = extract_signatures(notebook_obj)

# Get all source CIDs that have been processed
processed_from_cids = {
    sig.get('prov:wasDerivedFrom', {}).get('@id', '')
    for sig in existing_sigs.values()
}

# Check if this chunk was already processed
chunk_cid_uri = f"ipfs://{chunk_cid}"
if chunk_cid_uri in processed_from_cids:
    print("Already processed, skipping")
```

---

## DSPy Patterns

### Signature Definition

```python
import dspy

class ExtractStatements(dspy.Signature):
    """Docstring becomes the system instruction."""
    
    # Inputs
    chunk_text: str = dspy.InputField(desc="Description for LM")
    section_context: str = dspy.InputField(desc="Brief description")
    
    # Outputs
    statements: list[str] = dspy.OutputField(desc="What the LM should return")
```

### Module Definition

```python
class StatementExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = dspy.ChainOfThought(ExtractStatements)
    
    def forward(self, chunk_text: str, section_context: str):
        return self.extract(
            chunk_text=chunk_text,
            section_context=section_context,
        )
```

### Metric Functions for DSPy.Evaluate

Metrics MUST return a float 0-100 (or 0-1 scaled to 100):

```python
def statement_quality_metric(example, pred, trace=None):
    """
    Args:
        example: dspy.Example with inputs
        pred: Module output (Prediction)
        trace: Optional trace info
        
    Returns:
        float: Score 0-100
    """
    judge = StatementQualityJudge()
    result = judge(
        chunk_text=example.chunk_text,
        section_context=example.section_context,
        statements=pred.statements,
    )
    return result.weighted_score * 100  # Scale to 0-100
```

### Creating Training Examples

```python
ex = dspy.Example(
    chunk_text="...",
    section_context="...",
).with_inputs('chunk_text', 'section_context')  # ← Mark inputs

# For labeled data (when you have expected outputs):
ex = dspy.Example(
    chunk_text="...",
    section_context="...",
    statements=["stmt1", "stmt2"],
).with_inputs('chunk_text', 'section_context')
```

---

## Testing Conventions

### File Organization

- All tests in `tests/test_*.py`
- Use pytest fixtures for reusable setup
- Name tests descriptively: `test_<what>_<condition>_<expected>`

### Standard Test Structure

```python
"""Tests for <module_name>."""

import pytest
from pathlib import Path

from ontological_engineer.module_name import (
    function_to_test,
    ClassToTest,
)


class TestFunctionName:
    """Tests for function_name()."""
    
    def test_basic_usage(self):
        """Test normal usage."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case description."""
        ...
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Using Temporary Files

```python
def test_with_temp_file(tmp_path):
    """tmp_path is a pytest fixture that auto-cleans."""
    test_file = tmp_path / "test.ipynb"
    # ... create and test ...
```

### Running Tests

```bash
# All tests (ALWAYS use pytest, not python -m pytest)
pytest tests/ -v

# Specific file
pytest tests/test_provenance.py -v

# Specific test
pytest tests/test_provenance.py::TestStatementsNotebook::test_basic_generation -v

# With short traceback
pytest tests/ -v --tb=short

# With virtual environment activated
source .venv/bin/activate && pytest tests/ -v
```

### Where Tests Live

**All tests go in `tests/` directory** - not in `ontological_engineer/tests/` or anywhere else.

```
tests/
├── __init__.py
├── test_bootstrap.py       # Training data loading
├── test_config.py          # LM configuration
├── test_extractors.py      # StatementExtractor
├── test_judges.py          # Quality judges + StatementClassifier
├── test_provenance.py      # Notebook generation with CID provenance
├── test_rdf_generator.py   # RDF generation
├── test_schema_context.py  # Schema context builder
├── test_schema_library.py  # Schema library
└── test_signatures.py      # DSPy signatures
```

---

## MLflow Integration

### Setup Pattern

```python
import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("wiki3-kg-stage1-statements")
    mlflow.dspy.autolog()  # Auto-trace DSPy calls
    MLFLOW_ENABLED = True
except Exception:
    MLFLOW_ENABLED = False
```

### Logging Spans

```python
with mlflow.start_run(run_name="descriptive_name"):
    for i, example in enumerate(examples):
        with mlflow.start_span(name=f"example_{i}") as span:
            # Do work
            result = module(...)
            
            # Log to span
            span.set_inputs({"key": value})
            span.set_outputs({"key": value})
    
    # Log aggregate metrics
    mlflow.log_metric("avg_score", avg_score)
```

### Starting MLflow Server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.sqlite \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Or use the convenience script: `./scripts/start_mlflow.sh`

---

## Notebook Generation (Provenance)

### Using the Provenance Module

```python
from ontological_engineer.provenance import (
    generate_statements_notebook_header,
    append_statements_cell,
    generate_classifications_notebook_header,
    append_classifications_cell,
    get_processed_chunk_cids,
)

# Create statements notebook
nb = generate_statements_notebook_header(
    pipeline_version="1.0.0",
    model_name="qwen3-30b",
    config={"temperature": 0.7},
)

# Append statement cells
append_statements_cell(
    nb,
    cell_num=1,
    chunk_num=0,
    chunk_text="...",
    chunk_cid="bafkreig...",
    section_context="Article > Section",
    statements=["stmt1", "stmt2"],
)

# Check what's already processed
processed = get_processed_chunk_cids(nb)  # Returns set of ipfs:// URIs
```

### Notebook Cell Format

**Statements cells** use Markdown:

```markdown
## Chunk 0: Article > Section

**Statements:**

1. First statement here.
2. Second statement here.

<details>
<summary>📋 Source Chunk</summary>

Original chunk text here...

</details>
```

**Signature cells** use Raw (JSON):

```json
{
  "@context": {...},
  "@id": "ipfs://bafkreig...",
  ...
}
```

---

## Common Mistakes & Fixes

### ❌ Forgetting to mark DSPy inputs

```python
# Wrong - no input marking
ex = dspy.Example(chunk_text="...", section_context="...")

# Correct
ex = dspy.Example(chunk_text="...", section_context="...").with_inputs('chunk_text', 'section_context')
```

### ❌ Wrong signature format in tests

```python
# Wrong - missing required JSON-LD fields
sig = {"cid": "...", "from_cid": "..."}

# Correct - use make_signature() or include all fields
from src.cid import make_signature
sig = make_signature(cell_num=1, cell_type="test", cid="...", from_cid="...")
```

### ❌ Checking from_cid incorrectly

```python
# Wrong - missing @id nesting
if chunk_cid in sig.get('prov:wasDerivedFrom', ''):

# Correct - JSON-LD uses nested structure
if f"ipfs://{chunk_cid}" == sig.get('prov:wasDerivedFrom', {}).get('@id', ''):
```

### ❌ Metric returning wrong scale

```python
# Wrong - returns 0-1
def metric(example, pred, trace=None):
    return score  # 0.85

# Correct - returns 0-100
def metric(example, pred, trace=None):
    return score * 100  # 85.0
```

### ❌ Using mlflow.evaluate with DSPy

```python
# Wrong - mlflow.evaluate API changed in 3.x
mlflow.evaluate(data=df, model=module, ...)

# Correct - use manual spans
with mlflow.start_span(name="eval") as span:
    result = module(...)
    span.set_outputs({...})
```

### ❌ Importing from wrong location

```python
# Wrong - old location
from src.cid import ...

# Correct for DSPy modules - use package
from ontological_engineer import StatementExtractor

# Correct for CID utilities - use src
from src.cid import compute_cid, make_signature
```

---

## LM Configuration

### Default: LM Studio with Qwen-30B

```python
from ontological_engineer import configure_lm

lm = configure_lm(
    model="qwen/qwen3-coder-30b",
    api_base="http://host.docker.internal:1234/v1",  # Docker → host
    temperature=0.7,
)
```

### API Base URLs

| Environment | URL |
|-------------|-----|
| Docker devcontainer → host LM Studio | `http://host.docker.internal:1234/v1` |
| Local → local LM Studio | `http://localhost:1234/v1` |
| MLflow (inside container) | `http://127.0.0.1:5000` |

---

## Data Provenance Chain

```
chunks.ipynb                    statements.ipynb                classifications.ipynb
┌───────────────┐              ┌────────────────┐              ┌────────────────────┐
│ Chunk Cell    │───derives───▶│ Statement Cell │───derives───▶│ Classification Cell│
│ CID: bafkA    │              │ CID: bafkB     │              │ CID: bafkC         │
│               │              │ from: bafkA    │              │ from: bafkB        │
└───────────────┘              └────────────────┘              └────────────────────┘
```

Each output links to its source via `prov:wasDerivedFrom`, creating a verifiable chain.

---

## Files to Read for Context

When starting a new session or making changes, reference:

1. **This file** (`DEVELOPMENT_PRACTICES.md`) - Patterns & conventions
2. **`DSPY_PIPELINE_DESIGN.md`** - Architecture & stage design
3. **`DESIGN.md`** - Original pipeline design (CID, provenance, notebooks)
4. **`src/cid.py`** - CID utilities & signature format
5. **`ontological_engineer/__init__.py`** - Package exports

---

## Checklist Before Committing

- [ ] Tests pass: `pytest tests/ -v`
- [ ] No import errors in package: `python -c "from ontological_engineer import *"`
- [ ] CID signatures use JSON-LD format with `@id`, `prov:wasDerivedFrom`
- [ ] Metrics return 0-100 scale
- [ ] DSPy examples have `.with_inputs()` called
- [ ] Notebooks use `generate_*_header()` and `append_*_cell()` from provenance module
