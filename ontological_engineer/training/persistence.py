"""
Persistence utilities for Stage 1 training artifacts.

Handles saving and loading of:
- Configuration (model settings, hyperparameters)
- Training/dev datasets
- Few-shot examples
- Optimized extractors

All artifacts include CID provenance for reproducibility.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from ontological_engineer.cid import compute_cid


def save_stage1_config(
    output_dir: Path,
    model: str,
    api_base: str,
    temperature: float,
    num_fewshot: int,
    **extra_config,
) -> dict:
    """
    Save Stage 1 configuration with CID provenance.
    
    Returns the config dict with CID.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = {
        "model": model,
        "api_base": api_base,
        "temperature": temperature,
        "num_fewshot": num_fewshot,
        "timestamp": datetime.now().isoformat(),
        **extra_config,
    }
    
    # Compute CID for provenance
    config_str = json.dumps(config, sort_keys=True)
    config["cid"] = compute_cid(config_str.encode())
    
    config_path = output_dir / "stage1_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return config


def load_stage1_config(training_dir: Path) -> dict:
    """Load Stage 1 configuration from saved artifact."""
    config_path = Path(training_dir) / "stage1_config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Stage 1 config not found at {config_path}. "
            "Run stage1_statements.ipynb first."
        )
    
    with open(config_path) as f:
        return json.load(f)


def save_trainset(trainset: list[dspy.Example], output_dir: Path) -> str:
    """
    Save training set with CID provenance.
    
    Returns the CID of the saved dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = []
    for ex in trainset:
        data.append({
            "chunk_text": ex.chunk_text,
            "section_context": ex.section_context,
        })
    
    # Compute CID
    data_str = json.dumps(data, sort_keys=True)
    cid = compute_cid(data_str.encode())
    
    artifact = {
        "cid": cid,
        "count": len(data),
        "timestamp": datetime.now().isoformat(),
        "examples": data,
    }
    
    trainset_path = output_dir / "trainset.json"
    with open(trainset_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    return cid


def load_trainset(training_dir: Path) -> list[dspy.Example]:
    """Load training set and convert back to DSPy Examples."""
    trainset_path = Path(training_dir) / "trainset.json"
    
    if not trainset_path.exists():
        raise FileNotFoundError(
            f"Trainset not found at {trainset_path}. "
            "Run stage1_statements.ipynb first."
        )
    
    with open(trainset_path) as f:
        artifact = json.load(f)
    
    trainset = []
    for item in artifact["examples"]:
        ex = dspy.Example(
            chunk_text=item["chunk_text"],
            section_context=item["section_context"],
        ).with_inputs("chunk_text", "section_context")
        trainset.append(ex)
    
    return trainset


def save_devset(devset: list[dspy.Example], output_dir: Path) -> str:
    """
    Save dev set with CID provenance.
    
    Returns the CID of the saved dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    data = []
    for ex in devset:
        data.append({
            "chunk_text": ex.chunk_text,
            "section_context": ex.section_context,
        })
    
    # Compute CID
    data_str = json.dumps(data, sort_keys=True)
    cid = compute_cid(data_str.encode())
    
    artifact = {
        "cid": cid,
        "count": len(data),
        "timestamp": datetime.now().isoformat(),
        "examples": data,
    }
    
    devset_path = output_dir / "devset.json"
    with open(devset_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    return cid


def load_devset(training_dir: Path) -> list[dspy.Example]:
    """Load dev set and convert back to DSPy Examples."""
    devset_path = Path(training_dir) / "devset.json"
    
    if not devset_path.exists():
        raise FileNotFoundError(
            f"Devset not found at {devset_path}. "
            "Run stage1_statements.ipynb first."
        )
    
    with open(devset_path) as f:
        artifact = json.load(f)
    
    devset = []
    for item in artifact["examples"]:
        ex = dspy.Example(
            chunk_text=item["chunk_text"],
            section_context=item["section_context"],
        ).with_inputs("chunk_text", "section_context")
        devset.append(ex)
    
    return devset


def save_fewshot_examples(
    fewshot_examples: list[dspy.Example],
    output_dir: Path,
) -> str:
    """
    Save few-shot examples with CID provenance.
    
    Returns the CID of the saved examples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = []
    for ex in fewshot_examples:
        data.append({
            "chunk_text": ex.chunk_text,
            "section_context": ex.section_context,
            "statements": list(ex.statements),
        })
    
    # Compute CID
    data_str = json.dumps(data, sort_keys=True)
    cid = compute_cid(data_str.encode())
    
    artifact = {
        "cid": cid,
        "count": len(data),
        "timestamp": datetime.now().isoformat(),
        "examples": data,
    }
    
    fewshot_path = output_dir / "fewshot_examples.json"
    with open(fewshot_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    return cid


def load_fewshot_examples(training_dir: Path) -> list[dspy.Example]:
    """Load few-shot examples and convert back to DSPy Examples."""
    fewshot_path = Path(training_dir) / "fewshot_examples.json"
    
    if not fewshot_path.exists():
        raise FileNotFoundError(
            f"Few-shot examples not found at {fewshot_path}. "
            "Run stage1_statements.ipynb first."
        )
    
    with open(fewshot_path) as f:
        artifact = json.load(f)
    
    examples = []
    for item in artifact["examples"]:
        ex = dspy.Example(
            chunk_text=item["chunk_text"],
            section_context=item["section_context"],
            statements=item["statements"],
        ).with_inputs("chunk_text", "section_context")
        examples.append(ex)
    
    return examples


def compute_module_cid(module: dspy.Module) -> str:
    """
    Compute a CID for a DSPy module based on its definition.
    
    Captures:
    - The signature(s) used (field names, descriptions)
    - Any demonstrations attached to the module
    - The module class name and structure
    
    This ensures cache invalidation when the module changes.
    """
    module_spec = {
        "class": module.__class__.__name__,
        "module": module.__class__.__module__,
    }
    
    # Capture signature info from predictors
    predictors = []
    for name, predictor in module.named_predictors():
        sig_info = {"name": name}
        
        # Get signature class and fields
        if hasattr(predictor, 'signature'):
            sig = predictor.signature
            sig_info["signature_class"] = sig.__class__.__name__
            sig_info["signature_doc"] = sig.__doc__ or ""
            
            # Capture input/output field definitions
            if hasattr(sig, 'input_fields'):
                sig_info["input_fields"] = {
                    k: {"desc": getattr(v, 'desc', '')} 
                    for k, v in sig.input_fields.items()
                }
            if hasattr(sig, 'output_fields'):
                sig_info["output_fields"] = {
                    k: {"desc": getattr(v, 'desc', '')} 
                    for k, v in sig.output_fields.items()
                }
        
        # Capture demos if any
        if hasattr(predictor, 'demos') and predictor.demos:
            sig_info["num_demos"] = len(predictor.demos)
            # Hash demo content for change detection
            demo_strs = []
            for demo in predictor.demos:
                demo_dict = {k: str(v) for k, v in demo.items()}
                demo_strs.append(json.dumps(demo_dict, sort_keys=True))
            sig_info["demos_hash"] = compute_cid("".join(demo_strs).encode())
        
        predictors.append(sig_info)
    
    module_spec["predictors"] = predictors
    
    return compute_cid(json.dumps(module_spec, sort_keys=True).encode())


def compute_baseline_input_cid(
    config_cid: str,
    devset_cid: str,
    eval_size: int,
    extractor_cid: str | None = None,
) -> str:
    """
    Compute a CID representing the inputs to baseline evaluation.
    
    This allows checking if cached results match current inputs.
    Includes extractor_cid to invalidate cache when module definition changes.
    """
    input_spec = {
        "config_cid": config_cid,
        "devset_cid": devset_cid,
        "eval_size": eval_size,
    }
    if extractor_cid:
        input_spec["extractor_cid"] = extractor_cid
    
    return compute_cid(json.dumps(input_spec, sort_keys=True).encode())


def load_baseline_results(training_dir: Path) -> dict | None:
    """
    Load baseline results if they exist.
    
    Returns the results dict or None if not found.
    """
    results_path = Path(training_dir) / "baseline_results.json"
    if not results_path.exists():
        return None
    
    with open(results_path) as f:
        return json.load(f)


def check_baseline_cache(
    training_dir: Path,
    config_cid: str,
    devset_cid: str,
    eval_size: int,
    extractor_cid: str | None = None,
) -> dict | None:
    """
    Check if cached baseline results match current inputs.
    
    Returns the cached results if input CIDs match, None otherwise.
    This uses CID-based dependency checking (not timestamps).
    
    Args:
        training_dir: Directory containing baseline_results.json
        config_cid: CID of the stage1 config
        devset_cid: CID of the dev set
        eval_size: Number of examples evaluated
        extractor_cid: CID of the StatementExtractor module (optional but recommended)
    """
    cached = load_baseline_results(training_dir)
    if cached is None:
        return None
    
    # Check if the inputs match
    cached_input_cid = cached.get("input_cid")
    if cached_input_cid is None:
        # Old format without input_cid - can't verify, skip cache
        return None
    
    current_input_cid = compute_baseline_input_cid(
        config_cid, devset_cid, eval_size, extractor_cid
    )
    
    if cached_input_cid == current_input_cid:
        return cached
    
    return None


def save_baseline_results(
    output_dir: Path,
    score: float,
    eval_size: int,
    config_cid: str | None = None,
    extractor_cid: str | None = None,
    devset_cid: str | None = None,
) -> str:
    """
    Save baseline evaluation results with CID provenance.
    
    Includes input_cid for cache validation on future runs.
    Returns the CID of the saved results.
    
    Args:
        output_dir: Directory to save results
        score: Baseline evaluation score
        eval_size: Number of examples evaluated
        config_cid: CID of the stage1 config
        extractor_cid: CID of the StatementExtractor module
        devset_cid: CID of the dev set
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute input CID for cache validation
    input_cid = None
    if config_cid and devset_cid:
        input_cid = compute_baseline_input_cid(
            config_cid, devset_cid, eval_size, extractor_cid
        )
    
    results = {
        "score": score,
        "eval_size": eval_size,
        "config_cid": config_cid,
        "extractor_cid": extractor_cid,
        "devset_cid": devset_cid,
        "input_cid": input_cid,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Compute CID for the results themselves
    results_str = json.dumps(results, sort_keys=True)
    cid = compute_cid(results_str.encode())
    results["cid"] = cid
    
    results_path = output_dir / "baseline_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return cid


def save_optimized_extractor(
    extractor: Any,
    output_dir: Path,
    config: dict,
    baseline_score: float | None,
    optimized_score: float,
) -> Path:
    """
    Save optimized extractor with full provenance.
    
    Creates a timestamped directory with:
    - extractor state (JSON)
    - provenance metadata
    
    Returns the output path.
    """
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extractor_dir = output_dir / "optimized_extractors" / timestamp
    extractor_dir.mkdir(parents=True, exist_ok=True)
    
    # Save extractor state
    try:
        extractor.save(extractor_dir / "model")
    except Exception:
        # Fallback: save as JSON if DSPy save fails
        if hasattr(extractor, "dump_state"):
            state = extractor.dump_state()
            with open(extractor_dir / "model_state.json", "w") as f:
                json.dump(state, f, indent=2)
    
    # Save provenance metadata
    provenance = {
        "timestamp": timestamp,
        "config_cid": config.get("cid"),
        "baseline_score": baseline_score,
        "optimized_score": optimized_score,
        "improvement": optimized_score - baseline_score if baseline_score else None,
        "model": config.get("model"),
    }
    
    # Compute CID for provenance
    prov_str = json.dumps(provenance, sort_keys=True)
    provenance["cid"] = compute_cid(prov_str.encode())
    
    with open(extractor_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    
    # Create symlink to latest
    latest_link = output_dir / "optimized_extractor_latest"
    if latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(extractor_dir, target_is_directory=True)
    
    return extractor_dir
