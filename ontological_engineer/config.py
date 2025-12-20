"""
LM Configuration for Ontological Engineer.

Provides utilities for configuring DSPy language models for different use cases:
- Default inference with Qwen-30B via LM Studio
- GRPO training with Arbor
- Evaluation with different model sizes
"""

import os
from typing import Optional
import dspy


# Default configuration
DEFAULT_LM_STUDIO_URL = "http://host.docker.internal:1234/v1"
DEFAULT_MODEL = "qwen/qwen3-coder-30b"


def get_default_lm(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dspy.LM:
    """
    Get the default language model configured for LM Studio.
    
    Args:
        model: Model name (default: qwen/qwen3-coder-30b)
        api_base: API base URL (default: LM Studio on host)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured dspy.LM instance
    """
    return dspy.LM(
        model=f"openai/{model or DEFAULT_MODEL}",
        api_base=api_base or DEFAULT_LM_STUDIO_URL,
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        temperature=temperature,
        max_tokens=max_tokens,
        # Disable response_format - LM Studio doesn't support json_schema mode
        # DSPy will use text mode and parse structured outputs from text
        response_format=None,
    )


def configure_lm(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dspy.LM:
    """
    Configure and set the default DSPy language model.
    
    Args:
        model: Model name (default: qwen/qwen3-coder-30b)
        api_base: API base URL (default: LM Studio on host)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured dspy.LM instance (also set as dspy default)
    """
    lm = get_default_lm(
        model=model,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    dspy.configure(lm=lm)
    return lm


def get_arbor_lm(
    model_name: str,
    arbor_base_url: str,
    temperature: float = 1.0,  # Required for GRPO
    max_tokens: int = 2048,
) -> dspy.LM:
    """
    Get a language model configured for Arbor GRPO training.
    
    Args:
        model_name: Name of the model loaded in Arbor
        arbor_base_url: Base URL from arbor.init()
        temperature: Sampling temperature (1.0 required for GRPO)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Configured dspy.LM instance for Arbor
    """
    # Import here to avoid requiring arbor for basic usage
    from arbor.integrations.dspy.provider import ArborProvider
    
    return dspy.LM(
        model=f"openai/arbor:{model_name}",
        provider=ArborProvider(),
        api_base=arbor_base_url,
        api_key="arbor",
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )


# Model presets for quick access
MODEL_PRESETS = {
    "qwen-30b": "qwen/qwen3-coder-30b",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
}


def get_preset_lm(preset: str, **kwargs) -> dspy.LM:
    """
    Get a language model from a preset name.
    
    Args:
        preset: One of 'qwen-30b', 'qwen-7b', 'qwen-3b', 'llama-3b'
        **kwargs: Additional arguments passed to get_default_lm
        
    Returns:
        Configured dspy.LM instance
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    
    return get_default_lm(model=MODEL_PRESETS[preset], **kwargs)
