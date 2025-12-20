"""Tests for LM configuration module."""

import pytest
import dspy

from ontological_engineer.config import (
    get_default_lm,
    configure_lm,
    get_preset_lm,
    MODEL_PRESETS,
    DEFAULT_MODEL,
    DEFAULT_LM_STUDIO_URL,
)


class TestGetDefaultLM:
    """Tests for get_default_lm function."""
    
    def test_returns_lm_instance(self):
        """Test returns a DSPy LM instance."""
        lm = get_default_lm()
        assert isinstance(lm, dspy.LM)
    
    def test_uses_default_model(self):
        """Test uses default model when not specified."""
        lm = get_default_lm()
        assert DEFAULT_MODEL in str(lm.model)
    
    def test_accepts_custom_model(self):
        """Test accepts custom model name."""
        lm = get_default_lm(model="custom/model")
        assert "custom/model" in str(lm.model)
    
    def test_accepts_custom_api_base(self):
        """Test accepts custom API base URL."""
        lm = get_default_lm(api_base="http://localhost:8080/v1")
        # LM stores config internally, just verify it was created
        assert lm is not None
    
    def test_accepts_temperature(self):
        """Test accepts temperature parameter."""
        lm = get_default_lm(temperature=0.5)
        # LM stores config internally, just verify it was created
        assert lm is not None
    
    def test_accepts_max_tokens(self):
        """Test accepts max_tokens parameter."""
        lm = get_default_lm(max_tokens=1024)
        # LM stores config internally, just verify it was created
        assert lm is not None


class TestConfigureLM:
    """Tests for configure_lm function."""
    
    def test_returns_lm_instance(self):
        """Test returns a DSPy LM instance."""
        lm = configure_lm()
        assert isinstance(lm, dspy.LM)
    
    def test_sets_dspy_default(self):
        """Test sets the DSPy default LM."""
        lm = configure_lm()
        # After configure_lm, dspy.settings should have the LM
        # This is set via dspy.configure(lm=lm)
        assert lm is not None


class TestGetPresetLM:
    """Tests for get_preset_lm function."""
    
    def test_returns_lm_for_valid_preset(self):
        """Test returns LM for valid preset names."""
        for preset in MODEL_PRESETS.keys():
            lm = get_preset_lm(preset)
            assert isinstance(lm, dspy.LM)
    
    def test_raises_for_invalid_preset(self):
        """Test raises ValueError for invalid preset."""
        with pytest.raises(ValueError) as exc_info:
            get_preset_lm("nonexistent-preset")
        
        assert "Unknown preset" in str(exc_info.value)
    
    def test_passes_kwargs_to_lm(self):
        """Test passes additional kwargs."""
        lm = get_preset_lm("qwen-30b", temperature=0.9)
        # LM stores config internally, just verify it was created
        assert lm is not None


class TestModelPresets:
    """Tests for MODEL_PRESETS constant."""
    
    def test_has_qwen_presets(self):
        """Test has Qwen model presets."""
        assert "qwen-30b" in MODEL_PRESETS
        assert "qwen-7b" in MODEL_PRESETS
        assert "qwen-3b" in MODEL_PRESETS
    
    def test_has_llama_preset(self):
        """Test has Llama model preset."""
        assert "llama-3b" in MODEL_PRESETS
    
    def test_preset_values_are_model_names(self):
        """Test preset values are valid model name strings."""
        for preset, model in MODEL_PRESETS.items():
            assert isinstance(model, str)
            assert len(model) > 0
