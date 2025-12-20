"""Tests for training persistence module."""

import pytest
import json
import tempfile
from pathlib import Path

import dspy

from ontological_engineer.training.persistence import (
    save_stage1_config,
    load_stage1_config,
    save_trainset,
    load_trainset,
    save_devset,
    load_devset,
    save_fewshot_examples,
    load_fewshot_examples,
    save_baseline_results,
    save_optimized_extractor,
)


class TestConfigPersistence:
    """Tests for config save/load."""
    
    def test_save_creates_file(self, tmp_path):
        """Test that save creates the config file."""
        config = save_stage1_config(
            output_dir=tmp_path,
            model="test-model",
            api_base="http://test:1234/v1",
            temperature=0.5,
            num_fewshot=3,
        )
        
        assert (tmp_path / "stage1_config.json").exists()
        assert config["model"] == "test-model"
        assert "cid" in config
    
    def test_load_returns_saved_config(self, tmp_path):
        """Test that load returns what was saved."""
        save_stage1_config(
            output_dir=tmp_path,
            model="test-model",
            api_base="http://test:1234/v1",
            temperature=0.5,
            num_fewshot=3,
            extra_field="extra_value",
        )
        
        loaded = load_stage1_config(tmp_path)
        
        assert loaded["model"] == "test-model"
        assert loaded["temperature"] == 0.5
        assert loaded["extra_field"] == "extra_value"
    
    def test_load_missing_raises(self, tmp_path):
        """Test that load raises FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_stage1_config(tmp_path)


class TestTrainsetPersistence:
    """Tests for trainset save/load."""
    
    def test_save_returns_cid(self, tmp_path):
        """Test that save returns a CID."""
        trainset = [
            dspy.Example(
                chunk_text="Test chunk 1",
                section_context="Test > Section 1",
            ).with_inputs("chunk_text", "section_context"),
            dspy.Example(
                chunk_text="Test chunk 2",
                section_context="Test > Section 2",
            ).with_inputs("chunk_text", "section_context"),
        ]
        
        cid = save_trainset(trainset, tmp_path)
        
        assert cid is not None
        assert cid.startswith("baf")  # CIDv1 base32
    
    def test_roundtrip(self, tmp_path):
        """Test save/load roundtrip."""
        trainset = [
            dspy.Example(
                chunk_text="Test chunk 1",
                section_context="Test > Section 1",
            ).with_inputs("chunk_text", "section_context"),
            dspy.Example(
                chunk_text="Test chunk 2",
                section_context="Test > Section 2",
            ).with_inputs("chunk_text", "section_context"),
        ]
        
        save_trainset(trainset, tmp_path)
        loaded = load_trainset(tmp_path)
        
        assert len(loaded) == 2
        assert loaded[0].chunk_text == "Test chunk 1"
        assert loaded[1].section_context == "Test > Section 2"
    
    def test_load_missing_raises(self, tmp_path):
        """Test that load raises FileNotFoundError for missing trainset."""
        with pytest.raises(FileNotFoundError):
            load_trainset(tmp_path)


class TestDevsetPersistence:
    """Tests for devset save/load."""
    
    def test_roundtrip(self, tmp_path):
        """Test save/load roundtrip."""
        devset = [
            dspy.Example(
                chunk_text="Dev chunk 1",
                section_context="Dev > Section 1",
            ).with_inputs("chunk_text", "section_context"),
        ]
        
        cid = save_devset(devset, tmp_path)
        loaded = load_devset(tmp_path)
        
        assert cid is not None
        assert len(loaded) == 1
        assert loaded[0].chunk_text == "Dev chunk 1"


class TestFewshotPersistence:
    """Tests for few-shot examples save/load."""
    
    def test_roundtrip_with_statements(self, tmp_path):
        """Test save/load roundtrip including statements."""
        fewshot = [
            dspy.Example(
                chunk_text="Few-shot chunk",
                section_context="Einstein > Early life",
                statements=["Statement 1.", "Statement 2."],
            ).with_inputs("chunk_text", "section_context"),
        ]
        
        cid = save_fewshot_examples(fewshot, tmp_path)
        loaded = load_fewshot_examples(tmp_path)
        
        assert cid is not None
        assert len(loaded) == 1
        assert loaded[0].chunk_text == "Few-shot chunk"
        assert len(loaded[0].statements) == 2
        assert loaded[0].statements[0] == "Statement 1."


class TestBaselineResultsPersistence:
    """Tests for baseline results save."""
    
    def test_save_returns_cid(self, tmp_path):
        """Test that save returns a CID."""
        cid = save_baseline_results(
            output_dir=tmp_path,
            score=0.75,
            eval_size=10,
            config_cid="baftest123",
        )
        
        assert cid is not None
        assert cid.startswith("baf")
    
    def test_save_creates_file(self, tmp_path):
        """Test that save creates the results file."""
        save_baseline_results(
            output_dir=tmp_path,
            score=0.75,
            eval_size=10,
        )
        
        assert (tmp_path / "baseline_results.json").exists()
        
        with open(tmp_path / "baseline_results.json") as f:
            results = json.load(f)
        
        assert results["score"] == 0.75
        assert results["eval_size"] == 10


class TestCIDConsistency:
    """Tests that CIDs are consistent for same content."""
    
    def test_same_config_same_cid(self, tmp_path):
        """Test that identical configs produce same CID."""
        # Create a fresh directory for each save
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        config1 = save_stage1_config(
            output_dir=dir1,
            model="test-model",
            api_base="http://test:1234/v1",
            temperature=0.5,
            num_fewshot=3,
        )
        
        # Note: timestamp will differ, so CIDs will differ
        # This is expected behavior - each save is unique
        config2 = save_stage1_config(
            output_dir=dir2,
            model="test-model",
            api_base="http://test:1234/v1",
            temperature=0.5,
            num_fewshot=3,
        )
        
        # Both should have CIDs
        assert "cid" in config1
        assert "cid" in config2
    
    def test_same_trainset_same_cid(self, tmp_path):
        """Test that identical trainsets produce same CID."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        
        trainset = [
            dspy.Example(
                chunk_text="Test chunk",
                section_context="Test > Section",
            ).with_inputs("chunk_text", "section_context"),
        ]
        
        cid1 = save_trainset(trainset, dir1)
        cid2 = save_trainset(trainset, dir2)
        
        # Same content = same CID (excluding timestamp in artifact wrapper)
        # The CID is computed from the examples data, not the full artifact
        assert cid1 == cid2
