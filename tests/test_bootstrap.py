"""Tests for training data bootstrap module."""

import pytest
import json
import tempfile
from pathlib import Path

import dspy

from ontological_engineer.training.bootstrap import (
    load_chunks_from_notebook,
    load_facts_from_notebook,
    create_training_examples,
    load_training_data,
    save_training_data,
    _parse_chunk_cell,
    _parse_facts_cell,
)


class TestParseChunkCell:
    """Tests for _parse_chunk_cell helper."""
    
    def test_parses_context(self):
        """Test parsing context from chunk cell."""
        source = """**Context:** Albert Einstein > Early life
**Chunk:** 1 of 10

---

Some text here."""
        
        result = _parse_chunk_cell(source)
        assert result['section_context'] == 'Albert Einstein > Early life'
    
    def test_parses_chunk_number(self):
        """Test parsing chunk number."""
        source = """**Context:** Test
**Chunk:** 5 of 10

---

Text."""
        
        result = _parse_chunk_cell(source)
        assert result['chunk_num'] == 5
    
    def test_parses_text_after_separator(self):
        """Test parsing text content after separator."""
        source = """**Context:** Test
**Chunk:** 1 of 1

---

This is the actual text content.
It spans multiple lines."""
        
        result = _parse_chunk_cell(source)
        assert 'This is the actual text content.' in result['text']
        assert 'spans multiple lines' in result['text']
    
    def test_returns_none_for_no_text(self):
        """Test returns None when no text found."""
        source = """**Context:** Test
**Chunk:** 1 of 1

---
"""
        result = _parse_chunk_cell(source)
        assert result is None


class TestParseFctsCell:
    """Tests for _parse_facts_cell helper."""
    
    def test_parses_statements(self):
        """Test parsing bulleted statements."""
        source = """**Context:** Test
**Chunk:** 1 of 1

---

- Statement one.
- Statement two.
- Statement three."""
        
        result = _parse_facts_cell(source)
        assert len(result['statements']) == 3
        assert result['statements'][0] == 'Statement one.'
    
    def test_parses_context(self):
        """Test parsing context from facts cell."""
        source = """**Context:** Albert Einstein > Physics
**Chunk:** 2 of 10

---

- A statement."""
        
        result = _parse_facts_cell(source)
        assert result['section_context'] == 'Albert Einstein > Physics'
    
    def test_returns_none_for_no_statements(self):
        """Test returns None when no statements found."""
        source = """**Context:** Test
**Chunk:** 1 of 1

---

No bullet points here."""
        
        result = _parse_facts_cell(source)
        assert result is None


class TestCreateTrainingExamples:
    """Tests for create_training_examples function."""
    
    def test_creates_dspy_examples(self):
        """Test creating DSPy examples from chunks and facts."""
        chunks = [
            {'text': 'Some text', 'section_context': 'Test', 'chunk_num': 1},
        ]
        facts = [
            {'statements': ['Statement 1', 'Statement 2'], 'section_context': 'Test', 'chunk_num': 1},
        ]
        
        examples = create_training_examples(chunks, facts)
        
        assert len(examples) == 1
        assert isinstance(examples[0], dspy.Example)
        assert examples[0].chunk_text == 'Some text'
        assert examples[0].statements == ['Statement 1', 'Statement 2']
    
    def test_filters_by_min_statements(self):
        """Test filtering by minimum statement count."""
        chunks = [
            {'text': 'Text 1', 'section_context': 'Test', 'chunk_num': 1},
            {'text': 'Text 2', 'section_context': 'Test', 'chunk_num': 2},
        ]
        facts = [
            {'statements': ['Only one'], 'section_context': 'Test', 'chunk_num': 1},
            {'statements': ['One', 'Two', 'Three'], 'section_context': 'Test', 'chunk_num': 2},
        ]
        
        examples = create_training_examples(chunks, facts, min_statements=2)
        
        assert len(examples) == 1
        assert examples[0].chunk_text == 'Text 2'
    
    def test_filters_by_max_statements(self):
        """Test filtering by maximum statement count."""
        chunks = [
            {'text': 'Text', 'section_context': 'Test', 'chunk_num': 1},
        ]
        facts = [
            {'statements': ['S' + str(i) for i in range(100)], 'section_context': 'Test', 'chunk_num': 1},
        ]
        
        examples = create_training_examples(chunks, facts, max_statements=50)
        
        assert len(examples) == 0
    
    def test_handles_missing_matches(self):
        """Test handling when chunks and facts don't match."""
        chunks = [
            {'text': 'Text', 'section_context': 'Test', 'chunk_num': 1},
        ]
        facts = [
            {'statements': ['Statement'], 'section_context': 'Test', 'chunk_num': 99},
        ]
        
        examples = create_training_examples(chunks, facts)
        
        assert len(examples) == 0


class TestSaveLoadTrainingData:
    """Tests for save and load training data functions."""
    
    def test_save_and_load_roundtrip(self):
        """Test saving and loading preserves data."""
        trainset = [
            dspy.Example(
                chunk_text='Training text',
                section_context='Train > Section',
                statements=['Statement 1', 'Statement 2'],
            ).with_inputs('chunk_text', 'section_context'),
        ]
        devset = [
            dspy.Example(
                chunk_text='Dev text',
                section_context='Dev > Section',
                statements=['Dev statement'],
            ).with_inputs('chunk_text', 'section_context'),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            
            save_training_data(trainset, devset, data_dir)
            
            loaded_train, loaded_dev = load_training_data(data_dir)
            
            assert len(loaded_train) == 1
            assert len(loaded_dev) == 1
            assert loaded_train[0].chunk_text == 'Training text'
            assert loaded_dev[0].chunk_text == 'Dev text'
    
    def test_save_creates_directory(self):
        """Test save creates directory if needed."""
        trainset = []
        devset = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / 'nested' / 'dir'
            
            save_training_data(trainset, devset, data_dir)
            
            assert data_dir.exists()
            assert (data_dir / 'statement_trainset.json').exists()
