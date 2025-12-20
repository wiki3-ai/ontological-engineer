"""
Bootstrap training data from existing pipeline outputs.

This module provides utilities to load chunks and extracted facts from
existing notebook outputs and convert them into DSPy training examples.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import dspy


def load_chunks_from_notebook(notebook_path: Path) -> List[Dict[str, Any]]:
    """
    Load chunks from a chunks.ipynb notebook.
    
    Args:
        notebook_path: Path to the chunks.ipynb file
        
    Returns:
        List of chunk dicts with 'text' and 'section_context' keys
    """
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    chunks = []
    current_chunk = None
    
    for cell in nb['cells']:
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])
        
        if cell_type == 'markdown' and '**Context:**' in source:
            # Parse chunk metadata and content
            chunk = _parse_chunk_cell(source)
            if chunk:
                current_chunk = chunk
                
        elif cell_type == 'raw' and current_chunk:
            # This is the signature cell - save the chunk
            try:
                sig = json.loads(source)
                if sig.get('type') == 'chunk':
                    current_chunk['cid'] = sig.get('cid')
                    chunks.append(current_chunk)
            except json.JSONDecodeError:
                pass
            current_chunk = None
    
    return chunks


def _parse_chunk_cell(source: str) -> Optional[Dict[str, Any]]:
    """Parse a chunk markdown cell into structured data."""
    lines = source.strip().split('\n')
    
    context = None
    chunk_num = None
    text_lines = []
    in_text = False
    
    for line in lines:
        if line.startswith('**Context:**'):
            context = line.replace('**Context:**', '').strip()
        elif line.startswith('**Chunk:**'):
            match = re.search(r'(\d+) of (\d+)', line)
            if match:
                chunk_num = int(match.group(1))
        elif line == '---':
            in_text = True
        elif in_text:
            text_lines.append(line)
    
    if not text_lines:
        return None
    
    return {
        'text': '\n'.join(text_lines).strip(),
        'section_context': context or '',
        'chunk_num': chunk_num,
    }


def load_facts_from_notebook(notebook_path: Path) -> List[Dict[str, Any]]:
    """
    Load extracted facts from a facts.ipynb notebook.
    
    Args:
        notebook_path: Path to the facts.ipynb file
        
    Returns:
        List of fact set dicts with 'statements' and 'section_context' keys
    """
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    fact_sets = []
    current_facts = None
    
    for cell in nb['cells']:
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])
        
        if cell_type == 'markdown' and '**Context:**' in source:
            # Parse facts metadata and content
            facts = _parse_facts_cell(source)
            if facts:
                current_facts = facts
                
        elif cell_type == 'raw' and current_facts:
            # This is the signature cell
            try:
                sig = json.loads(source)
                if sig.get('type') == 'facts':
                    current_facts['cid'] = sig.get('cid')
                    current_facts['from_cid'] = sig.get('from_cid')
                    fact_sets.append(current_facts)
            except json.JSONDecodeError:
                pass
            current_facts = None
    
    return fact_sets


def _parse_facts_cell(source: str) -> Optional[Dict[str, Any]]:
    """Parse a facts markdown cell into structured data."""
    lines = source.strip().split('\n')
    
    context = None
    chunk_num = None
    statements = []
    in_facts = False
    
    for line in lines:
        if line.startswith('**Context:**'):
            context = line.replace('**Context:**', '').strip()
        elif line.startswith('**Chunk:**'):
            match = re.search(r'(\d+) of (\d+)', line)
            if match:
                chunk_num = int(match.group(1))
        elif line == '---':
            in_facts = True
        elif in_facts and line.startswith('- '):
            statements.append(line[2:].strip())
    
    if not statements:
        return None
    
    return {
        'statements': statements,
        'section_context': context or '',
        'chunk_num': chunk_num,
    }


def create_training_examples(
    chunks: List[Dict[str, Any]],
    facts: List[Dict[str, Any]],
    min_statements: int = 2,
    max_statements: int = 50,
) -> List[dspy.Example]:
    """
    Create DSPy training examples from chunks and facts.
    
    Args:
        chunks: List of chunk dicts from load_chunks_from_notebook
        facts: List of fact set dicts from load_facts_from_notebook
        min_statements: Minimum number of statements to include chunk
        max_statements: Maximum number of statements to include chunk
        
    Returns:
        List of dspy.Example objects ready for training
    """
    # Build mapping from chunk number to data
    chunk_map = {c['chunk_num']: c for c in chunks if c.get('chunk_num')}
    facts_map = {f['chunk_num']: f for f in facts if f.get('chunk_num')}
    
    examples = []
    
    for chunk_num in sorted(chunk_map.keys()):
        chunk = chunk_map.get(chunk_num)
        fact_set = facts_map.get(chunk_num)
        
        if not chunk or not fact_set:
            continue
        
        statements = fact_set['statements']
        
        # Filter by statement count
        if len(statements) < min_statements or len(statements) > max_statements:
            continue
        
        example = dspy.Example(
            chunk_text=chunk['text'],
            section_context=chunk['section_context'],
            statements=statements,
        ).with_inputs('chunk_text', 'section_context')
        
        examples.append(example)
    
    return examples


def load_training_data(
    data_dir: Path,
    train_file: str = "statement_trainset.json",
    dev_file: str = "statement_devset.json",
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """
    Load saved training data from JSON files.
    
    Args:
        data_dir: Directory containing the JSON files
        train_file: Filename for training set
        dev_file: Filename for dev set
        
    Returns:
        Tuple of (trainset, devset) as lists of dspy.Example
    """
    def load_file(path: Path) -> List[dspy.Example]:
        with open(path, 'r') as f:
            data = json.load(f)
        
        return [
            dspy.Example(**item).with_inputs('chunk_text', 'section_context')
            for item in data
        ]
    
    trainset = load_file(data_dir / train_file)
    devset = load_file(data_dir / dev_file)
    
    return trainset, devset


def save_training_data(
    trainset: List[dspy.Example],
    devset: List[dspy.Example],
    data_dir: Path,
    train_file: str = "statement_trainset.json",
    dev_file: str = "statement_devset.json",
) -> None:
    """
    Save training data to JSON files.
    
    Args:
        trainset: List of training examples
        devset: List of dev examples
        data_dir: Directory to save files
        train_file: Filename for training set
        dev_file: Filename for dev set
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    
    def example_to_dict(ex: dspy.Example) -> Dict[str, Any]:
        return {
            'chunk_text': ex.chunk_text,
            'section_context': ex.section_context,
            'statements': ex.statements,
        }
    
    with open(data_dir / train_file, 'w') as f:
        json.dump([example_to_dict(ex) for ex in trainset], f, indent=2)
    
    with open(data_dir / dev_file, 'w') as f:
        json.dump([example_to_dict(ex) for ex in devset], f, indent=2)
