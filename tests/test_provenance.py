"""Tests for provenance tracking module."""

import pytest
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_raw_cell

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


# Mock StatementClassification for testing
@dataclass
class MockClassification:
    """Mock classification for testing."""
    index: int
    statement: str
    classification: str
    reason: str
    
    @property
    def is_good(self) -> bool:
        return self.classification.upper() == "GOOD"


class TestStatementsNotebook:
    """Tests for statements notebook generation."""
    
    def test_generate_header_creates_valid_notebook(self):
        """Test header generation creates valid notebook structure."""
        provenance = {
            "article_title": "Albert Einstein",
            "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
            "license": "CC BY-SA 4.0",
        }
        model_config = {
            "model": "qwen/qwen3-30b",
            "temperature": 0.7,
        }
        
        nb = generate_statements_notebook_header(provenance, model_config)
        
        assert nb is not None
        assert len(nb.cells) == 2  # Markdown header + raw config
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[1].cell_type == "raw"
    
    def test_generate_header_includes_provenance(self):
        """Test header includes provenance information."""
        provenance = {
            "article_title": "Test Article",
            "source_url": "https://example.com",
        }
        model_config = {"model": "test-model"}
        
        nb = generate_statements_notebook_header(provenance, model_config)
        
        header = nb.cells[0].source
        assert "Test Article" in header
        assert "https://example.com" in header
        assert "test-model" in header
    
    def test_append_statements_cell_creates_content(self):
        """Test appending statements creates markdown and signature."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        statements = [
            "[Einstein](/wiki/Albert_Einstein) was born in 1879.",
            "[Einstein](/wiki/Albert_Einstein) won the Nobel Prize.",
        ]
        
        cid, sig = append_statements_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=5,
            section_context="Albert Einstein > Early life",
            statements=statements,
            chunk_cid="bafytest123",
        )
        
        assert len(nb.cells) == 4  # Header + config + statements + signature
        assert nb.cells[2].cell_type == "markdown"
        assert nb.cells[3].cell_type == "raw"
        assert cid is not None
        assert len(cid) > 0
        assert sig["_type"] == "statements"
    
    def test_append_statements_preserves_all_statements(self):
        """Test all statements are included in the cell."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        statements = [
            "Statement one.",
            "Statement two.",
            "Statement three.",
        ]
        
        append_statements_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=1,
            section_context="Test",
            statements=statements,
            chunk_cid="test",
        )
        
        content = nb.cells[2].source
        assert "1. Statement one." in content
        assert "2. Statement two." in content
        assert "3. Statement three." in content
    
    def test_signature_links_to_chunk_cid(self):
        """Test signature references source chunk CID."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        chunk_cid = "bafyexample123"
        _, sig = append_statements_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=1,
            section_context="Test",
            statements=["Test statement."],
            chunk_cid=chunk_cid,
        )
        
        # Check wasDerivedFrom references the chunk
        from_ref = sig.get("prov:wasDerivedFrom", {})
        assert f"ipfs://{chunk_cid}" in from_ref.get("@id", "")
    
    def test_save_and_load_roundtrip(self):
        """Test saving and loading preserves data."""
        provenance = {"article_title": "Test"}
        model_config = {"model": "test"}
        
        nb = generate_statements_notebook_header(provenance, model_config)
        
        statements = [
            "Statement A.",
            "Statement B.",
        ]
        cid1, _ = append_statements_cell(
            nb, 1, 2, "Context 1", statements, "chunk_cid_1"
        )
        
        statements2 = ["Statement C."]
        cid2, _ = append_statements_cell(
            nb, 2, 2, "Context 2", statements2, "chunk_cid_2"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "statements.ipynb"
            save_statements_notebook(nb, path)
            
            loaded = load_statements_from_notebook(path)
            
            assert len(loaded) == 2
            assert loaded[0]["chunk_num"] == 1
            assert loaded[0]["section_context"] == "Context 1"
            assert len(loaded[0]["statements"]) == 2
            assert loaded[1]["chunk_num"] == 2


class TestClassificationsNotebook:
    """Tests for classifications notebook generation."""
    
    def test_generate_header_creates_valid_notebook(self):
        """Test header generation creates valid structure."""
        provenance = {"article_title": "Test Article"}
        model_config = {"model": "classifier-model"}
        
        nb = generate_classifications_notebook_header(provenance, model_config)
        
        assert len(nb.cells) == 2
        assert "Statement Classifications" in nb.cells[0].source
    
    def test_append_classifications_creates_table(self):
        """Test classification cell creates markdown table."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        classifications = [
            MockClassification(0, "Good statement.", "GOOD", "atomic and accurate"),
            MockClassification(1, "Bad statement.", "BAD", "multiple claims"),
        ]
        
        cid, sig = append_classifications_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=1,
            section_context="Test > Section",
            classifications=classifications,
            missing_facts="none",
            score=0.5,
            statements_cid="stmt_cid_123",
        )
        
        assert len(nb.cells) == 4
        content = nb.cells[2].source
        assert "| # | Classification |" in content
        assert "✅ GOOD" in content
        assert "❌ BAD" in content
    
    def test_classification_signature_links_to_statements(self):
        """Test signature references source statements CID."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        classifications = [
            MockClassification(0, "Test.", "GOOD", "ok"),
        ]
        
        statements_cid = "stmt_cid_abc"
        _, sig = append_classifications_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=1,
            section_context="Test",
            classifications=classifications,
            missing_facts="",
            score=1.0,
            statements_cid=statements_cid,
        )
        
        from_ref = sig.get("prov:wasDerivedFrom", {})
        assert statements_cid in from_ref.get("@id", "")
    
    def test_classification_data_embedded_in_signature(self):
        """Test classification data is embedded in signature for machine access."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        classifications = [
            MockClassification(0, "Test statement.", "GOOD", "reason"),
        ]
        
        _, sig = append_classifications_cell(
            nb=nb,
            chunk_num=1,
            total_chunks=1,
            section_context="Test",
            classifications=classifications,
            missing_facts="missing info",
            score=1.0,
            statements_cid="test",
        )
        
        assert "_classification_data" in sig
        data = sig["_classification_data"]
        assert data["score"] == 1.0
        assert data["missing_facts"] == "missing info"
        assert len(data["classifications"]) == 1
        assert data["classifications"][0]["classification"] == "GOOD"
    
    def test_save_and_load_roundtrip(self):
        """Test saving and loading preserves classification data."""
        provenance = {"article_title": "Test"}
        model_config = {"model": "test"}
        
        nb = generate_classifications_notebook_header(provenance, model_config)
        
        classifications = [
            MockClassification(0, "Good one.", "GOOD", "ok"),
            MockClassification(1, "Bad one.", "BAD", "not ok"),
        ]
        
        append_classifications_cell(
            nb, 1, 1, "Context", classifications, "none", 0.5, "stmt_cid"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "classifications.ipynb"
            save_classifications_notebook(nb, path)
            
            loaded = load_classifications_from_notebook(path)
            
            assert len(loaded) == 1
            assert loaded[0]["chunk_num"] == 1
            assert loaded[0]["score"] == 0.5
            assert len(loaded[0]["classifications"]) == 2


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_get_processed_chunk_cids_empty_file(self):
        """Test returns empty set for non-existent file."""
        result = get_processed_chunk_cids(Path("/nonexistent/path.ipynb"))
        assert result == set()
    
    def test_get_processed_chunk_cids_extracts_from_signatures(self):
        """Test extracts from_cid values from notebook signatures."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        # Add a cell with properly formatted signature (matching make_signature output)
        nb.cells.append(new_markdown_cell("Content"))
        sig = {
            "@context": {},
            "@id": "ipfs://result_cid",
            "@type": ["prov:Entity"],
            "dcterms:identifier": "result_cid",
            "prov:label": "test:1",
            "prov:wasDerivedFrom": {"@id": "ipfs://chunk_cid_1"},
            "_cell": 2,
            "_type": "statements",
        }
        nb.cells.append(new_raw_cell(json.dumps(sig)))
        
        # Add another
        nb.cells.append(new_markdown_cell("Content 2"))
        sig2 = {
            "@context": {},
            "@id": "ipfs://result_cid_2",
            "@type": ["prov:Entity"],
            "dcterms:identifier": "result_cid_2",
            "prov:label": "test:2",
            "prov:wasDerivedFrom": {"@id": "ipfs://chunk_cid_2"},
            "_cell": 4,
            "_type": "statements",
        }
        nb.cells.append(new_raw_cell(json.dumps(sig2)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.ipynb"
            with open(path, 'w') as f:
                nbformat.write(nb, f)
            
            result = get_processed_chunk_cids(path)
            
            assert "chunk_cid_1" in result
            assert "chunk_cid_2" in result
    
    def test_create_output_directory_creates_structure(self):
        """Test creates proper directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            result = create_output_directory("Albert Einstein", base)
            
            assert result.exists()
            assert "albert_einstein" in str(result)
            # Check timestamp format in path
            parts = str(result).split("/")
            timestamp_part = parts[-1]
            # Should be YYYYMMDD_HHMMSS format
            assert len(timestamp_part) == 15
            assert "_" in timestamp_part
    
    def test_create_output_directory_handles_slashes(self):
        """Test handles article titles with slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            result = create_output_directory("AC/DC", base)
            
            assert result.exists()
            assert "ac_dc" in str(result)


class TestCIDComputation:
    """Tests for CID computation in provenance."""
    
    def test_different_content_produces_different_cids(self):
        """Test different content produces different CIDs."""
        nb = new_notebook()
        nb.cells.append(new_markdown_cell("Header"))
        nb.cells.append(new_raw_cell("{}"))
        
        cid1, _ = append_statements_cell(
            nb, 1, 2, "Context", ["Statement A."], "chunk_1"
        )
        cid2, _ = append_statements_cell(
            nb, 2, 2, "Context", ["Statement B."], "chunk_2"
        )
        
        assert cid1 != cid2
    
    def test_same_content_produces_same_cid(self):
        """Test identical content produces same CID (deterministic)."""
        nb1 = new_notebook()
        nb1.cells.append(new_markdown_cell("Header"))
        nb1.cells.append(new_raw_cell("{}"))
        
        nb2 = new_notebook()
        nb2.cells.append(new_markdown_cell("Header"))
        nb2.cells.append(new_raw_cell("{}"))
        
        cid1, _ = append_statements_cell(
            nb1, 1, 1, "Same Context", ["Same statement."], "same_chunk"
        )
        cid2, _ = append_statements_cell(
            nb2, 1, 1, "Same Context", ["Same statement."], "same_chunk"
        )
        
        assert cid1 == cid2


class TestIntegrationWithRealClassifications:
    """Integration tests using the actual StatementClassification class."""
    
    def test_with_real_classification_class(self):
        """Test works with actual StatementClassification from judges module."""
        try:
            from ontological_engineer.judges import StatementClassification
            
            nb = new_notebook()
            nb.cells.append(new_markdown_cell("Header"))
            nb.cells.append(new_raw_cell("{}"))
            
            classifications = [
                StatementClassification(
                    index=0,
                    statement="Test statement.",
                    classification="GOOD",
                    reason="Well formed",
                ),
            ]
            
            cid, sig = append_classifications_cell(
                nb=nb,
                chunk_num=1,
                total_chunks=1,
                section_context="Test",
                classifications=classifications,
                missing_facts="none",
                score=1.0,
                statements_cid="test_cid",
            )
            
            assert cid is not None
            assert sig["_type"] == "classifications"
            
        except ImportError:
            pytest.skip("StatementClassification not available")
