"""Tests for training data module with provenance tracking."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import nbformat

from ontological_engineer.training.data import (
    WikipediaPage,
    WikipediaChunk,
    fetch_top_pages,
    fetch_page_content,
    chunk_article,
    generate_sample_notebook_header,
    append_sample_page_cell,
    generate_chunks_notebook_header,
    append_chunk_cell,
    load_sample_from_notebook,
    load_chunks_from_notebook,
    get_processed_page_titles,
    save_notebook,
)


class TestWikipediaPage:
    """Tests for WikipediaPage dataclass."""
    
    def test_create_basic(self):
        """Test creating a basic WikipediaPage."""
        page = WikipediaPage(title="Test Article", views=1000)
        assert page.title == "Test Article"
        assert page.views == 1000
        assert page.content is None
    
    def test_create_with_content(self):
        """Test creating a WikipediaPage with content."""
        page = WikipediaPage(
            title="Test",
            views=500,
            content="Article content here",
            categories=["Category1", "Category2"],
        )
        assert page.content == "Article content here"
        assert len(page.categories) == 2


class TestWikipediaChunk:
    """Tests for WikipediaChunk dataclass."""
    
    def test_create_chunk(self):
        """Test creating a WikipediaChunk."""
        chunk = WikipediaChunk(
            text="Some chunk text",
            section_context="Article > Section",
            chunk_num=0,
            total_chunks=5,
            page_title="Test Article",
        )
        assert chunk.text == "Some chunk text"
        assert chunk.section_context == "Article > Section"
        assert chunk.chunk_num == 0
        assert chunk.total_chunks == 5


class TestChunkArticle:
    """Tests for chunk_article function."""
    
    def test_basic_chunking(self):
        """Test basic article chunking."""
        text = """This is the introduction paragraph with enough content to pass the minimum length filter.

== Section 1 ==

This is the content of section 1 which also needs to be long enough to pass filtering requirements.
"""
        chunks = chunk_article("Test Article", text, min_chunk_size=20)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, WikipediaChunk) for c in chunks)
    
    def test_preserves_section_context(self):
        """Test that section context is preserved."""
        text = """This is the introduction paragraph that is long enough to not be filtered out.

== Early Life ==

Albert Einstein was born in 1879 in Ulm, Germany. This section contains biographical information.
"""
        chunks = chunk_article("Albert Einstein", text, min_chunk_size=20)
        
        # Should have context with article title and section
        contexts = [c.section_context for c in chunks]
        assert any("Albert Einstein" in ctx for ctx in contexts)
    
    def test_empty_text_returns_empty(self):
        """Test empty text returns empty list."""
        chunks = chunk_article("Test", "")
        assert chunks == []
    
    def test_none_text_returns_empty(self):
        """Test None text returns empty list."""
        chunks = chunk_article("Test", None)
        assert chunks == []
    
    def test_filters_short_chunks(self):
        """Test that short chunks are filtered."""
        text = "Short.\n\n== Section ==\n\nThis is a longer section with more content that should pass the filter."
        chunks = chunk_article("Test", text, min_chunk_size=50)
        
        # Very short chunks should be filtered
        for chunk in chunks:
            assert len(chunk.text) >= 50
    
    def test_chunk_numbering(self):
        """Test that chunks are numbered correctly."""
        text = """Introduction here.

== Section 1 ==

Content for section 1 which is reasonably long.

== Section 2 ==

Content for section 2 which is also reasonably long.
"""
        chunks = chunk_article("Test", text, min_chunk_size=10)
        
        # Check numbering
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_num == i
            assert chunk.total_chunks == len(chunks)


class TestSampleNotebookGeneration:
    """Tests for sample notebook generation."""
    
    def test_generate_header(self):
        """Test generating sample notebook header."""
        nb = generate_sample_notebook_header(
            sampling_method="power_law",
            sample_size=100,
            source_url="https://example.com",
            alpha=0.5,
        )
        
        assert nb is not None
        assert len(nb.cells) >= 1
        assert "power_law" in nb.cells[0].source
    
    def test_append_page_cell(self):
        """Test appending a page cell with signature."""
        nb = generate_sample_notebook_header(
            sampling_method="test",
            sample_size=1,
        )
        
        page = WikipediaPage(title="Test Page", views=1000)
        cid, sig = append_sample_page_cell(nb, page, index=0, total=1)
        
        # Should have added 2 cells (content + signature)
        assert len(nb.cells) >= 3  # Header + content + signature
        
        # CID should be returned
        assert cid is not None
        assert len(cid) > 0
        
        # Signature should have required fields
        assert "@id" in sig
        assert "_type" in sig
        assert sig["_type"] == "sample_page"


class TestChunksNotebookGeneration:
    """Tests for chunks notebook generation."""
    
    def test_generate_header(self):
        """Test generating chunks notebook header."""
        nb = generate_chunks_notebook_header(
            page_title="Albert Einstein",
            source_url="https://en.wikipedia.org/wiki/Albert_Einstein",
        )
        
        assert nb is not None
        assert len(nb.cells) >= 1
        assert "Albert Einstein" in nb.cells[0].source
    
    def test_append_chunk_cell(self):
        """Test appending a chunk cell with signature."""
        nb = generate_chunks_notebook_header(
            page_title="Test",
            source_url="https://example.com",
        )
        
        chunk = WikipediaChunk(
            text="This is the chunk content.",
            section_context="Test > Section",
            chunk_num=0,
            total_chunks=1,
            page_title="Test",
        )
        
        cid, sig = append_chunk_cell(nb, chunk, source_cid="bafkreitest")
        
        # Should have added 2 cells
        assert len(nb.cells) >= 2
        
        # CID should be returned
        assert cid is not None
        
        # Signature should link to source
        assert sig.get("prov:wasDerivedFrom", {}).get("@id") == "ipfs://bafkreitest"


class TestNotebookSaveLoad:
    """Tests for notebook save/load roundtrip."""
    
    def test_save_and_load_sample(self, tmp_path):
        """Test saving and loading a sample notebook."""
        output_path = tmp_path / "test_sample.ipynb"
        
        # Create and save
        nb = generate_sample_notebook_header(
            sampling_method="test",
            sample_size=2,
        )
        
        pages = [
            WikipediaPage(title="Page 1", views=100),
            WikipediaPage(title="Page 2", views=200),
        ]
        
        for i, page in enumerate(pages):
            append_sample_page_cell(nb, page, index=i, total=len(pages))
        
        save_notebook(nb, output_path)
        
        # Load and verify
        loaded_pages = load_sample_from_notebook(output_path)
        
        assert len(loaded_pages) == 2
        assert loaded_pages[0].title == "Page 1"
        assert loaded_pages[1].title == "Page 2"
    
    def test_save_and_load_chunks(self, tmp_path):
        """Test saving and loading a chunks notebook."""
        output_path = tmp_path / "test_chunks.ipynb"
        
        # Create and save
        nb = generate_chunks_notebook_header(
            page_title="Test Article",
            source_url="https://example.com",
        )
        
        chunks = [
            WikipediaChunk(
                text="First chunk content here.",
                section_context="Test Article > Intro",
                chunk_num=0,
                total_chunks=2,
                page_title="Test Article",
            ),
            WikipediaChunk(
                text="Second chunk content here.",
                section_context="Test Article > Section 1",
                chunk_num=1,
                total_chunks=2,
                page_title="Test Article",
            ),
        ]
        
        for chunk in chunks:
            append_chunk_cell(nb, chunk)
        
        save_notebook(nb, output_path)
        
        # Load and verify
        loaded_chunks = load_chunks_from_notebook(output_path)
        
        assert len(loaded_chunks) == 2
        assert "First chunk" in loaded_chunks[0]["text"]
        assert loaded_chunks[0]["section_context"] == "Test Article > Intro"


class TestGetProcessedPageTitles:
    """Tests for get_processed_page_titles function."""
    
    def test_nonexistent_file_returns_empty(self, tmp_path):
        """Test that nonexistent file returns empty set."""
        result = get_processed_page_titles(tmp_path / "nonexistent.ipynb")
        assert result == set()
    
    def test_returns_page_titles(self, tmp_path):
        """Test that existing pages are returned."""
        output_path = tmp_path / "sample.ipynb"
        
        nb = generate_sample_notebook_header(
            sampling_method="test",
            sample_size=2,
        )
        
        append_sample_page_cell(nb, WikipediaPage(title="Page A", views=100), 0, 2)
        append_sample_page_cell(nb, WikipediaPage(title="Page B", views=200), 1, 2)
        
        save_notebook(nb, output_path)
        
        titles = get_processed_page_titles(output_path)
        
        assert "Page A" in titles
        assert "Page B" in titles


class TestFetchFunctions:
    """Tests for fetch functions (mocked)."""
    
    @patch('ontological_engineer.training.data.requests.get')
    def test_fetch_top_pages_parses_response(self, mock_get):
        """Test that fetch_top_pages parses API response correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "items": [{
                "articles": [
                    {"article": "Test_Article", "views": 1000},
                    {"article": "Another_Article", "views": 500},
                    {"article": "Special:Page", "views": 300},  # Should be filtered
                    {"article": "Main_Page", "views": 10000},  # Should be filtered
                ]
            }]
        }
        mock_get.return_value = mock_response
        
        pages = fetch_top_pages(limit=10)
        
        # Should filter special pages
        assert len(pages) == 2
        assert pages[0].title == "Test Article"  # Underscores replaced with spaces
        assert pages[0].views == 1000
    
    @patch('ontological_engineer.training.data.requests.get')
    def test_fetch_page_content_returns_extract(self, mock_get):
        """Test that fetch_page_content returns article text."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "12345": {
                        "extract": "This is the article content."
                    }
                }
            }
        }
        mock_get.return_value = mock_response
        
        content = fetch_page_content("Test Article")
        
        assert content == "This is the article content."
    
    @patch('ontological_engineer.training.data.requests.get')
    def test_fetch_page_content_returns_none_for_missing(self, mock_get):
        """Test that fetch_page_content returns None for missing page."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "query": {
                "pages": {
                    "-1": {}  # Missing page
                }
            }
        }
        mock_get.return_value = mock_response
        
        content = fetch_page_content("Nonexistent Page")
        
        assert content is None


class TestCIDIntegrity:
    """Tests for CID computation and provenance integrity."""
    
    def test_same_content_same_cid(self, tmp_path):
        """Test that same content produces same CID."""
        nb1 = generate_chunks_notebook_header(page_title="Test", source_url="https://example.com")
        nb2 = generate_chunks_notebook_header(page_title="Test", source_url="https://example.com")
        
        chunk = WikipediaChunk(
            text="Identical content",
            section_context="Test > Section",
            chunk_num=0,
            total_chunks=1,
            page_title="Test",
        )
        
        cid1, _ = append_chunk_cell(nb1, chunk)
        cid2, _ = append_chunk_cell(nb2, chunk)
        
        assert cid1 == cid2
    
    def test_different_content_different_cid(self, tmp_path):
        """Test that different content produces different CID."""
        nb = generate_chunks_notebook_header(page_title="Test", source_url="https://example.com")
        
        chunk1 = WikipediaChunk(
            text="Content A",
            section_context="Test > Section",
            chunk_num=0,
            total_chunks=2,
            page_title="Test",
        )
        chunk2 = WikipediaChunk(
            text="Content B",
            section_context="Test > Section",
            chunk_num=1,
            total_chunks=2,
            page_title="Test",
        )
        
        cid1, _ = append_chunk_cell(nb, chunk1)
        cid2, _ = append_chunk_cell(nb, chunk2)
        
        assert cid1 != cid2
