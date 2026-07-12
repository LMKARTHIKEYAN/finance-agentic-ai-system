"""
Tests for the PDF text chunker.

These tests verify:

* Chunk configuration validation
* DocumentChunk dataclass validation
* Single-page text chunking
* Multi-page document chunking
* Character overlap
* Exact chunk boundaries
* Empty-page handling
* Metadata preservation
* Deterministic chunk identifiers
* Convenience-function behaviour

The tests intentionally do not cover PDF reading, embeddings, vector stores,
retrieval, LLM calls, OCR, or finance calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.rag.document_loader import PDFDocument, PDFPage
from src.rag.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DocumentChunk,
    InvalidChunkConfigurationError,
    TextChunker,
    chunk_document,
)


def create_page(
    *,
    page_number: int = 1,
    text: str = "Budget assumptions",
    source_filename: str = "budget_assumptions.pdf",
) -> PDFPage:
    """Create a valid PDFPage for chunker tests."""

    return PDFPage(
        page_number=page_number,
        text=text,
        source_filename=source_filename,
        is_empty=not bool(text.strip()),
    )


def create_document(
    tmp_path: Path,
    *,
    page_texts: tuple[str, ...],
    source_filename: str = "budget_assumptions.pdf",
) -> PDFDocument:
    """Create a valid PDFDocument using supplied page texts."""

    pages = tuple(
        create_page(
            page_number=index,
            text=text,
            source_filename=source_filename,
        )
        for index, text in enumerate(page_texts, start=1)
    )

    empty_page_numbers = tuple(
        page.page_number for page in pages if page.is_empty
    )

    return PDFDocument(
        source_path=(tmp_path / source_filename).resolve(),
        source_filename=source_filename,
        pages=pages,
        total_pages=len(pages),
        empty_page_numbers=empty_page_numbers,
    )


def test_text_chunker_uses_default_configuration() -> None:
    """The chunker should expose the documented default values."""

    chunker = TextChunker()

    assert chunker.chunk_size == DEFAULT_CHUNK_SIZE
    assert chunker.chunk_overlap == DEFAULT_CHUNK_OVERLAP
    assert chunker.step_size == (
        DEFAULT_CHUNK_SIZE - DEFAULT_CHUNK_OVERLAP
    )


def test_text_chunker_accepts_valid_configuration() -> None:
    """Valid chunk size and overlap should be preserved."""

    chunker = TextChunker(
        chunk_size=500,
        chunk_overlap=100,
    )

    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100
    assert chunker.step_size == 400


@pytest.mark.parametrize(
    "chunk_size",
    [True, 10.5, "100", None],
)
def test_text_chunker_rejects_invalid_chunk_size_type(
    chunk_size: Any,
) -> None:
    """chunk_size must be an integer and must not accept booleans."""

    with pytest.raises(
        TypeError,
        match="chunk_size must be an integer",
    ):
        TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=0,
        )


@pytest.mark.parametrize(
    "chunk_size",
    [0, -1, -100],
)
def test_text_chunker_rejects_non_positive_chunk_size(
    chunk_size: int,
) -> None:
    """chunk_size must be greater than zero."""

    with pytest.raises(
        InvalidChunkConfigurationError,
        match="chunk_size must be greater than zero",
    ):
        TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=0,
        )


@pytest.mark.parametrize(
    "chunk_overlap",
    [True, 1.5, "20", None],
)
def test_text_chunker_rejects_invalid_overlap_type(
    chunk_overlap: Any,
) -> None:
    """chunk_overlap must be an integer and must not accept booleans."""

    with pytest.raises(
        TypeError,
        match="chunk_overlap must be an integer",
    ):
        TextChunker(
            chunk_size=100,
            chunk_overlap=chunk_overlap,
        )


def test_text_chunker_rejects_negative_overlap() -> None:
    """chunk_overlap cannot be negative."""

    with pytest.raises(
        InvalidChunkConfigurationError,
        match="chunk_overlap cannot be negative",
    ):
        TextChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (100, 100),
        (100, 101),
        (10, 20),
    ],
)
def test_text_chunker_rejects_overlap_not_smaller_than_chunk_size(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Overlap must always be smaller than the chunk size."""

    with pytest.raises(
        InvalidChunkConfigurationError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_document_chunk_creates_valid_instance() -> None:
    """DocumentChunk should preserve valid chunk metadata."""

    chunk = DocumentChunk(
        chunk_id="chunk_123",
        text="Budget assumptions",
        source_filename="budget.pdf",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=18,
        metadata={
            "document_category": "budget",
        },
    )

    assert chunk.chunk_id == "chunk_123"
    assert chunk.text == "Budget assumptions"
    assert chunk.source_filename == "budget.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 18
    assert chunk.metadata == {
        "document_category": "budget",
    }


def test_document_chunk_normalizes_string_fields() -> None:
    """Identifier and filename whitespace should be removed."""

    chunk = DocumentChunk(
        chunk_id="  chunk_123  ",
        text="Finance",
        source_filename="  finance.pdf  ",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=7,
    )

    assert chunk.chunk_id == "chunk_123"
    assert chunk.source_filename == "finance.pdf"


def test_document_chunk_copies_metadata() -> None:
    """Chunk metadata should not retain the caller's dictionary object."""

    metadata = {
        "financial_year": "2026",
    }

    chunk = DocumentChunk(
        chunk_id="chunk_123",
        text="Forecast",
        source_filename="forecast.pdf",
        page_number=1,
        chunk_index=0,
        start_char=0,
        end_char=8,
        metadata=metadata,
    )

    assert chunk.metadata == metadata
    assert chunk.metadata is not metadata


def test_document_chunk_rejects_empty_chunk_id() -> None:
    """A chunk identifier cannot be empty."""

    with pytest.raises(
        ValueError,
        match="chunk_id cannot be empty",
    ):
        DocumentChunk(
            chunk_id="   ",
            text="Text",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=4,
        )


def test_document_chunk_rejects_empty_text() -> None:
    """A generated chunk must contain text."""

    with pytest.raises(
        ValueError,
        match="chunk text cannot be empty",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=1,
        )


def test_document_chunk_rejects_empty_source_filename() -> None:
    """Every chunk must preserve a source filename."""

    with pytest.raises(
        ValueError,
        match="source_filename cannot be empty",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="   ",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=4,
        )


@pytest.mark.parametrize(
    "page_number",
    [0, -1],
)
def test_document_chunk_rejects_non_positive_page_number(
    page_number: int,
) -> None:
    """Page numbers must remain one-based positive integers."""

    with pytest.raises(
        ValueError,
        match="page_number must be greater than zero",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="document.pdf",
            page_number=page_number,
            chunk_index=0,
            start_char=0,
            end_char=4,
        )


@pytest.mark.parametrize(
    "chunk_index",
    [-1, -10],
)
def test_document_chunk_rejects_negative_chunk_index(
    chunk_index: int,
) -> None:
    """Chunk indexes cannot be negative."""

    with pytest.raises(
        ValueError,
        match="chunk_index cannot be negative",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=chunk_index,
            start_char=0,
            end_char=4,
        )


def test_document_chunk_rejects_negative_start_char() -> None:
    """start_char cannot be negative."""

    with pytest.raises(
        ValueError,
        match="start_char cannot be negative",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=0,
            start_char=-1,
            end_char=3,
        )


@pytest.mark.parametrize(
    ("start_char", "end_char"),
    [
        (0, 0),
        (5, 5),
        (10, 9),
    ],
)
def test_document_chunk_rejects_invalid_end_char(
    start_char: int,
    end_char: int,
) -> None:
    """end_char must be greater than start_char."""

    with pytest.raises(
        ValueError,
        match="end_char must be greater than start_char",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=0,
            start_char=start_char,
            end_char=end_char,
        )


def test_document_chunk_rejects_offset_length_mismatch() -> None:
    """The character range must match the chunk text length."""

    with pytest.raises(
        ValueError,
        match="Character offsets must match",
    ):
        DocumentChunk(
            chunk_id="chunk_1",
            text="Text",
            source_filename="document.pdf",
            page_number=1,
            chunk_index=0,
            start_char=0,
            end_char=10,
        )


def test_chunk_text_returns_empty_tuple_for_empty_text() -> None:
    """Empty or whitespace-only text should not produce chunks."""

    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    assert chunker.chunk_text(
        text="",
        source_filename="document.pdf",
        page_number=1,
    ) == ()

    assert chunker.chunk_text(
        text="   \n\t ",
        source_filename="document.pdf",
        page_number=1,
    ) == ()


def test_chunk_text_returns_one_chunk_when_text_is_smaller_than_limit() -> None:
    """Short text should produce a single complete chunk."""

    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    chunks = chunker.chunk_text(
        text="Budget data",
        source_filename="budget.pdf",
        page_number=2,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.text == "Budget data"
    assert chunk.source_filename == "budget.pdf"
    assert chunk.page_number == 2
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 11


def test_chunk_text_returns_one_chunk_at_exact_chunk_size() -> None:
    """Text equal to chunk_size should produce exactly one chunk."""

    chunker = TextChunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    chunks = chunker.chunk_text(
        text="abcdefghij",
        source_filename="document.pdf",
        page_number=1,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "abcdefghij"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 10


def test_chunk_text_splits_text_without_overlap() -> None:
    """Zero-overlap chunking should create adjacent character ranges."""

    chunker = TextChunker(
        chunk_size=5,
        chunk_overlap=0,
    )

    chunks = chunker.chunk_text(
        text="abcdefghijkl",
        source_filename="document.pdf",
        page_number=1,
    )

    assert len(chunks) == 3

    assert tuple(chunk.text for chunk in chunks) == (
        "abcde",
        "fghij",
        "kl",
    )

    assert tuple(
        (chunk.start_char, chunk.end_char)
        for chunk in chunks
    ) == (
        (0, 5),
        (5, 10),
        (10, 12),
    )

    assert tuple(chunk.chunk_index for chunk in chunks) == (
        0,
        1,
        2,
    )


def test_chunk_text_applies_character_overlap() -> None:
    """Consecutive chunks should repeat the configured overlap."""

    chunker = TextChunker(
        chunk_size=5,
        chunk_overlap=2,
    )

    chunks = chunker.chunk_text(
        text="abcdefghij",
        source_filename="document.pdf",
        page_number=1,
    )

    assert len(chunks) == 3

    assert tuple(chunk.text for chunk in chunks) == (
        "abcde",
        "defgh",
        "ghij",
    )

    assert tuple(
        (chunk.start_char, chunk.end_char)
        for chunk in chunks
    ) == (
        (0, 5),
        (3, 8),
        (6, 10),
    )

    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].text[-2:] == chunks[2].text[:2]


def test_chunk_text_trims_outer_whitespace_before_chunking() -> None:
    """Direct text chunking should normalize outer whitespace."""

    chunker = TextChunker(
        chunk_size=5,
        chunk_overlap=0,
    )

    chunks = chunker.chunk_text(
        text="  abcdef  ",
        source_filename="document.pdf",
        page_number=1,
    )

    assert tuple(chunk.text for chunk in chunks) == (
        "abcde",
        "f",
    )

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == 6


def test_chunk_page_returns_no_chunks_for_empty_page() -> None:
    """An empty PDFPage should be retained upstream but not chunked."""

    page = create_page(
        page_number=3,
        text="",
        source_filename="finance_policy.pdf",
    )

    chunks = TextChunker(
        chunk_size=10,
        chunk_overlap=2,
    ).chunk_page(page)

    assert chunks == ()


def test_chunk_page_preserves_page_metadata() -> None:
    """Page number and source filename should appear on every chunk."""

    page = create_page(
        page_number=4,
        text="abcdefghij",
        source_filename="forecast_methodology.pdf",
    )

    chunks = TextChunker(
        chunk_size=5,
        chunk_overlap=1,
    ).chunk_page(page)

    assert len(chunks) == 3

    assert all(
        chunk.page_number == 4
        for chunk in chunks
    )

    assert all(
        chunk.source_filename == "forecast_methodology.pdf"
        for chunk in chunks
    )


def test_chunk_document_processes_multiple_pages_in_order(
    tmp_path: Path,
) -> None:
    """Chunks should follow document page order."""

    document = create_document(
        tmp_path,
        page_texts=(
            "abcdefgh",
            "ijklmnop",
        ),
        source_filename="finance_policy.pdf",
    )

    chunks = TextChunker(
        chunk_size=5,
        chunk_overlap=0,
    ).chunk_document(document)

    assert tuple(
        (chunk.page_number, chunk.chunk_index, chunk.text)
        for chunk in chunks
    ) == (
        (1, 0, "abcde"),
        (1, 1, "fgh"),
        (2, 0, "ijklm"),
        (2, 1, "nop"),
    )


def test_chunk_document_skips_empty_pages(
    tmp_path: Path,
) -> None:
    """Empty pages should not produce placeholder chunks."""

    document = create_document(
        tmp_path,
        page_texts=(
            "Page one",
            "",
            "Page three",
        ),
    )

    chunks = TextChunker(
        chunk_size=50,
        chunk_overlap=5,
    ).chunk_document(document)

    assert len(chunks) == 2

    assert tuple(chunk.page_number for chunk in chunks) == (
        1,
        3,
    )

    assert tuple(chunk.text for chunk in chunks) == (
        "Page one",
        "Page three",
    )


def test_chunk_document_returns_empty_tuple_when_all_pages_are_empty(
    tmp_path: Path,
) -> None:
    """A document containing only empty pages should produce no chunks."""

    document = create_document(
        tmp_path,
        page_texts=(
            "",
            "   ",
        ),
    )

    chunks = TextChunker().chunk_document(document)

    assert chunks == ()


def test_chunk_document_preserves_custom_metadata(
    tmp_path: Path,
) -> None:
    """Document-level metadata should be added to every chunk."""

    document = create_document(
        tmp_path,
        page_texts=(
            "abcdefghijkl",
        ),
    )

    metadata = {
        "document_category": "budget_assumptions",
        "financial_year": "2026",
        "business_unit": "Chennai",
        "version": "1.0",
        "access_level": "finance",
    }

    chunks = TextChunker(
        chunk_size=5,
        chunk_overlap=1,
    ).chunk_document(
        document=document,
        metadata=metadata,
    )

    assert len(chunks) == 3

    for chunk in chunks:
        assert chunk.metadata["document_category"] == (
            "budget_assumptions"
        )
        assert chunk.metadata["financial_year"] == "2026"
        assert chunk.metadata["business_unit"] == "Chennai"
        assert chunk.metadata["version"] == "1.0"
        assert chunk.metadata["access_level"] == "finance"


def test_required_metadata_overrides_conflicting_custom_values() -> None:
    """System-generated metadata must remain internally consistent."""

    chunks = TextChunker(
        chunk_size=10,
        chunk_overlap=0,
    ).chunk_text(
        text="Forecast",
        source_filename="forecast.pdf",
        page_number=2,
        metadata={
            "chunk_id": "wrong-id",
            "source_filename": "wrong.pdf",
            "page_number": 99,
            "chunk_index": 99,
            "start_char": 99,
            "end_char": 999,
            "financial_year": "2026",
        },
    )

    chunk = chunks[0]

    assert chunk.metadata["chunk_id"] == chunk.chunk_id
    assert chunk.metadata["source_filename"] == "forecast.pdf"
    assert chunk.metadata["page_number"] == 2
    assert chunk.metadata["chunk_index"] == 0
    assert chunk.metadata["start_char"] == 0
    assert chunk.metadata["end_char"] == 8
    assert chunk.metadata["financial_year"] == "2026"


def test_chunk_metadata_is_independent_between_chunks() -> None:
    """Each chunk should receive its own metadata dictionary."""

    chunks = TextChunker(
        chunk_size=5,
        chunk_overlap=0,
    ).chunk_text(
        text="abcdefghij",
        source_filename="document.pdf",
        page_number=1,
        metadata={
            "category": "policy",
        },
    )

    assert len(chunks) == 2
    assert chunks[0].metadata is not chunks[1].metadata


def test_chunk_ids_are_deterministic() -> None:
    """Repeated chunking of the same input should produce identical IDs."""

    chunker = TextChunker(
        chunk_size=5,
        chunk_overlap=2,
    )

    first_result = chunker.chunk_text(
        text="abcdefghij",
        source_filename="document.pdf",
        page_number=1,
    )

    second_result = chunker.chunk_text(
        text="abcdefghij",
        source_filename="document.pdf",
        page_number=1,
    )

    assert tuple(
        chunk.chunk_id for chunk in first_result
    ) == tuple(
        chunk.chunk_id for chunk in second_result
    )


def test_chunk_ids_differ_for_different_pages() -> None:
    """Identical text on different pages should receive different IDs."""

    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=0,
    )

    page_one_chunks = chunker.chunk_text(
        text="Same text",
        source_filename="document.pdf",
        page_number=1,
    )

    page_two_chunks = chunker.chunk_text(
        text="Same text",
        source_filename="document.pdf",
        page_number=2,
    )

    assert (
        page_one_chunks[0].chunk_id
        != page_two_chunks[0].chunk_id
    )


def test_chunk_ids_differ_for_different_source_files() -> None:
    """Identical page text from different documents should have unique IDs."""

    chunker = TextChunker(
        chunk_size=20,
        chunk_overlap=0,
    )

    first_chunks = chunker.chunk_text(
        text="Same text",
        source_filename="budget.pdf",
        page_number=1,
    )

    second_chunks = chunker.chunk_text(
        text="Same text",
        source_filename="forecast.pdf",
        page_number=1,
    )

    assert (
        first_chunks[0].chunk_id
        != second_chunks[0].chunk_id
    )


def test_chunk_ids_use_expected_prefix() -> None:
    """Generated identifiers should use the chunk_ prefix."""

    chunks = TextChunker(
        chunk_size=20,
        chunk_overlap=0,
    ).chunk_text(
        text="Finance policy",
        source_filename="policy.pdf",
        page_number=1,
    )

    assert chunks[0].chunk_id.startswith("chunk_")
    assert len(chunks[0].chunk_id) == len("chunk_") + 64


def test_chunk_text_rejects_invalid_text_type() -> None:
    """Direct chunking should require string text."""

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        TextChunker().chunk_text(
            text=None,  # type: ignore[arg-type]
            source_filename="document.pdf",
            page_number=1,
        )


def test_chunk_text_rejects_empty_source_filename() -> None:
    """Direct chunking should require a source filename."""

    with pytest.raises(
        ValueError,
        match="source_filename cannot be empty",
    ):
        TextChunker().chunk_text(
            text="Finance",
            source_filename="   ",
            page_number=1,
        )


@pytest.mark.parametrize(
    "page_number",
    [0, -1],
)
def test_chunk_text_rejects_invalid_page_number(
    page_number: int,
) -> None:
    """Direct chunking should require a positive page number."""

    with pytest.raises(
        ValueError,
        match="page_number must be greater than zero",
    ):
        TextChunker().chunk_text(
            text="Finance",
            source_filename="document.pdf",
            page_number=page_number,
        )


def test_chunk_page_rejects_invalid_page_type() -> None:
    """chunk_page should accept only PDFPage instances."""

    with pytest.raises(
        TypeError,
        match="page must be a PDFPage",
    ):
        TextChunker().chunk_page(
            "not-a-page"  # type: ignore[arg-type]
        )


def test_chunk_document_rejects_invalid_document_type() -> None:
    """chunk_document should accept only PDFDocument instances."""

    with pytest.raises(
        TypeError,
        match="document must be a PDFDocument",
    ):
        TextChunker().chunk_document(
            "not-a-document"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "metadata",
    [
        "invalid",
        100,
        ["category", "budget"],
    ],
)
def test_chunker_rejects_non_mapping_metadata(
    metadata: Any,
) -> None:
    """Optional metadata must implement the Mapping interface."""

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        TextChunker().chunk_text(
            text="Finance",
            source_filename="document.pdf",
            page_number=1,
            metadata=metadata,
        )


def test_chunk_document_convenience_function_matches_class(
    tmp_path: Path,
) -> None:
    """The convenience function should match direct class usage."""

    document = create_document(
        tmp_path,
        page_texts=(
            "abcdefghijklmnop",
            "qrstuvwxyz",
        ),
        source_filename="forecast_methodology.pdf",
    )

    metadata = {
        "document_category": "forecast",
        "financial_year": "2026",
    }

    direct_result = TextChunker(
        chunk_size=6,
        chunk_overlap=2,
    ).chunk_document(
        document=document,
        metadata=metadata,
    )

    convenience_result = chunk_document(
        document=document,
        chunk_size=6,
        chunk_overlap=2,
        metadata=metadata,
    )

    assert convenience_result == direct_result