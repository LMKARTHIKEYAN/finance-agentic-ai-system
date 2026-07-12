"""
Tests for the PDF ingestion service.

These tests verify:

* Ingestion result dataclasses
* Single-PDF ingestion
* Batch ingestion
* Directory ingestion
* Metadata propagation
* Empty-page handling
* Duplicate-document detection
* Failure recording
* Convenience-function behaviour

The tests intentionally do not verify PDF parsing internals, chunking
algorithms, embedding mathematics, retrieval, LLM calls, pgvector, S3,
FastAPI, Streamlit, or finance calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from src.rag.document_loader import (
    DocumentFileNotFoundError,
    PDFDocument,
    PDFPage,
)
from src.rag.embeddings import DeterministicEmbeddingService
from src.rag.ingestion import (
    BatchIngestionResult,
    DirectoryIngestionError,
    DocumentIngestionResult,
    DuplicateDocumentError,
    IngestionFailure,
    PDFIngestionService,
    ingest_pdf,
)
from src.rag.text_chunker import TextChunker
from src.rag.vector_store import InMemoryVectorStore


def create_blank_pdf(
    file_path: Path,
    *,
    page_count: int = 1,
) -> Path:
    """Create a real blank PDF for filesystem-level ingestion tests."""

    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(
            width=612,
            height=792,
        )

    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    return file_path


def create_page(
    *,
    page_number: int,
    text: str,
    source_filename: str,
) -> PDFPage:
    """Create a valid page result."""

    return PDFPage(
        page_number=page_number,
        text=text,
        source_filename=source_filename,
        is_empty=not bool(text.strip()),
    )


def create_document(
    tmp_path: Path,
    *,
    source_filename: str = "finance_policy.pdf",
    page_texts: tuple[str, ...] = (
        "Finance policy and approval process.",
    ),
) -> PDFDocument:
    """Create a typed PDFDocument without reading a real PDF."""

    pages = tuple(
        create_page(
            page_number=index,
            text=text,
            source_filename=source_filename,
        )
        for index, text in enumerate(
            page_texts,
            start=1,
        )
    )

    return PDFDocument(
        source_path=(
            tmp_path / source_filename
        ).resolve(),
        source_filename=source_filename,
        pages=pages,
        total_pages=len(pages),
        empty_page_numbers=tuple(
            page.page_number
            for page in pages
            if page.is_empty
        ),
    )


class StubDocumentLoader:
    """Test loader that returns configured documents by filename."""

    def __init__(
        self,
        documents: dict[str, PDFDocument],
    ) -> None:
        self.documents = documents
        self.loaded_paths: list[Path] = []

    def load(
        self,
        file_path: str | Path,
    ) -> PDFDocument:
        """Return a configured document or raise a missing-file error."""

        path = Path(file_path)
        self.loaded_paths.append(path)

        try:
            return self.documents[path.name]
        except KeyError as exc:
            raise DocumentFileNotFoundError(
                f"PDF file does not exist: '{path}'."
            ) from exc


def create_vector_store() -> InMemoryVectorStore:
    """Create an in-memory store with deterministic local embeddings."""

    embedding_service = DeterministicEmbeddingService(
        dimension=32,
    )

    return InMemoryVectorStore(
        embedding_service=embedding_service,
    )


def test_ingestion_failure_creates_valid_instance(
    tmp_path: Path,
) -> None:
    """Failure details should preserve path and error information."""

    failure = IngestionFailure(
        source_path=tmp_path / "missing.pdf",
        error_type="DocumentFileNotFoundError",
        message="PDF file does not exist.",
    )

    assert failure.source_path == (
        tmp_path / "missing.pdf"
    )
    assert failure.error_type == (
        "DocumentFileNotFoundError"
    )
    assert failure.message == (
        "PDF file does not exist."
    )


def test_ingestion_failure_normalizes_strings(
    tmp_path: Path,
) -> None:
    """Error type and message should be trimmed."""

    failure = IngestionFailure(
        source_path=tmp_path / "missing.pdf",
        error_type="  ValueError  ",
        message="  Invalid PDF  ",
    )

    assert failure.error_type == "ValueError"
    assert failure.message == "Invalid PDF"


def test_ingestion_failure_rejects_empty_error_type(
    tmp_path: Path,
) -> None:
    """Failure error type cannot be empty."""

    with pytest.raises(
        ValueError,
        match="error_type cannot be empty",
    ):
        IngestionFailure(
            source_path=tmp_path / "missing.pdf",
            error_type=" ",
            message="Failure",
        )


def test_ingestion_failure_rejects_empty_message(
    tmp_path: Path,
) -> None:
    """Failure message cannot be empty."""

    with pytest.raises(
        ValueError,
        match="message cannot be empty",
    ):
        IngestionFailure(
            source_path=tmp_path / "missing.pdf",
            error_type="ValueError",
            message=" ",
        )


def test_document_ingestion_result_creates_valid_instance(
    tmp_path: Path,
) -> None:
    """A valid single-document result should retain all totals."""

    result = DocumentIngestionResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        source_filename="budget.pdf",
        total_pages=3,
        empty_page_numbers=(2,),
        chunks_created=2,
        vectors_stored=2,
        document_ids=(
            "chunk_a",
            "chunk_b",
        ),
        metadata={
            "document_category": "budget",
        },
    )

    assert result.source_filename == "budget.pdf"
    assert result.total_pages == 3
    assert result.empty_page_numbers == (2,)
    assert result.chunks_created == 2
    assert result.vectors_stored == 2
    assert result.document_ids == (
        "chunk_a",
        "chunk_b",
    )
    assert result.has_empty_pages is True


def test_document_ingestion_result_reports_no_empty_pages(
    tmp_path: Path,
) -> None:
    """has_empty_pages should be false when no page is empty."""

    result = DocumentIngestionResult(
        source_path=(
            tmp_path / "forecast.pdf"
        ).resolve(),
        source_filename="forecast.pdf",
        total_pages=1,
        empty_page_numbers=(),
        chunks_created=1,
        vectors_stored=1,
        document_ids=("chunk_a",),
    )

    assert result.has_empty_pages is False


def test_document_ingestion_result_rejects_vector_count_mismatch(
    tmp_path: Path,
) -> None:
    """Stored vector count must match created chunks."""

    with pytest.raises(
        ValueError,
        match="vectors_stored must match chunks_created",
    ):
        DocumentIngestionResult(
            source_path=(
                tmp_path / "document.pdf"
            ).resolve(),
            source_filename="document.pdf",
            total_pages=1,
            empty_page_numbers=(),
            chunks_created=2,
            vectors_stored=1,
            document_ids=("chunk_a",),
        )


def test_document_ingestion_result_rejects_document_id_count_mismatch(
    tmp_path: Path,
) -> None:
    """Number of identifiers must match stored vectors."""

    with pytest.raises(
        ValueError,
        match="document_ids must match vectors_stored",
    ):
        DocumentIngestionResult(
            source_path=(
                tmp_path / "document.pdf"
            ).resolve(),
            source_filename="document.pdf",
            total_pages=1,
            empty_page_numbers=(),
            chunks_created=2,
            vectors_stored=2,
            document_ids=("chunk_a",),
        )


def test_document_ingestion_result_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    """Stored identifiers must be unique."""

    with pytest.raises(
        ValueError,
        match="document_ids cannot contain duplicates",
    ):
        DocumentIngestionResult(
            source_path=(
                tmp_path / "document.pdf"
            ).resolve(),
            source_filename="document.pdf",
            total_pages=1,
            empty_page_numbers=(),
            chunks_created=2,
            vectors_stored=2,
            document_ids=(
                "chunk_a",
                "chunk_a",
            ),
        )


def test_batch_ingestion_result_creates_valid_instance(
    tmp_path: Path,
) -> None:
    """A batch result should aggregate successes and failures."""

    success = DocumentIngestionResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        source_filename="budget.pdf",
        total_pages=2,
        empty_page_numbers=(),
        chunks_created=3,
        vectors_stored=3,
        document_ids=(
            "chunk_1",
            "chunk_2",
            "chunk_3",
        ),
    )

    failure = IngestionFailure(
        source_path=tmp_path / "missing.pdf",
        error_type="DocumentFileNotFoundError",
        message="Missing file",
    )

    batch = BatchIngestionResult(
        successful_documents=(success,),
        failures=(failure,),
        total_files=2,
        documents_ingested=1,
        documents_failed=1,
        total_pages=2,
        chunks_created=3,
        vectors_stored=3,
    )

    assert batch.succeeded is False
    assert batch.partially_succeeded is True
    assert batch.documents_ingested == 1
    assert batch.documents_failed == 1


def test_batch_result_reports_full_success(
    tmp_path: Path,
) -> None:
    """A batch with no failures should report success."""

    success = DocumentIngestionResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        source_filename="budget.pdf",
        total_pages=1,
        empty_page_numbers=(),
        chunks_created=1,
        vectors_stored=1,
        document_ids=("chunk_1",),
    )

    batch = BatchIngestionResult(
        successful_documents=(success,),
        failures=(),
        total_files=1,
        documents_ingested=1,
        documents_failed=0,
        total_pages=1,
        chunks_created=1,
        vectors_stored=1,
    )

    assert batch.succeeded is True
    assert batch.partially_succeeded is False


def test_ingestion_service_rejects_invalid_loader() -> None:
    """Service dependencies should use supported types."""

    with pytest.raises(
        TypeError,
        match="document_loader must be",
    ):
        PDFIngestionService(
            document_loader="invalid",  # type: ignore[arg-type]
        )


def test_ingestion_service_rejects_invalid_chunker() -> None:
    """Service should reject invalid text chunkers."""

    with pytest.raises(
        TypeError,
        match="text_chunker must be",
    ):
        PDFIngestionService(
            text_chunker="invalid",  # type: ignore[arg-type]
        )


def test_ingestion_service_rejects_invalid_vector_store() -> None:
    """Service should reject unsupported vector stores."""

    with pytest.raises(
        TypeError,
        match="vector_store must be",
    ):
        PDFIngestionService(
            vector_store="invalid",  # type: ignore[arg-type]
        )


def test_ingest_pdf_loads_chunks_and_stores_vectors(
    tmp_path: Path,
) -> None:
    """One PDF should move through loading, chunking, and storage."""

    document = create_document(
        tmp_path,
        source_filename="finance_policy.pdf",
        page_texts=(
            "abcdefghijkl",
        ),
    )

    loader = StubDocumentLoader(
        {
            "finance_policy.pdf": document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        text_chunker=TextChunker(
            chunk_size=5,
            chunk_overlap=1,
        ),
        vector_store=vector_store,
    )

    result = service.ingest_pdf(
        tmp_path / "finance_policy.pdf"
    )

    assert result.source_path == (
        tmp_path / "finance_policy.pdf"
    ).resolve()
    assert result.source_filename == (
        "finance_policy.pdf"
    )
    assert result.total_pages == 1
    assert result.chunks_created == 3
    assert result.vectors_stored == 3
    assert len(result.document_ids) == 3
    assert len(vector_store) == 3


def test_ingest_pdf_preserves_chunk_metadata(
    tmp_path: Path,
) -> None:
    """Custom and required metadata should be stored with vectors."""

    document = create_document(
        tmp_path,
        source_filename="budget.pdf",
        page_texts=(
            "Budget assumptions for FY 2026.",
        ),
    )

    loader = StubDocumentLoader(
        {
            "budget.pdf": document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        text_chunker=TextChunker(
            chunk_size=100,
            chunk_overlap=10,
        ),
        vector_store=vector_store,
    )

    result = service.ingest_pdf(
        tmp_path / "budget.pdf",
        metadata={
            "document_category": "budget_assumptions",
            "financial_year": "2026",
            "business_unit": "Chennai",
            "version": "1.0",
            "access_level": "finance",
        },
    )

    assert result.metadata == {
        "document_category": "budget_assumptions",
        "financial_year": "2026",
        "business_unit": "Chennai",
        "version": "1.0",
        "access_level": "finance",
    }

    stored_document = vector_store.get_document(
        result.document_ids[0]
    )

    assert stored_document.metadata[
        "document_category"
    ] == "budget_assumptions"
    assert stored_document.metadata[
        "financial_year"
    ] == "2026"
    assert stored_document.metadata[
        "business_unit"
    ] == "Chennai"
    assert stored_document.metadata[
        "source_filename"
    ] == "budget.pdf"
    assert stored_document.metadata["page_number"] == 1
    assert stored_document.metadata["chunk_index"] == 0


def test_ingest_pdf_preserves_empty_page_information(
    tmp_path: Path,
) -> None:
    """Empty pages should remain in the ingestion summary."""

    document = create_document(
        tmp_path,
        source_filename="policy.pdf",
        page_texts=(
            "Approval policy",
            "",
            "Escalation policy",
        ),
    )

    loader = StubDocumentLoader(
        {
            "policy.pdf": document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        text_chunker=TextChunker(
            chunk_size=100,
            chunk_overlap=10,
        ),
        vector_store=vector_store,
    )

    result = service.ingest_pdf(
        tmp_path / "policy.pdf"
    )

    assert result.total_pages == 3
    assert result.empty_page_numbers == (2,)
    assert result.has_empty_pages is True
    assert result.chunks_created == 2
    assert result.vectors_stored == 2


def test_ingest_pdf_all_empty_pages_stores_nothing(
    tmp_path: Path,
) -> None:
    """A PDF with no text should succeed without storing vectors."""

    document = create_document(
        tmp_path,
        source_filename="scanned.pdf",
        page_texts=(
            "",
            "   ",
        ),
    )

    loader = StubDocumentLoader(
        {
            "scanned.pdf": document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        vector_store=vector_store,
    )

    result = service.ingest_pdf(
        tmp_path / "scanned.pdf"
    )

    assert result.total_pages == 2
    assert result.empty_page_numbers == (1, 2)
    assert result.chunks_created == 0
    assert result.vectors_stored == 0
    assert result.document_ids == ()
    assert len(vector_store) == 0


def test_ingesting_same_pdf_twice_raises_duplicate_error(
    tmp_path: Path,
) -> None:
    """Deterministic chunk IDs should prevent silent duplicate ingestion."""

    document = create_document(
        tmp_path,
        source_filename="budget.pdf",
        page_texts=(
            "Budget assumptions",
        ),
    )

    loader = StubDocumentLoader(
        {
            "budget.pdf": document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        vector_store=vector_store,
    )

    service.ingest_pdf(
        tmp_path / "budget.pdf"
    )

    original_count = len(vector_store)

    with pytest.raises(
        DuplicateDocumentError,
        match="already been ingested",
    ):
        service.ingest_pdf(
            tmp_path / "budget.pdf"
        )

    assert len(vector_store) == original_count


def test_ingest_pdf_rejects_invalid_metadata(
    tmp_path: Path,
) -> None:
    """Document metadata must be a mapping."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        service.ingest_pdf(
            tmp_path / "document.pdf",
            metadata="invalid",  # type: ignore[arg-type]
        )


def test_ingest_pdfs_processes_multiple_documents(
    tmp_path: Path,
) -> None:
    """Batch ingestion should aggregate successful documents."""

    budget_document = create_document(
        tmp_path,
        source_filename="budget.pdf",
        page_texts=("Budget assumptions",),
    )

    forecast_document = create_document(
        tmp_path,
        source_filename="forecast.pdf",
        page_texts=("Forecast methodology",),
    )

    loader = StubDocumentLoader(
        {
            "budget.pdf": budget_document,
            "forecast.pdf": forecast_document,
        }
    )

    vector_store = create_vector_store()

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        vector_store=vector_store,
    )

    result = service.ingest_pdfs(
        [
            tmp_path / "budget.pdf",
            tmp_path / "forecast.pdf",
        ]
    )

    assert result.total_files == 2
    assert result.documents_ingested == 2
    assert result.documents_failed == 0
    assert result.succeeded is True
    assert result.total_pages == 2
    assert result.chunks_created == 2
    assert result.vectors_stored == 2
    assert len(vector_store) == 2


def test_ingest_pdfs_records_failures_and_continues(
    tmp_path: Path,
) -> None:
    """Batch ingestion should continue after individual failures."""

    valid_document = create_document(
        tmp_path,
        source_filename="valid.pdf",
        page_texts=("Valid finance policy",),
    )

    loader = StubDocumentLoader(
        {
            "valid.pdf": valid_document,
        }
    )

    service = PDFIngestionService(
        document_loader=loader,  # type: ignore[arg-type]
        vector_store=create_vector_store(),
    )

    result = service.ingest_pdfs(
        [
            tmp_path / "missing.pdf",
            tmp_path / "valid.pdf",
        ],
        continue_on_error=True,
    )

    assert result.total_files == 2
    assert result.documents_ingested == 1
    assert result.documents_failed == 1
    assert result.partially_succeeded is True

    failure = result.failures[0]

    assert failure.source_path == (
        tmp_path / "missing.pdf"
    )
    assert failure.error_type == (
        "DocumentFileNotFoundError"
    )
    assert "does not exist" in failure.message


def test_ingest_pdfs_raises_when_continue_on_error_is_false(
    tmp_path: Path,
) -> None:
    """The first failure should propagate when continuation is disabled."""

    service = PDFIngestionService(
        document_loader=StubDocumentLoader({}),  # type: ignore[arg-type]
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        DocumentFileNotFoundError,
    ):
        service.ingest_pdfs(
            [
                tmp_path / "missing.pdf",
            ],
            continue_on_error=False,
        )


def test_ingest_pdfs_accepts_empty_sequence() -> None:
    """An empty batch should return a successful zero-result summary."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    result = service.ingest_pdfs([])

    assert result.total_files == 0
    assert result.documents_ingested == 0
    assert result.documents_failed == 0
    assert result.total_pages == 0
    assert result.chunks_created == 0
    assert result.vectors_stored == 0
    assert result.succeeded is True


@pytest.mark.parametrize(
    "file_paths",
    [
        "document.pdf",
        Path("document.pdf"),
        b"document.pdf",
    ],
)
def test_ingest_pdfs_rejects_single_path_as_sequence(
    file_paths: Any,
) -> None:
    """A single path must be sent to ingest_pdf, not ingest_pdfs."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        TypeError,
        match="sequence of paths, not one path",
    ):
        service.ingest_pdfs(file_paths)


def test_ingest_pdfs_rejects_invalid_path_item() -> None:
    """Every batch item must be a string or Path."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        TypeError,
        match="Each file path must be",
    ):
        service.ingest_pdfs(
            [
                "valid.pdf",
                123,
            ]  # type: ignore[list-item]
        )


def test_ingest_pdfs_rejects_invalid_continue_flag() -> None:
    """continue_on_error must be boolean."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        TypeError,
        match="continue_on_error must be a boolean",
    ):
        service.ingest_pdfs(
            [],
            continue_on_error="yes",  # type: ignore[arg-type]
        )


def test_ingest_directory_processes_pdf_files_only(
    tmp_path: Path,
) -> None:
    """Directory ingestion should ignore non-PDF files."""

    create_blank_pdf(
        tmp_path / "budget.pdf"
    )
    create_blank_pdf(
        tmp_path / "forecast.PDF"
    )

    (tmp_path / "notes.txt").write_text(
        "Ignore this file",
        encoding="utf-8",
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    result = service.ingest_directory(
        tmp_path
    )

    assert result.total_files == 2
    assert result.documents_ingested == 2
    assert result.documents_failed == 0
    assert result.total_pages == 2

    # Blank PDFs have pages but no extractable text.
    assert result.chunks_created == 0
    assert result.vectors_stored == 0


def test_ingest_directory_is_non_recursive_by_default(
    tmp_path: Path,
) -> None:
    """Nested PDF files should be ignored unless recursive is enabled."""

    create_blank_pdf(
        tmp_path / "root.pdf"
    )

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    create_blank_pdf(
        nested_directory / "nested.pdf"
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    result = service.ingest_directory(
        tmp_path
    )

    assert result.total_files == 1


def test_ingest_directory_can_process_nested_pdfs(
    tmp_path: Path,
) -> None:
    """Recursive directory ingestion should include nested PDFs."""

    create_blank_pdf(
        tmp_path / "root.pdf"
    )

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()

    create_blank_pdf(
        nested_directory / "nested.pdf"
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    result = service.ingest_directory(
        tmp_path,
        recursive=True,
    )

    assert result.total_files == 2
    assert result.documents_ingested == 2


def test_ingest_directory_returns_zero_result_when_no_pdfs(
    tmp_path: Path,
) -> None:
    """A valid empty directory should produce a zero-result summary."""

    (tmp_path / "notes.txt").write_text(
        "No PDFs",
        encoding="utf-8",
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    result = service.ingest_directory(
        tmp_path
    )

    assert result.total_files == 0
    assert result.documents_ingested == 0
    assert result.documents_failed == 0
    assert result.succeeded is True


def test_ingest_directory_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """A nonexistent ingestion folder should fail."""

    missing_directory = (
        tmp_path / "missing_documents"
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        DirectoryIngestionError,
        match="Directory does not exist",
    ):
        service.ingest_directory(
            missing_directory
        )


def test_ingest_directory_rejects_file_path(
    tmp_path: Path,
) -> None:
    """Directory ingestion should reject a regular file."""

    pdf_path = create_blank_pdf(
        tmp_path / "document.pdf"
    )

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    with pytest.raises(
        DirectoryIngestionError,
        match="not a directory",
    ):
        service.ingest_directory(
            pdf_path
        )


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("recursive", "yes"),
        ("continue_on_error", "yes"),
    ],
)
def test_ingest_directory_rejects_invalid_boolean_flags(
    tmp_path: Path,
    parameter_name: str,
    parameter_value: Any,
) -> None:
    """Directory control flags must be boolean."""

    service = PDFIngestionService(
        vector_store=create_vector_store(),
    )

    arguments: dict[str, Any] = {
        "directory_path": tmp_path,
        parameter_name: parameter_value,
    }

    with pytest.raises(
        TypeError,
        match=f"{parameter_name} must be a boolean",
    ):
        service.ingest_directory(**arguments)


def test_ingest_pdf_convenience_function_stores_in_supplied_store(
    tmp_path: Path,
) -> None:
    """The helper should use the caller's vector store."""

    pdf_path = create_blank_pdf(
        tmp_path / "blank.pdf"
    )

    vector_store = create_vector_store()

    result = ingest_pdf(
        file_path=pdf_path,
        vector_store=vector_store,
        metadata={
            "document_category": "test",
        },
        chunk_size=100,
        chunk_overlap=10,
    )

    assert result.source_filename == "blank.pdf"
    assert result.total_pages == 1
    assert result.chunks_created == 0
    assert result.vectors_stored == 0
    assert len(vector_store) == 0