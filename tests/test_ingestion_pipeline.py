"""
Tests for the high-level PDF ingestion pipeline.

These tests verify:

* PDF discovery
* Deterministic SHA-256 checksums
* Recursive and non-recursive scanning
* New-document ingestion
* Unchanged-document skipping
* Changed-document replacement
* Rollback after failed replacement
* Pipeline metadata propagation
* Duplicate filename detection
* Individual file failure handling
* Aggregate pipeline summaries
* Convenience-function behaviour

The tests intentionally do not verify:

* PDF text extraction
* Text chunking algorithms
* Embedding mathematics
* Similarity search
* PostgreSQL connectivity
* LLM behaviour
* Finance calculations
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import pytest

from src.rag.embeddings import DeterministicEmbeddingService
from src.rag.ingestion import (
    DocumentIngestionResult,
    PDFIngestionService,
)
from src.rag.ingestion_pipeline import (
    INGESTION_CHECKSUM_KEY,
    INGESTION_FILE_SIZE_KEY,
    INGESTION_PIPELINE_VERSION,
    INGESTION_PIPELINE_VERSION_KEY,
    INGESTION_RELATIVE_PATH_KEY,
    INGESTION_SOURCE_PATH_KEY,
    DiscoveredPDF,
    DocumentIngestionPipeline,
    DocumentReplacementError,
    DuplicateSourceFilenameError,
    IngestionPipelineResult,
    InvalidIngestionDirectoryError,
    PipelineFileResult,
    PipelineFileStatus,
    run_ingestion_pipeline,
)
from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
)


def create_pdf_like_file(
    file_path: Path,
    content: bytes = b"%PDF-1.4\nTest PDF content\n",
) -> Path:
    """
    Create a lightweight file with a PDF extension.

    Pipeline discovery and checksum tests do not parse PDF contents. Lower-level
    PDF parsing is already covered by test_document_loader.py.
    """

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    file_path.write_bytes(content)
    return file_path


def calculate_sha256(content: bytes) -> str:
    """Calculate the expected SHA-256 value for test content."""

    return hashlib.sha256(content).hexdigest()


def create_vector_store() -> InMemoryVectorStore:
    """Create a deterministic local vector store."""

    return InMemoryVectorStore(
        embedding_service=(
            DeterministicEmbeddingService(
                dimension=32,
            )
        )
    )


class RecordingIngestionService(PDFIngestionService):
    """
    Configurable ingestion-service test double.

    The class inherits from PDFIngestionService so it satisfies the pipeline's
    concrete runtime validation. It records calls and optionally stores
    configured documents in the supplied vector store.
    """

    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        *,
        pages_by_filename: Mapping[str, int] | None = None,
        chunks_by_filename: Mapping[str, tuple[str, ...]] | None = None,
        failures_by_filename: Mapping[str, Exception] | None = None,
    ) -> None:
        super().__init__(
            vector_store=vector_store,
        )

        self.pages_by_filename = dict(
            pages_by_filename or {}
        )
        self.chunks_by_filename = dict(
            chunks_by_filename or {}
        )
        self.failures_by_filename = dict(
            failures_by_filename or {}
        )

        self.calls: list[
            tuple[Path, dict[str, Any]]
        ] = []

    def ingest_pdf(
        self,
        file_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentIngestionResult:
        """Return a configured result and persist configured chunks."""

        path = Path(file_path).resolve()
        resolved_metadata = dict(metadata or {})

        self.calls.append(
            (
                path,
                resolved_metadata,
            )
        )

        configured_failure = (
            self.failures_by_filename.get(path.name)
        )

        if configured_failure is not None:
            raise configured_failure

        chunk_texts = self.chunks_by_filename.get(
            path.name,
            (
                f"Content from {path.name}",
            ),
        )

        documents: list[Document] = []

        for index, text in enumerate(
            chunk_texts,
        ):
            document_id = (
                f"{path.stem}_generated_chunk_{index}"
            )

            document_metadata = dict(
                resolved_metadata
            )
            document_metadata.update(
                {
                    "source_filename": path.name,
                    "page_number": 1,
                    "chunk_index": index,
                    "start_char": 0,
                    "end_char": len(text),
                    "chunk_id": document_id,
                }
            )

            documents.append(
                Document(
                    id=document_id,
                    text=text,
                    metadata=document_metadata,
                )
            )

        stored_documents = (
            self.vector_store.add_documents(
                documents
            )
            if documents
            else []
        )

        total_pages = self.pages_by_filename.get(
            path.name,
            1,
        )

        return DocumentIngestionResult(
            source_path=path,
            source_filename=path.name,
            total_pages=total_pages,
            empty_page_numbers=(),
            chunks_created=len(stored_documents),
            vectors_stored=len(stored_documents),
            document_ids=tuple(
                document.id
                for document in stored_documents
            ),
            metadata=resolved_metadata,
        )


def create_pipeline(
    tmp_path: Path,
    *,
    service: PDFIngestionService | None = None,
    recursive: bool = False,
    checksum_block_size: int = 1024,
) -> DocumentIngestionPipeline:
    """Create a pipeline with a deterministic local store."""

    if service is None:
        service = RecordingIngestionService(
            create_vector_store()
        )

    return DocumentIngestionPipeline(
        ingestion_service=service,
        document_directory=tmp_path,
        recursive=recursive,
        checksum_block_size=checksum_block_size,
    )


def create_existing_source_document(
    store: InMemoryVectorStore,
    *,
    document_id: str,
    relative_path: str,
    checksum: str,
    text: str = "Existing document content",
    page_number: int = 1,
    chunk_index: int = 0,
) -> Document:
    """Add one previously ingested chunk to the vector store."""

    document = Document(
        id=document_id,
        text=text,
        metadata={
            INGESTION_RELATIVE_PATH_KEY: relative_path,
            INGESTION_CHECKSUM_KEY: checksum,
            "source_filename": Path(
                relative_path
            ).name,
            "page_number": page_number,
            "chunk_index": chunk_index,
        },
    )

    store.add_documents([document])
    return document


# ---------------------------------------------------------------------------
# DiscoveredPDF tests
# ---------------------------------------------------------------------------


def test_discovered_pdf_creates_valid_instance(
    tmp_path: Path,
) -> None:
    """Discovery metadata should preserve valid file information."""

    checksum = "a" * 64

    discovered = DiscoveredPDF(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        relative_path="budget.pdf",
        source_filename="budget.pdf",
        checksum=checksum,
        file_size=100,
    )

    assert discovered.source_filename == "budget.pdf"
    assert discovered.relative_path == "budget.pdf"
    assert discovered.checksum == checksum
    assert discovered.file_size == 100


def test_discovered_pdf_normalizes_checksum_and_strings(
    tmp_path: Path,
) -> None:
    """Discovery values should be trimmed and checksum lowercased."""

    discovered = DiscoveredPDF(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        relative_path="  finance/budget.pdf  ",
        source_filename="  budget.pdf  ",
        checksum="A" * 64,
        file_size=10,
    )

    assert discovered.relative_path == (
        "finance/budget.pdf"
    )
    assert discovered.source_filename == "budget.pdf"
    assert discovered.checksum == "a" * 64


@pytest.mark.parametrize(
    "checksum",
    [
        "",
        "a" * 63,
        "a" * 65,
        "z" * 64,
    ],
)
def test_discovered_pdf_rejects_invalid_checksum(
    tmp_path: Path,
    checksum: str,
) -> None:
    """Checksums must be 64-character hexadecimal SHA-256 values."""

    with pytest.raises(ValueError):
        DiscoveredPDF(
            source_path=(
                tmp_path / "budget.pdf"
            ).resolve(),
            relative_path="budget.pdf",
            source_filename="budget.pdf",
            checksum=checksum,
            file_size=10,
        )


def test_discovered_pdf_rejects_negative_file_size(
    tmp_path: Path,
) -> None:
    """File size cannot be negative."""

    with pytest.raises(
        ValueError,
        match="file_size cannot be negative",
    ):
        DiscoveredPDF(
            source_path=(
                tmp_path / "budget.pdf"
            ).resolve(),
            relative_path="budget.pdf",
            source_filename="budget.pdf",
            checksum="a" * 64,
            file_size=-1,
        )


# ---------------------------------------------------------------------------
# PipelineFileResult tests
# ---------------------------------------------------------------------------


def test_pipeline_file_result_creates_success_result(
    tmp_path: Path,
) -> None:
    """A successful result should preserve page and chunk totals."""

    result = PipelineFileResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        relative_path="budget.pdf",
        source_filename="budget.pdf",
        checksum="a" * 64,
        status=PipelineFileStatus.INGESTED,
        pages_processed=2,
        chunks_stored=2,
        document_ids=(
            "chunk_1",
            "chunk_2",
        ),
    )

    assert result.status == (
        PipelineFileStatus.INGESTED
    )
    assert result.pages_processed == 2
    assert result.chunks_stored == 2


def test_pipeline_file_result_creates_failed_result(
    tmp_path: Path,
) -> None:
    """A failure should require error details."""

    result = PipelineFileResult(
        source_path=(
            tmp_path / "invalid.pdf"
        ).resolve(),
        relative_path="invalid.pdf",
        source_filename="invalid.pdf",
        checksum="b" * 64,
        status=PipelineFileStatus.FAILED,
        error_type="ValueError",
        error_message="Invalid PDF",
    )

    assert result.status == PipelineFileStatus.FAILED
    assert result.error_type == "ValueError"
    assert result.error_message == "Invalid PDF"


def test_pipeline_file_result_rejects_failure_without_error(
    tmp_path: Path,
) -> None:
    """Failed statuses must contain error information."""

    with pytest.raises(
        ValueError,
        match="must include error_type",
    ):
        PipelineFileResult(
            source_path=(
                tmp_path / "invalid.pdf"
            ).resolve(),
            relative_path="invalid.pdf",
            source_filename="invalid.pdf",
            checksum="b" * 64,
            status=PipelineFileStatus.FAILED,
        )


def test_pipeline_file_result_rejects_id_count_mismatch(
    tmp_path: Path,
) -> None:
    """Stored chunk count must match document identifiers."""

    with pytest.raises(
        ValueError,
        match="document_ids must match chunks_stored",
    ):
        PipelineFileResult(
            source_path=(
                tmp_path / "budget.pdf"
            ).resolve(),
            relative_path="budget.pdf",
            source_filename="budget.pdf",
            checksum="a" * 64,
            status=PipelineFileStatus.INGESTED,
            chunks_stored=2,
            document_ids=("chunk_1",),
        )


# ---------------------------------------------------------------------------
# IngestionPipelineResult tests
# ---------------------------------------------------------------------------


def test_pipeline_result_creates_valid_summary(
    tmp_path: Path,
) -> None:
    """Aggregate values should match individual file results."""

    ingested = PipelineFileResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        relative_path="budget.pdf",
        source_filename="budget.pdf",
        checksum="a" * 64,
        status=PipelineFileStatus.INGESTED,
        pages_processed=2,
        chunks_stored=1,
        document_ids=("chunk_1",),
    )

    skipped = PipelineFileResult(
        source_path=(
            tmp_path / "policy.pdf"
        ).resolve(),
        relative_path="policy.pdf",
        source_filename="policy.pdf",
        checksum="b" * 64,
        status=PipelineFileStatus.SKIPPED,
        chunks_stored=1,
        document_ids=("chunk_2",),
    )

    result = IngestionPipelineResult(
        directory_path=tmp_path.resolve(),
        files=(
            ingested,
            skipped,
        ),
        files_discovered=2,
        files_ingested=1,
        files_updated=0,
        files_skipped=1,
        files_failed=0,
        total_pages=2,
        total_chunks_stored=1,
    )

    assert result.succeeded is True
    assert result.partially_succeeded is False


def test_pipeline_result_reports_partial_success(
    tmp_path: Path,
) -> None:
    """A mixture of success and failure should report partial success."""

    ingested = PipelineFileResult(
        source_path=(
            tmp_path / "budget.pdf"
        ).resolve(),
        relative_path="budget.pdf",
        source_filename="budget.pdf",
        checksum="a" * 64,
        status=PipelineFileStatus.INGESTED,
        pages_processed=1,
        chunks_stored=1,
        document_ids=("chunk_1",),
    )

    failed = PipelineFileResult(
        source_path=(
            tmp_path / "broken.pdf"
        ).resolve(),
        relative_path="broken.pdf",
        source_filename="broken.pdf",
        checksum="b" * 64,
        status=PipelineFileStatus.FAILED,
        error_type="RuntimeError",
        error_message="Failed",
    )

    result = IngestionPipelineResult(
        directory_path=tmp_path.resolve(),
        files=(
            ingested,
            failed,
        ),
        files_discovered=2,
        files_ingested=1,
        files_updated=0,
        files_skipped=0,
        files_failed=1,
        total_pages=1,
        total_chunks_stored=1,
    )

    assert result.succeeded is False
    assert result.partially_succeeded is True


# ---------------------------------------------------------------------------
# Constructor and configuration tests
# ---------------------------------------------------------------------------


def test_pipeline_rejects_invalid_ingestion_service(
    tmp_path: Path,
) -> None:
    """Pipeline construction requires PDFIngestionService."""

    with pytest.raises(
        TypeError,
        match="ingestion_service must be",
    ):
        DocumentIngestionPipeline(
            ingestion_service="invalid",  # type: ignore[arg-type]
            document_directory=tmp_path,
        )


@pytest.mark.parametrize(
    "recursive",
    [
        "yes",
        1,
        None,
    ],
)
def test_pipeline_rejects_invalid_recursive_setting(
    tmp_path: Path,
    recursive: Any,
) -> None:
    """Constructor recursive setting must be boolean."""

    with pytest.raises(
        TypeError,
        match="recursive must be a boolean",
    ):
        DocumentIngestionPipeline(
            ingestion_service=(
                RecordingIngestionService(
                    create_vector_store()
                )
            ),
            document_directory=tmp_path,
            recursive=recursive,
        )


@pytest.mark.parametrize(
    "block_size",
    [
        True,
        1.5,
        "1024",
        None,
    ],
)
def test_pipeline_rejects_invalid_checksum_block_type(
    tmp_path: Path,
    block_size: Any,
) -> None:
    """Checksum block size must be an integer."""

    with pytest.raises(
        TypeError,
        match="checksum_block_size must be an integer",
    ):
        create_pipeline(
            tmp_path,
            checksum_block_size=block_size,
        )


@pytest.mark.parametrize(
    "block_size",
    [
        0,
        -1,
    ],
)
def test_pipeline_rejects_non_positive_checksum_block_size(
    tmp_path: Path,
    block_size: int,
) -> None:
    """Checksum block size must be positive."""

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        create_pipeline(
            tmp_path,
            checksum_block_size=block_size,
        )


def test_pipeline_exposes_configuration(
    tmp_path: Path,
) -> None:
    """Pipeline properties should expose configured dependencies."""

    service = RecordingIngestionService(
        create_vector_store()
    )

    pipeline = create_pipeline(
        tmp_path,
        service=service,
        recursive=True,
        checksum_block_size=2048,
    )

    assert pipeline.ingestion_service is service
    assert pipeline.vector_store is (
        service.vector_store
    )
    assert pipeline.document_directory == tmp_path
    assert pipeline.recursive is True


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


def test_discover_pdfs_returns_pdf_files_only(
    tmp_path: Path,
) -> None:
    """Non-PDF files should be ignored."""

    create_pdf_like_file(
        tmp_path / "budget.pdf"
    )
    create_pdf_like_file(
        tmp_path / "forecast.PDF"
    )
    (tmp_path / "notes.txt").write_text(
        "Ignore",
        encoding="utf-8",
    )

    discovered = create_pipeline(
        tmp_path
    ).discover_pdfs()

    assert tuple(
        item.source_filename
        for item in discovered
    ) == (
        "budget.pdf",
        "forecast.PDF",
    )


def test_discover_pdfs_is_sorted_by_relative_path(
    tmp_path: Path,
) -> None:
    """Discovery order should be deterministic."""

    create_pdf_like_file(
        tmp_path / "zeta.pdf"
    )
    create_pdf_like_file(
        tmp_path / "Alpha.pdf"
    )
    create_pdf_like_file(
        tmp_path / "middle.pdf"
    )

    discovered = create_pipeline(
        tmp_path
    ).discover_pdfs()

    assert tuple(
        item.source_filename
        for item in discovered
    ) == (
        "Alpha.pdf",
        "middle.pdf",
        "zeta.pdf",
    )


def test_discover_pdfs_calculates_checksum_and_size(
    tmp_path: Path,
) -> None:
    """Discovery should calculate exact file metadata."""

    content = b"%PDF-test-finance-policy"
    pdf_path = create_pdf_like_file(
        tmp_path / "policy.pdf",
        content,
    )

    discovered = create_pipeline(
        tmp_path
    ).discover_pdfs()

    assert len(discovered) == 1

    result = discovered[0]

    assert result.source_path == pdf_path.resolve()
    assert result.relative_path == "policy.pdf"
    assert result.file_size == len(content)
    assert result.checksum == calculate_sha256(
        content
    )


def test_discover_pdfs_is_non_recursive_by_default(
    tmp_path: Path,
) -> None:
    """Nested PDFs should be ignored by default."""

    create_pdf_like_file(
        tmp_path / "root.pdf"
    )
    create_pdf_like_file(
        tmp_path / "nested" / "child.pdf"
    )

    discovered = create_pipeline(
        tmp_path
    ).discover_pdfs()

    assert tuple(
        item.relative_path
        for item in discovered
    ) == ("root.pdf",)


def test_discover_pdfs_supports_recursive_scanning(
    tmp_path: Path,
) -> None:
    """Recursive discovery should include nested PDFs."""

    create_pdf_like_file(
        tmp_path / "root.pdf"
    )
    create_pdf_like_file(
        tmp_path / "nested" / "child.pdf"
    )

    discovered = create_pipeline(
        tmp_path,
        recursive=True,
    ).discover_pdfs()

    assert tuple(
        item.relative_path
        for item in discovered
    ) == (
        "nested/child.pdf",
        "root.pdf",
    )


def test_discover_pdfs_supports_recursive_override(
    tmp_path: Path,
) -> None:
    """Method-level recursive setting should override constructor default."""

    create_pdf_like_file(
        tmp_path / "root.pdf"
    )
    create_pdf_like_file(
        tmp_path / "nested" / "child.pdf"
    )

    pipeline = create_pipeline(
        tmp_path,
        recursive=False,
    )

    discovered = pipeline.discover_pdfs(
        recursive=True
    )

    assert len(discovered) == 2


def test_discover_pdfs_returns_empty_tuple_for_empty_directory(
    tmp_path: Path,
) -> None:
    """A valid directory without PDFs should return no discoveries."""

    discovered = create_pipeline(
        tmp_path
    ).discover_pdfs()

    assert discovered == ()


def test_pipeline_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Running against a missing directory should fail clearly."""

    missing_directory = (
        tmp_path / "missing"
    )

    pipeline = create_pipeline(
        missing_directory
    )

    with pytest.raises(
        InvalidIngestionDirectoryError,
        match="does not exist",
    ):
        pipeline.run()


def test_pipeline_rejects_regular_file_as_directory(
    tmp_path: Path,
) -> None:
    """A regular file cannot be used as the document directory."""

    file_path = tmp_path / "not_a_directory"
    file_path.write_text(
        "test",
        encoding="utf-8",
    )

    pipeline = DocumentIngestionPipeline(
        ingestion_service=(
            RecordingIngestionService(
                create_vector_store()
            )
        ),
        document_directory=file_path,
    )

    with pytest.raises(
        InvalidIngestionDirectoryError,
        match="not a directory",
    ):
        pipeline.run()


# ---------------------------------------------------------------------------
# Ingestion and metadata tests
# ---------------------------------------------------------------------------


def test_pipeline_ingests_new_pdf(
    tmp_path: Path,
) -> None:
    """A previously unknown PDF should be ingested."""

    create_pdf_like_file(
        tmp_path / "budget.pdf"
    )

    store = create_vector_store()
    service = RecordingIngestionService(
        store,
        pages_by_filename={
            "budget.pdf": 3,
        },
        chunks_by_filename={
            "budget.pdf": (
                "Budget chunk one",
                "Budget chunk two",
            ),
        },
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run()

    assert result.files_discovered == 1
    assert result.files_ingested == 1
    assert result.files_updated == 0
    assert result.files_skipped == 0
    assert result.files_failed == 0
    assert result.total_pages == 3
    assert result.total_chunks_stored == 2
    assert len(store) == 2

    file_result = result.files[0]

    assert file_result.status == (
        PipelineFileStatus.INGESTED
    )
    assert file_result.pages_processed == 3
    assert file_result.chunks_stored == 2


def test_pipeline_adds_system_metadata(
    tmp_path: Path,
) -> None:
    """Pipeline metadata should include checksum and source information."""

    pdf_path = create_pdf_like_file(
        tmp_path / "budget.pdf",
        b"budget file content",
    )

    store = create_vector_store()
    service = RecordingIngestionService(
        store
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run(
        metadata={
            "document_category": "budget",
            "financial_year": "2026",
        }
    )

    stored_document = store.get_document(
        result.files[0].document_ids[0]
    )

    assert stored_document.metadata[
        "document_category"
    ] == "budget"
    assert stored_document.metadata[
        "financial_year"
    ] == "2026"
    assert stored_document.metadata[
        INGESTION_SOURCE_PATH_KEY
    ] == str(pdf_path.resolve())
    assert stored_document.metadata[
        INGESTION_RELATIVE_PATH_KEY
    ] == "budget.pdf"
    assert stored_document.metadata[
        INGESTION_CHECKSUM_KEY
    ] == calculate_sha256(
        b"budget file content"
    )
    assert stored_document.metadata[
        INGESTION_FILE_SIZE_KEY
    ] == len(b"budget file content")
    assert stored_document.metadata[
        INGESTION_PIPELINE_VERSION_KEY
    ] == INGESTION_PIPELINE_VERSION


def test_system_metadata_overrides_conflicting_user_values(
    tmp_path: Path,
) -> None:
    """Required ingestion metadata should remain internally consistent."""

    create_pdf_like_file(
        tmp_path / "budget.pdf",
        b"correct content",
    )

    store = create_vector_store()
    service = RecordingIngestionService(
        store
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run(
        metadata={
            INGESTION_RELATIVE_PATH_KEY: "wrong.pdf",
            INGESTION_CHECKSUM_KEY: "wrong-checksum",
            "document_category": "budget",
        }
    )

    stored_document = store.get_document(
        result.files[0].document_ids[0]
    )

    assert stored_document.metadata[
        INGESTION_RELATIVE_PATH_KEY
    ] == "budget.pdf"

    assert stored_document.metadata[
        INGESTION_CHECKSUM_KEY
    ] == calculate_sha256(
        b"correct content"
    )


def test_pipeline_rejects_invalid_metadata(
    tmp_path: Path,
) -> None:
    """Optional pipeline metadata must be a mapping."""

    with pytest.raises(
        TypeError,
        match="metadata must be a mapping",
    ):
        create_pipeline(
            tmp_path
        ).run(
            metadata="invalid"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Skip behaviour
# ---------------------------------------------------------------------------


def test_pipeline_skips_unchanged_pdf(
    tmp_path: Path,
) -> None:
    """A matching stored checksum should prevent re-ingestion."""

    content = b"unchanged budget"
    create_pdf_like_file(
        tmp_path / "budget.pdf",
        content,
    )

    checksum = calculate_sha256(content)
    store = create_vector_store()

    existing = create_existing_source_document(
        store,
        document_id="existing_chunk",
        relative_path="budget.pdf",
        checksum=checksum,
    )

    service = RecordingIngestionService(
        store
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run()

    assert result.files_ingested == 0
    assert result.files_updated == 0
    assert result.files_skipped == 1
    assert result.files_failed == 0
    assert service.calls == []
    assert len(store) == 1

    file_result = result.files[0]

    assert file_result.status == (
        PipelineFileStatus.SKIPPED
    )
    assert file_result.pages_processed == 0
    assert file_result.chunks_stored == 1
    assert file_result.document_ids == (
        existing.id,
    )


def test_pipeline_skips_multiple_existing_chunks(
    tmp_path: Path,
) -> None:
    """All chunks belonging to an unchanged PDF should be reported."""

    content = b"unchanged forecast"
    create_pdf_like_file(
        tmp_path / "forecast.pdf",
        content,
    )

    checksum = calculate_sha256(content)
    store = create_vector_store()

    create_existing_source_document(
        store,
        document_id="forecast_1",
        relative_path="forecast.pdf",
        checksum=checksum,
        page_number=1,
        chunk_index=0,
    )

    create_existing_source_document(
        store,
        document_id="forecast_2",
        relative_path="forecast.pdf",
        checksum=checksum,
        page_number=1,
        chunk_index=1,
    )

    service = RecordingIngestionService(
        store
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run()

    assert result.files[0].document_ids == (
        "forecast_1",
        "forecast_2",
    )
    assert result.files[0].chunks_stored == 2


# ---------------------------------------------------------------------------
# Update and rollback behaviour
# ---------------------------------------------------------------------------


def test_pipeline_updates_changed_pdf(
    tmp_path: Path,
) -> None:
    """A different checksum should replace prior document chunks."""

    create_pdf_like_file(
        tmp_path / "budget.pdf",
        b"new budget content",
    )

    store = create_vector_store()

    create_existing_source_document(
        store,
        document_id="old_budget_chunk",
        relative_path="budget.pdf",
        checksum=calculate_sha256(
            b"old budget content"
        ),
        text="Old budget",
    )

    service = RecordingIngestionService(
        store,
        chunks_by_filename={
            "budget.pdf": (
                "New budget one",
                "New budget two",
            ),
        },
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run()

    assert result.files_ingested == 0
    assert result.files_updated == 1
    assert result.files_skipped == 0
    assert result.files_failed == 0

    assert "old_budget_chunk" not in store
    assert len(store) == 2

    file_result = result.files[0]

    assert file_result.status == (
        PipelineFileStatus.UPDATED
    )
    assert file_result.chunks_stored == 2


def test_changed_document_failure_restores_old_vectors(
    tmp_path: Path,
) -> None:
    """Failed replacement should restore the previous stored records."""

    create_pdf_like_file(
        tmp_path / "budget.pdf",
        b"new content",
    )

    store = create_vector_store()

    old_document = create_existing_source_document(
        store,
        document_id="old_budget_chunk",
        relative_path="budget.pdf",
        checksum=calculate_sha256(
            b"old content"
        ),
        text="Old content",
    )

    service = RecordingIngestionService(
        store,
        failures_by_filename={
            "budget.pdf": RuntimeError(
                "New ingestion failed"
            ),
        },
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run(
        continue_on_error=True
    )

    assert result.files_failed == 1
    assert result.files[0].status == (
        PipelineFileStatus.FAILED
    )
    assert result.files[0].error_type == (
        "DocumentReplacementError"
    )

    restored_document = store.get_document(
        "old_budget_chunk"
    )

    assert restored_document == old_document
    assert len(store) == 1


def test_changed_document_failure_raises_when_continuation_disabled(
    tmp_path: Path,
) -> None:
    """Replacement errors should propagate when continuation is disabled."""

    create_pdf_like_file(
        tmp_path / "budget.pdf",
        b"new content",
    )

    store = create_vector_store()

    create_existing_source_document(
        store,
        document_id="old_budget_chunk",
        relative_path="budget.pdf",
        checksum=calculate_sha256(
            b"old content"
        ),
    )

    service = RecordingIngestionService(
        store,
        failures_by_filename={
            "budget.pdf": RuntimeError(
                "Ingestion failed"
            ),
        },
    )

    with pytest.raises(
        DocumentReplacementError,
        match="Previous vectors were restored",
    ):
        create_pipeline(
            tmp_path,
            service=service,
        ).run(
            continue_on_error=False
        )

    assert "old_budget_chunk" in store


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_pipeline_records_individual_file_failure(
    tmp_path: Path,
) -> None:
    """A failed new PDF should be captured in the pipeline result."""

    create_pdf_like_file(
        tmp_path / "broken.pdf"
    )
    create_pdf_like_file(
        tmp_path / "valid.pdf"
    )

    store = create_vector_store()

    service = RecordingIngestionService(
        store,
        failures_by_filename={
            "broken.pdf": RuntimeError(
                "Cannot ingest broken document"
            ),
        },
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run(
        continue_on_error=True
    )

    assert result.files_discovered == 2
    assert result.files_ingested == 1
    assert result.files_failed == 1
    assert result.partially_succeeded is True

    failed = next(
        item
        for item in result.files
        if item.status
        == PipelineFileStatus.FAILED
    )

    assert failed.source_filename == "broken.pdf"
    assert failed.error_type == "RuntimeError"
    assert "Cannot ingest" in (
        failed.error_message or ""
    )


def test_pipeline_stops_on_first_failure_when_configured(
    tmp_path: Path,
) -> None:
    """continue_on_error=False should propagate the original exception."""

    create_pdf_like_file(
        tmp_path / "broken.pdf"
    )

    service = RecordingIngestionService(
        create_vector_store(),
        failures_by_filename={
            "broken.pdf": ValueError(
                "Invalid document"
            ),
        },
    )

    with pytest.raises(
        ValueError,
        match="Invalid document",
    ):
        create_pipeline(
            tmp_path,
            service=service,
        ).run(
            continue_on_error=False
        )


def test_pipeline_rejects_invalid_continue_flag(
    tmp_path: Path,
) -> None:
    """continue_on_error must be boolean."""

    with pytest.raises(
        TypeError,
        match="continue_on_error must be a boolean",
    ):
        create_pipeline(
            tmp_path
        ).run(
            continue_on_error="yes",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Duplicate filename protection
# ---------------------------------------------------------------------------


def test_pipeline_rejects_duplicate_filenames_in_nested_folders(
    tmp_path: Path,
) -> None:
    """Identical filenames in separate folders should be rejected."""

    create_pdf_like_file(
        tmp_path / "business_a" / "policy.pdf"
    )
    create_pdf_like_file(
        tmp_path / "business_b" / "policy.pdf"
    )

    pipeline = create_pipeline(
        tmp_path,
        recursive=True,
    )

    with pytest.raises(
        DuplicateSourceFilenameError,
        match="Duplicate PDF filenames",
    ):
        pipeline.run()


def test_duplicate_filename_check_is_case_insensitive(
    tmp_path: Path,
) -> None:
    """Filename collision detection should ignore case."""

    create_pdf_like_file(
        tmp_path / "one" / "Policy.pdf"
    )
    create_pdf_like_file(
        tmp_path / "two" / "policy.PDF"
    )

    with pytest.raises(
        DuplicateSourceFilenameError,
    ):
        create_pipeline(
            tmp_path,
            recursive=True,
        ).run()


# ---------------------------------------------------------------------------
# Multiple-file and empty-directory behaviour
# ---------------------------------------------------------------------------


def test_pipeline_processes_multiple_new_pdfs(
    tmp_path: Path,
) -> None:
    """Every discovered new PDF should be ingested."""

    create_pdf_like_file(
        tmp_path / "budget.pdf"
    )
    create_pdf_like_file(
        tmp_path / "forecast.pdf"
    )
    create_pdf_like_file(
        tmp_path / "policy.pdf"
    )

    store = create_vector_store()
    service = RecordingIngestionService(
        store
    )

    result = create_pipeline(
        tmp_path,
        service=service,
    ).run()

    assert result.files_discovered == 3
    assert result.files_ingested == 3
    assert result.files_failed == 0
    assert result.total_chunks_stored == 3
    assert len(store) == 3


def test_pipeline_returns_zero_summary_for_empty_directory(
    tmp_path: Path,
) -> None:
    """An empty directory should produce a successful zero result."""

    result = create_pipeline(
        tmp_path
    ).run()

    assert result.files == ()
    assert result.files_discovered == 0
    assert result.files_ingested == 0
    assert result.files_updated == 0
    assert result.files_skipped == 0
    assert result.files_failed == 0
    assert result.total_pages == 0
    assert result.total_chunks_stored == 0
    assert result.succeeded is True


def test_pipeline_uses_directory_override(
    tmp_path: Path,
) -> None:
    """run() should allow a one-time alternative directory."""

    default_directory = tmp_path / "default"
    override_directory = tmp_path / "override"

    default_directory.mkdir()
    override_directory.mkdir()

    create_pdf_like_file(
        override_directory / "budget.pdf"
    )

    service = RecordingIngestionService(
        create_vector_store()
    )

    pipeline = DocumentIngestionPipeline(
        ingestion_service=service,
        document_directory=default_directory,
    )

    result = pipeline.run(
        directory_path=override_directory
    )

    assert result.directory_path == (
        override_directory.resolve()
    )
    assert result.files_discovered == 1


# ---------------------------------------------------------------------------
# Convenience-function test
# ---------------------------------------------------------------------------


def test_run_ingestion_pipeline_matches_direct_pipeline(
    tmp_path: Path,
) -> None:
    """The convenience function should perform a normal pipeline run."""

    create_pdf_like_file(
        tmp_path / "budget.pdf"
    )

    service = RecordingIngestionService(
        create_vector_store()
    )

    result = run_ingestion_pipeline(
        ingestion_service=service,
        directory_path=tmp_path,
        metadata={
            "document_category": "budget",
        },
        recursive=False,
        continue_on_error=True,
    )

    assert isinstance(
        result,
        IngestionPipelineResult,
    )
    assert result.files_discovered == 1
    assert result.files_ingested == 1
    assert result.files_failed == 0