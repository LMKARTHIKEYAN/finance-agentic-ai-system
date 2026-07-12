"""
High-level PDF ingestion pipeline for the Finance Agentic AI System.

This module coordinates directory-based document ingestion while reusing the
completed lower-level RAG components.

Responsibilities:

* Scan a local directory for PDF files
* Calculate deterministic SHA-256 file checksums
* Skip PDFs that have already been ingested without changes
* Re-ingest PDFs whose content has changed
* Preserve document-level ingestion metadata
* Support recursive directory scanning
* Produce typed pipeline summaries
* Continue processing after individual file failures when configured

The pipeline delegates:

* PDF extraction to PDFDocumentLoader
* Text splitting to TextChunker
* Embedding and storage to PDFIngestionService and the configured vector store

This module contains no:

* PDF extraction implementation
* Text-chunking implementation
* Embedding algorithm
* Vector similarity logic
* Finance calculation logic
* LLM calls
* FastAPI or Streamlit logic
* S3 integration
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

from src.rag.ingestion import (
    DocumentIngestionResult,
    PDFIngestionService,
)
from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
)


PathLike: TypeAlias = str | Path

INGESTION_SOURCE_PATH_KEY = "ingestion_source_path"
INGESTION_RELATIVE_PATH_KEY = "ingestion_relative_path"
INGESTION_CHECKSUM_KEY = "ingestion_checksum"
INGESTION_FILE_SIZE_KEY = "ingestion_file_size"
INGESTION_PIPELINE_VERSION_KEY = "ingestion_pipeline_version"

INGESTION_PIPELINE_VERSION = "1.0"


class IngestionPipelineError(RuntimeError):
    """Base exception raised by the high-level ingestion pipeline."""


class InvalidIngestionDirectoryError(IngestionPipelineError):
    """Raised when the configured document directory is invalid."""


class DuplicateSourceFilenameError(IngestionPipelineError):
    """
    Raised when multiple discovered PDFs have the same filename.

    Document chunk identifiers currently include the source filename but not
    the full source path. Duplicate filenames could therefore produce
    conflicting chunk identifiers.
    """


class DocumentReplacementError(IngestionPipelineError):
    """Raised when a changed document cannot be safely replaced."""


class PipelineFileStatus(str, Enum):
    """Possible processing outcomes for one PDF."""

    INGESTED = "ingested"
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class DiscoveredPDF:
    """
    Metadata collected before ingesting one PDF.

    Attributes:
        source_path:
            Resolved absolute path of the PDF.

        relative_path:
            Path relative to the scanned document directory.

        source_filename:
            PDF filename.

        checksum:
            SHA-256 checksum of the complete file contents.

        file_size:
            File size in bytes.
    """

    source_path: Path
    relative_path: str
    source_filename: str
    checksum: str
    file_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path.")

        if not isinstance(self.relative_path, str):
            raise TypeError("relative_path must be a string.")

        cleaned_relative_path = self.relative_path.strip()

        if not cleaned_relative_path:
            raise ValueError("relative_path cannot be empty.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = self.source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        if not isinstance(self.checksum, str):
            raise TypeError("checksum must be a string.")

        cleaned_checksum = self.checksum.strip().lower()

        if len(cleaned_checksum) != 64:
            raise ValueError(
                "checksum must be a 64-character SHA-256 value."
            )

        if any(
            character not in "0123456789abcdef"
            for character in cleaned_checksum
        ):
            raise ValueError(
                "checksum must contain only hexadecimal characters."
            )

        if isinstance(self.file_size, bool) or not isinstance(
            self.file_size,
            int,
        ):
            raise TypeError("file_size must be an integer.")

        if self.file_size < 0:
            raise ValueError("file_size cannot be negative.")

        object.__setattr__(
            self,
            "relative_path",
            cleaned_relative_path,
        )
        object.__setattr__(
            self,
            "source_filename",
            cleaned_filename,
        )
        object.__setattr__(
            self,
            "checksum",
            cleaned_checksum,
        )


@dataclass(frozen=True)
class PipelineFileResult:
    """
    Processing result for one discovered PDF.

    Attributes:
        source_path:
            Resolved source PDF path.

        relative_path:
            Source path relative to the scanned directory.

        source_filename:
            Source PDF filename.

        checksum:
            SHA-256 checksum calculated before processing.

        status:
            Ingested, updated, skipped, or failed.

        pages_processed:
            Number of source PDF pages successfully processed.

        chunks_stored:
            Number of vectors stored for the current document version.

        document_ids:
            Vector-store IDs associated with the current ingestion.

        error_type:
            Exception class name for failures.

        error_message:
            Human-readable failure details.
    """

    source_path: Path
    relative_path: str
    source_filename: str
    checksum: str
    status: PipelineFileStatus
    pages_processed: int = 0
    chunks_stored: int = 0
    document_ids: tuple[str, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path.")

        if not isinstance(self.relative_path, str):
            raise TypeError("relative_path must be a string.")

        if not self.relative_path.strip():
            raise ValueError("relative_path cannot be empty.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        if not self.source_filename.strip():
            raise ValueError("source_filename cannot be empty.")

        if not isinstance(self.checksum, str):
            raise TypeError("checksum must be a string.")

        if not isinstance(self.status, PipelineFileStatus):
            raise TypeError(
                "status must be a PipelineFileStatus."
            )

        numeric_fields = {
            "pages_processed": self.pages_processed,
            "chunks_stored": self.chunks_stored,
        }

        for field_name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if not isinstance(self.document_ids, tuple):
            raise TypeError("document_ids must be a tuple.")

        if any(
            not isinstance(document_id, str)
            or not document_id.strip()
            for document_id in self.document_ids
        ):
            raise ValueError(
                "document_ids must contain non-empty strings."
            )

        if len(self.document_ids) != self.chunks_stored:
            raise ValueError(
                "document_ids must match chunks_stored."
            )

        if self.status == PipelineFileStatus.FAILED:
            if not self.error_type:
                raise ValueError(
                    "Failed results must include error_type."
                )

            if not self.error_message:
                raise ValueError(
                    "Failed results must include error_message."
                )
        else:
            if self.error_type is not None:
                raise ValueError(
                    "Successful or skipped results cannot include "
                    "error_type."
                )

            if self.error_message is not None:
                raise ValueError(
                    "Successful or skipped results cannot include "
                    "error_message."
                )


@dataclass(frozen=True)
class IngestionPipelineResult:
    """
    Aggregate result of one directory-ingestion run.

    Attributes:
        directory_path:
            Resolved directory that was scanned.

        files:
            Individual file-processing results.

        files_discovered:
            Number of PDF files found.

        files_ingested:
            Number of previously unknown PDFs ingested.

        files_updated:
            Number of changed PDFs replaced.

        files_skipped:
            Number of unchanged PDFs skipped.

        files_failed:
            Number of failed PDFs.

        total_pages:
            Pages processed for newly ingested or updated files.

        total_chunks_stored:
            Vectors stored for newly ingested or updated files.
    """

    directory_path: Path
    files: tuple[PipelineFileResult, ...]
    files_discovered: int
    files_ingested: int
    files_updated: int
    files_skipped: int
    files_failed: int
    total_pages: int
    total_chunks_stored: int

    def __post_init__(self) -> None:
        if not isinstance(self.directory_path, Path):
            raise TypeError(
                "directory_path must be a pathlib.Path."
            )

        if not isinstance(self.files, tuple):
            raise TypeError("files must be a tuple.")

        if not all(
            isinstance(result, PipelineFileResult)
            for result in self.files
        ):
            raise TypeError(
                "files must contain PipelineFileResult instances."
            )

        numeric_fields = {
            "files_discovered": self.files_discovered,
            "files_ingested": self.files_ingested,
            "files_updated": self.files_updated,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "total_pages": self.total_pages,
            "total_chunks_stored": self.total_chunks_stored,
        }

        for field_name, value in numeric_fields.items():
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{field_name} must be an integer."
                )

            if value < 0:
                raise ValueError(
                    f"{field_name} cannot be negative."
                )

        if self.files_discovered != len(self.files):
            raise ValueError(
                "files_discovered must match files."
            )

        expected_ingested = sum(
            result.status == PipelineFileStatus.INGESTED
            for result in self.files
        )
        expected_updated = sum(
            result.status == PipelineFileStatus.UPDATED
            for result in self.files
        )
        expected_skipped = sum(
            result.status == PipelineFileStatus.SKIPPED
            for result in self.files
        )
        expected_failed = sum(
            result.status == PipelineFileStatus.FAILED
            for result in self.files
        )

        if self.files_ingested != expected_ingested:
            raise ValueError(
                "files_ingested must match file results."
            )

        if self.files_updated != expected_updated:
            raise ValueError(
                "files_updated must match file results."
            )

        if self.files_skipped != expected_skipped:
            raise ValueError(
                "files_skipped must match file results."
            )

        if self.files_failed != expected_failed:
            raise ValueError(
                "files_failed must match file results."
            )

        expected_pages = sum(
            result.pages_processed
            for result in self.files
            if result.status
            in {
                PipelineFileStatus.INGESTED,
                PipelineFileStatus.UPDATED,
            }
        )

        if self.total_pages != expected_pages:
            raise ValueError(
                "total_pages must match processed file results."
            )

        expected_chunks = sum(
            result.chunks_stored
            for result in self.files
            if result.status
            in {
                PipelineFileStatus.INGESTED,
                PipelineFileStatus.UPDATED,
            }
        )

        if self.total_chunks_stored != expected_chunks:
            raise ValueError(
                "total_chunks_stored must match file results."
            )

    @property
    def succeeded(self) -> bool:
        """Return whether the run completed without file failures."""

        return self.files_failed == 0

    @property
    def partially_succeeded(self) -> bool:
        """Return whether some files succeeded and some failed."""

        successful_count = (
            self.files_ingested
            + self.files_updated
            + self.files_skipped
        )

        return successful_count > 0 and self.files_failed > 0


class DocumentIngestionPipeline:
    """
    Scan and incrementally ingest local PDF documents.

    The pipeline uses file checksums stored in chunk metadata to identify
    unchanged documents. Changed documents are replaced using the existing
    vector-store operations.

    Args:
        ingestion_service:
            Completed PDFIngestionService instance.

        document_directory:
            Default directory to scan. The project-development default is
            ``data/rag_documents``.

        recursive:
            Whether nested folders should be scanned by default.

        checksum_block_size:
            Number of bytes read per checksum operation.
    """

    def __init__(
        self,
        ingestion_service: PDFIngestionService,
        document_directory: PathLike = "data/rag_documents",
        recursive: bool = False,
        checksum_block_size: int = 1024 * 1024,
    ) -> None:
        if not isinstance(
            ingestion_service,
            PDFIngestionService,
        ):
            raise TypeError(
                "ingestion_service must be a PDFIngestionService."
            )

        if not isinstance(recursive, bool):
            raise TypeError("recursive must be a boolean.")

        if (
            isinstance(checksum_block_size, bool)
            or not isinstance(checksum_block_size, int)
        ):
            raise TypeError(
                "checksum_block_size must be an integer."
            )

        if checksum_block_size <= 0:
            raise ValueError(
                "checksum_block_size must be greater than zero."
            )

        self._ingestion_service = ingestion_service
        self._document_directory = self._validate_directory_value(
            document_directory
        )
        self._recursive = recursive
        self._checksum_block_size = checksum_block_size

    @property
    def ingestion_service(self) -> PDFIngestionService:
        """Return the configured lower-level ingestion service."""

        return self._ingestion_service

    @property
    def vector_store(self) -> InMemoryVectorStore:
        """Return the vector store used by the ingestion service."""

        return self._ingestion_service.vector_store

    @property
    def document_directory(self) -> Path:
        """Return the configured document directory path."""

        return self._document_directory

    @property
    def recursive(self) -> bool:
        """Return the default recursive scanning setting."""

        return self._recursive

    def run(
        self,
        directory_path: PathLike | None = None,
        metadata: Mapping[str, Any] | None = None,
        recursive: bool | None = None,
        continue_on_error: bool = True,
    ) -> IngestionPipelineResult:
        """
        Scan a directory and incrementally ingest its PDF documents.

        Args:
            directory_path:
                Optional directory override.

            metadata:
                Optional metadata added to every ingested chunk.

            recursive:
                Optional scan-setting override.

            continue_on_error:
                Whether remaining PDFs should continue after a failure.

        Returns:
            Typed pipeline summary.
        """

        if not isinstance(continue_on_error, bool):
            raise TypeError(
                "continue_on_error must be a boolean."
            )

        if recursive is not None and not isinstance(
            recursive,
            bool,
        ):
            raise TypeError("recursive must be a boolean.")

        validated_metadata = self._validate_metadata(
            metadata
        )

        directory = self._resolve_directory(
            directory_path
        )

        use_recursive = (
            self._recursive
            if recursive is None
            else recursive
        )

        discovered_files = self.discover_pdfs(
            directory_path=directory,
            recursive=use_recursive,
        )

        self._validate_unique_filenames(
            discovered_files
        )

        file_results: list[PipelineFileResult] = []

        for discovered_pdf in discovered_files:
            try:
                file_result = self._process_file(
                    discovered_pdf=discovered_pdf,
                    metadata=validated_metadata,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise

                file_result = PipelineFileResult(
                    source_path=discovered_pdf.source_path,
                    relative_path=discovered_pdf.relative_path,
                    source_filename=(
                        discovered_pdf.source_filename
                    ),
                    checksum=discovered_pdf.checksum,
                    status=PipelineFileStatus.FAILED,
                    error_type=type(exc).__name__,
                    error_message=(
                        str(exc) or type(exc).__name__
                    ),
                )

            file_results.append(file_result)

        return self._build_pipeline_result(
            directory_path=directory,
            file_results=file_results,
        )

    def discover_pdfs(
        self,
        directory_path: PathLike | None = None,
        recursive: bool | None = None,
    ) -> tuple[DiscoveredPDF, ...]:
        """
        Discover PDF files and calculate their checksums.

        Files are returned in deterministic relative-path order.
        """

        if recursive is not None and not isinstance(
            recursive,
            bool,
        ):
            raise TypeError("recursive must be a boolean.")

        directory = self._resolve_directory(
            directory_path
        )

        use_recursive = (
            self._recursive
            if recursive is None
            else recursive
        )

        pattern = "**/*" if use_recursive else "*"

        pdf_paths = sorted(
            (
                path
                for path in directory.glob(pattern)
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: str(
                path.relative_to(directory)
            ).lower(),
        )

        return tuple(
            self._discover_pdf(
                path=path,
                root_directory=directory,
            )
            for path in pdf_paths
        )

    def _process_file(
        self,
        discovered_pdf: DiscoveredPDF,
        metadata: Mapping[str, Any],
    ) -> PipelineFileResult:
        """Process one discovered PDF."""

        existing_documents = (
            self._find_documents_for_source(
                discovered_pdf.relative_path
            )
        )

        existing_checksums = {
            str(
                document.metadata.get(
                    INGESTION_CHECKSUM_KEY,
                    "",
                )
            )
            for document in existing_documents
        }

        if (
            existing_documents
            and existing_checksums
            == {discovered_pdf.checksum}
        ):
            return PipelineFileResult(
                source_path=discovered_pdf.source_path,
                relative_path=discovered_pdf.relative_path,
                source_filename=(
                    discovered_pdf.source_filename
                ),
                checksum=discovered_pdf.checksum,
                status=PipelineFileStatus.SKIPPED,
                pages_processed=0,
                chunks_stored=len(existing_documents),
                document_ids=tuple(
                    document.id
                    for document in existing_documents
                ),
            )

        resolved_metadata = dict(metadata)

        resolved_metadata.update(
            {
                INGESTION_SOURCE_PATH_KEY: str(
                    discovered_pdf.source_path
                ),
                INGESTION_RELATIVE_PATH_KEY: (
                    discovered_pdf.relative_path
                ),
                INGESTION_CHECKSUM_KEY: (
                    discovered_pdf.checksum
                ),
                INGESTION_FILE_SIZE_KEY: (
                    discovered_pdf.file_size
                ),
                INGESTION_PIPELINE_VERSION_KEY: (
                    INGESTION_PIPELINE_VERSION
                ),
            }
        )

        if not existing_documents:
            ingestion_result = (
                self._ingestion_service.ingest_pdf(
                    file_path=discovered_pdf.source_path,
                    metadata=resolved_metadata,
                )
            )

            return self._create_success_result(
                discovered_pdf=discovered_pdf,
                ingestion_result=ingestion_result,
                status=PipelineFileStatus.INGESTED,
            )

        return self._replace_changed_document(
            discovered_pdf=discovered_pdf,
            existing_documents=existing_documents,
            metadata=resolved_metadata,
        )

    def _replace_changed_document(
        self,
        discovered_pdf: DiscoveredPDF,
        existing_documents: Sequence[Document],
        metadata: Mapping[str, Any],
    ) -> PipelineFileResult:
        """
        Replace vectors for a changed PDF.

        Existing documents are retained in memory during replacement. When
        ingestion fails, the previous records are restored.
        """

        previous_documents = [
            Document(
                id=document.id,
                text=document.text,
                metadata=dict(document.metadata),
            )
            for document in existing_documents
        ]

        previous_ids = [
            document.id
            for document in previous_documents
        ]

        try:
            for document_id in previous_ids:
                self.vector_store.delete_document(
                    document_id
                )

            ingestion_result = (
                self._ingestion_service.ingest_pdf(
                    file_path=discovered_pdf.source_path,
                    metadata=metadata,
                )
            )
        except Exception as ingestion_error:
            rollback_errors: list[str] = []

            for document in previous_documents:
                try:
                    if document.id not in self.vector_store:
                        self.vector_store.add_documents(
                            [document]
                        )
                except Exception as rollback_error:
                    rollback_errors.append(
                        str(rollback_error)
                    )

            if rollback_errors:
                raise DocumentReplacementError(
                    "Document update failed and previous vectors "
                    "could not be completely restored. "
                    f"Ingestion error: {ingestion_error}. "
                    "Rollback errors: "
                    + "; ".join(rollback_errors)
                ) from ingestion_error

            raise DocumentReplacementError(
                "Document update failed. Previous vectors were "
                f"restored successfully: {ingestion_error}"
            ) from ingestion_error

        return self._create_success_result(
            discovered_pdf=discovered_pdf,
            ingestion_result=ingestion_result,
            status=PipelineFileStatus.UPDATED,
        )

    def _find_documents_for_source(
        self,
        relative_path: str,
    ) -> list[Document]:
        """Return stored chunks belonging to one source PDF."""

        documents = self.vector_store.list_documents()

        return sorted(
            (
                document
                for document in documents
                if document.metadata.get(
                    INGESTION_RELATIVE_PATH_KEY
                )
                == relative_path
            ),
            key=lambda document: (
                int(
                    document.metadata.get(
                        "page_number",
                        0,
                    )
                ),
                int(
                    document.metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
                document.id,
            ),
        )

    def _discover_pdf(
        self,
        path: Path,
        root_directory: Path,
    ) -> DiscoveredPDF:
        """Build discovery metadata for one PDF."""

        resolved_path = path.resolve()
        relative_path = (
            resolved_path.relative_to(
                root_directory
            ).as_posix()
        )

        return DiscoveredPDF(
            source_path=resolved_path,
            relative_path=relative_path,
            source_filename=resolved_path.name,
            checksum=self._calculate_checksum(
                resolved_path
            ),
            file_size=resolved_path.stat().st_size,
        )

    def _calculate_checksum(
        self,
        file_path: Path,
    ) -> str:
        """Calculate a streaming SHA-256 file checksum."""

        digest = hashlib.sha256()

        with file_path.open("rb") as file_handle:
            while True:
                block = file_handle.read(
                    self._checksum_block_size
                )

                if not block:
                    break

                digest.update(block)

        return digest.hexdigest()

    @staticmethod
    def _validate_unique_filenames(
        discovered_files: Sequence[DiscoveredPDF],
    ) -> None:
        """
        Reject duplicate PDF filenames within one ingestion run.

        Current deterministic chunk IDs are based on source filename, page,
        position, and text. Two PDFs with the same filename could therefore
        create conflicting identifiers even when located in separate folders.
        """

        filename_paths: dict[str, list[str]] = {}

        for discovered_file in discovered_files:
            normalized_filename = (
                discovered_file.source_filename.lower()
            )

            filename_paths.setdefault(
                normalized_filename,
                [],
            ).append(
                discovered_file.relative_path
            )

        duplicates = {
            filename: paths
            for filename, paths in filename_paths.items()
            if len(paths) > 1
        }

        if not duplicates:
            return

        details = "; ".join(
            f"{filename}: {', '.join(paths)}"
            for filename, paths in sorted(
                duplicates.items()
            )
        )

        raise DuplicateSourceFilenameError(
            "Duplicate PDF filenames were found. Rename the "
            "documents before ingestion because current chunk IDs "
            f"include only the source filename. {details}"
        )

    @staticmethod
    def _create_success_result(
        discovered_pdf: DiscoveredPDF,
        ingestion_result: DocumentIngestionResult,
        status: PipelineFileStatus,
    ) -> PipelineFileResult:
        """Convert a lower-level result into a pipeline result."""

        if status not in {
            PipelineFileStatus.INGESTED,
            PipelineFileStatus.UPDATED,
        }:
            raise ValueError(
                "Success result status must be INGESTED or UPDATED."
            )

        return PipelineFileResult(
            source_path=discovered_pdf.source_path,
            relative_path=discovered_pdf.relative_path,
            source_filename=(
                discovered_pdf.source_filename
            ),
            checksum=discovered_pdf.checksum,
            status=status,
            pages_processed=ingestion_result.total_pages,
            chunks_stored=ingestion_result.vectors_stored,
            document_ids=ingestion_result.document_ids,
        )

    @staticmethod
    def _build_pipeline_result(
        directory_path: Path,
        file_results: Sequence[PipelineFileResult],
    ) -> IngestionPipelineResult:
        """Build the aggregate pipeline summary."""

        results = tuple(file_results)

        return IngestionPipelineResult(
            directory_path=directory_path,
            files=results,
            files_discovered=len(results),
            files_ingested=sum(
                result.status
                == PipelineFileStatus.INGESTED
                for result in results
            ),
            files_updated=sum(
                result.status
                == PipelineFileStatus.UPDATED
                for result in results
            ),
            files_skipped=sum(
                result.status
                == PipelineFileStatus.SKIPPED
                for result in results
            ),
            files_failed=sum(
                result.status
                == PipelineFileStatus.FAILED
                for result in results
            ),
            total_pages=sum(
                result.pages_processed
                for result in results
                if result.status
                in {
                    PipelineFileStatus.INGESTED,
                    PipelineFileStatus.UPDATED,
                }
            ),
            total_chunks_stored=sum(
                result.chunks_stored
                for result in results
                if result.status
                in {
                    PipelineFileStatus.INGESTED,
                    PipelineFileStatus.UPDATED,
                }
            ),
        )

    def _resolve_directory(
        self,
        directory_path: PathLike | None,
    ) -> Path:
        """Resolve and validate the directory used for the current run."""

        if directory_path is None:
            path = self._document_directory
        else:
            path = self._validate_directory_value(
                directory_path
            )

        if not path.exists():
            raise InvalidIngestionDirectoryError(
                f"Document directory does not exist: '{path}'."
            )

        if not path.is_dir():
            raise InvalidIngestionDirectoryError(
                f"Document path is not a directory: '{path}'."
            )

        return path.resolve()

    @staticmethod
    def _validate_directory_value(
        directory_path: PathLike,
    ) -> Path:
        """Validate a directory-path value without requiring existence."""

        if not isinstance(
            directory_path,
            (str, Path),
        ):
            raise TypeError(
                "document_directory must be a string or "
                "pathlib.Path."
            )

        if (
            isinstance(directory_path, str)
            and not directory_path.strip()
        ):
            raise ValueError(
                "document_directory cannot be empty."
            )

        return Path(
            directory_path
        ).expanduser()

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy optional pipeline metadata."""

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping."
            )

        return dict(metadata)


def run_ingestion_pipeline(
    ingestion_service: PDFIngestionService,
    directory_path: PathLike = "data/rag_documents",
    metadata: Mapping[str, Any] | None = None,
    recursive: bool = False,
    continue_on_error: bool = True,
) -> IngestionPipelineResult:
    """
    Convenience function for one directory-ingestion run.

    Args:
        ingestion_service:
            Configured lower-level PDF ingestion service.

        directory_path:
            Local directory containing PDF files.

        metadata:
            Optional metadata applied to every ingested chunk.

        recursive:
            Whether nested folders should be scanned.

        continue_on_error:
            Whether remaining PDFs should continue after a failure.

    Returns:
        Typed pipeline result.
    """

    pipeline = DocumentIngestionPipeline(
        ingestion_service=ingestion_service,
        document_directory=directory_path,
        recursive=recursive,
    )

    return pipeline.run(
        metadata=metadata,
        continue_on_error=continue_on_error,
    )