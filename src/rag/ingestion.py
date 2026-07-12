"""
PDF ingestion orchestration for the Finance Agentic AI System.

This module coordinates the existing RAG infrastructure:

1. Load a local PDF using a compatible document loader.
2. Split the extracted pages using TextChunker.
3. Convert chunks into vector-store Document objects.
4. Store the documents using InMemoryVectorStore.

Embedding generation is delegated to the embedding service configured inside
the vector store. The ingestion service does not calculate embeddings itself.

This module deliberately contains no:

* PDF extraction implementation
* Text-chunking implementation
* Embedding algorithm
* Vector similarity logic
* Retrieval logic
* LLM calls
* Finance calculation logic
* S3 integration
* PostgreSQL or pgvector integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)

from src.rag.document_loader import (
    PDFDocument,
    PDFDocumentLoader,
)
from src.rag.text_chunker import (
    DocumentChunk,
    TextChunker,
)
from src.rag.vector_store import (
    Document,
    InMemoryVectorStore,
    VectorStoreError,
)


PathLike: TypeAlias = str | Path


@runtime_checkable
class DocumentLoaderProtocol(Protocol):
    """
    Interface required by PDFIngestionService.

    Any object with a compatible callable ``load`` method can be supplied.
    This supports testing with lightweight stub loaders without requiring
    inheritance from PDFDocumentLoader.
    """

    def load(
        self,
        file_path: PathLike,
    ) -> PDFDocument:
        """Load one PDF document."""
        ...


class IngestionError(RuntimeError):
    """Base exception raised when document ingestion fails."""


class DuplicateDocumentError(IngestionError):
    """Raised when one or more chunks already exist in the vector store."""


class DirectoryIngestionError(IngestionError):
    """Raised when a directory cannot be used for PDF ingestion."""


@dataclass(frozen=True)
class IngestionFailure:
    """
    Represents one PDF that failed during batch ingestion.

    Attributes:
        source_path:
            Path supplied for the failed document.

        error_type:
            Exception class name.

        message:
            Human-readable error message.
    """

    source_path: Path
    error_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path.")

        if not isinstance(self.error_type, str):
            raise TypeError("error_type must be a string.")

        cleaned_error_type = self.error_type.strip()

        if not cleaned_error_type:
            raise ValueError("error_type cannot be empty.")

        if not isinstance(self.message, str):
            raise TypeError("message must be a string.")

        cleaned_message = self.message.strip()

        if not cleaned_message:
            raise ValueError("message cannot be empty.")

        object.__setattr__(
            self,
            "error_type",
            cleaned_error_type,
        )
        object.__setattr__(
            self,
            "message",
            cleaned_message,
        )


@dataclass(frozen=True)
class DocumentIngestionResult:
    """
    Result of successfully ingesting one PDF.

    Attributes:
        source_path:
            Resolved path of the source PDF.

        source_filename:
            Source PDF filename.

        total_pages:
            Total pages found in the source PDF.

        empty_page_numbers:
            One-based page numbers with no extractable text.

        chunks_created:
            Number of chunks created from non-empty pages.

        vectors_stored:
            Number of chunk documents stored in the vector store.

        document_ids:
            Vector-store identifiers created for the chunks.

        metadata:
            Document-level metadata supplied for ingestion.
    """

    source_path: Path
    source_filename: str
    total_pages: int
    empty_page_numbers: tuple[int, ...]
    chunks_created: int
    vectors_stored: int
    document_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = self.source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        if isinstance(self.total_pages, bool) or not isinstance(
            self.total_pages,
            int,
        ):
            raise TypeError("total_pages must be an integer.")

        if self.total_pages < 0:
            raise ValueError("total_pages cannot be negative.")

        if not isinstance(self.empty_page_numbers, tuple):
            raise TypeError(
                "empty_page_numbers must be a tuple."
            )

        for page_number in self.empty_page_numbers:
            if isinstance(page_number, bool) or not isinstance(
                page_number,
                int,
            ):
                raise TypeError(
                    "empty page numbers must be integers."
                )

            if page_number <= 0:
                raise ValueError(
                    "empty page numbers must be greater than zero."
                )

            if page_number > self.total_pages:
                raise ValueError(
                    "empty page numbers cannot exceed total_pages."
                )

        if isinstance(self.chunks_created, bool) or not isinstance(
            self.chunks_created,
            int,
        ):
            raise TypeError(
                "chunks_created must be an integer."
            )

        if self.chunks_created < 0:
            raise ValueError(
                "chunks_created cannot be negative."
            )

        if isinstance(self.vectors_stored, bool) or not isinstance(
            self.vectors_stored,
            int,
        ):
            raise TypeError(
                "vectors_stored must be an integer."
            )

        if self.vectors_stored < 0:
            raise ValueError(
                "vectors_stored cannot be negative."
            )

        if self.vectors_stored != self.chunks_created:
            raise ValueError(
                "vectors_stored must match chunks_created."
            )

        if not isinstance(self.document_ids, tuple):
            raise TypeError("document_ids must be a tuple.")

        if len(self.document_ids) != self.vectors_stored:
            raise ValueError(
                "document_ids must match vectors_stored."
            )

        if any(
            not isinstance(document_id, str)
            or not document_id.strip()
            for document_id in self.document_ids
        ):
            raise ValueError(
                "document_ids must contain non-empty strings."
            )

        if len(self.document_ids) != len(
            set(self.document_ids)
        ):
            raise ValueError(
                "document_ids cannot contain duplicates."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        object.__setattr__(
            self,
            "source_filename",
            cleaned_filename,
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    @property
    def has_empty_pages(self) -> bool:
        """Return whether the source PDF contains empty pages."""

        return bool(self.empty_page_numbers)


@dataclass(frozen=True)
class BatchIngestionResult:
    """
    Result of ingesting multiple PDF documents.

    Attributes:
        successful_documents:
            Successful per-document results.

        failures:
            Documents that could not be ingested.

        total_files:
            Number of PDF paths attempted.

        documents_ingested:
            Number of successfully ingested PDFs.

        documents_failed:
            Number of failed PDFs.

        total_pages:
            Total pages across successful PDFs.

        chunks_created:
            Total chunks created across successful PDFs.

        vectors_stored:
            Total vectors stored across successful PDFs.
    """

    successful_documents: tuple[
        DocumentIngestionResult,
        ...,
    ]
    failures: tuple[IngestionFailure, ...]
    total_files: int
    documents_ingested: int
    documents_failed: int
    total_pages: int
    chunks_created: int
    vectors_stored: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.successful_documents,
            tuple,
        ):
            raise TypeError(
                "successful_documents must be a tuple."
            )

        if not all(
            isinstance(result, DocumentIngestionResult)
            for result in self.successful_documents
        ):
            raise TypeError(
                "successful_documents must contain "
                "DocumentIngestionResult instances."
            )

        if not isinstance(self.failures, tuple):
            raise TypeError("failures must be a tuple.")

        if not all(
            isinstance(failure, IngestionFailure)
            for failure in self.failures
        ):
            raise TypeError(
                "failures must contain IngestionFailure instances."
            )

        numeric_fields = {
            "total_files": self.total_files,
            "documents_ingested": self.documents_ingested,
            "documents_failed": self.documents_failed,
            "total_pages": self.total_pages,
            "chunks_created": self.chunks_created,
            "vectors_stored": self.vectors_stored,
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

        if self.documents_ingested != len(
            self.successful_documents
        ):
            raise ValueError(
                "documents_ingested must match "
                "successful_documents."
            )

        if self.documents_failed != len(self.failures):
            raise ValueError(
                "documents_failed must match failures."
            )

        if self.total_files != (
            self.documents_ingested
            + self.documents_failed
        ):
            raise ValueError(
                "total_files must equal successful and failed "
                "document counts."
            )

        expected_pages = sum(
            result.total_pages
            for result in self.successful_documents
        )

        if self.total_pages != expected_pages:
            raise ValueError(
                "total_pages must match successful results."
            )

        expected_chunks = sum(
            result.chunks_created
            for result in self.successful_documents
        )

        if self.chunks_created != expected_chunks:
            raise ValueError(
                "chunks_created must match successful results."
            )

        expected_vectors = sum(
            result.vectors_stored
            for result in self.successful_documents
        )

        if self.vectors_stored != expected_vectors:
            raise ValueError(
                "vectors_stored must match successful results."
            )

    @property
    def succeeded(self) -> bool:
        """Return whether every requested PDF was ingested."""

        return self.documents_failed == 0

    @property
    def partially_succeeded(self) -> bool:
        """Return whether the batch contains successes and failures."""

        return (
            self.documents_ingested > 0
            and self.documents_failed > 0
        )


class PDFIngestionService:
    """
    Coordinate PDF loading, chunking, embedding, and storage.

    The vector store owns the embedding service. Calling
    ``vector_store.add_documents`` therefore performs both embedding generation
    and vector storage through the existing public interface.

    Args:
        document_loader:
            Compatible loader providing a callable ``load`` method.
            A PDFDocumentLoader is created when omitted.

        text_chunker:
            Existing text chunker. A default chunker is created when omitted.

        vector_store:
            Existing in-memory vector store. A default store is created when
            omitted.
    """

    def __init__(
        self,
        document_loader: DocumentLoaderProtocol | None = None,
        text_chunker: TextChunker | None = None,
        vector_store: InMemoryVectorStore | None = None,
    ) -> None:
        if (
            document_loader is not None
            and not isinstance(
                document_loader,
                DocumentLoaderProtocol,
            )
        ):
            raise TypeError(
                 "document_loader must be a compatible loader "
                 "with a callable load method."
            )

        if (
            text_chunker is not None
            and not isinstance(
                text_chunker,
                TextChunker,
            )
        ):
            raise TypeError(
                "text_chunker must be a TextChunker."
            )

        if (
            vector_store is not None
            and not isinstance(
                vector_store,
                InMemoryVectorStore,
            )
        ):
            raise TypeError(
                "vector_store must be an InMemoryVectorStore."
            )

        if document_loader is None:
            self._document_loader: DocumentLoaderProtocol = (
              PDFDocumentLoader()
           )
        else:
            self._document_loader = document_loader

        if text_chunker is None:
            self._text_chunker = TextChunker()
        else:
            self._text_chunker = text_chunker

        if vector_store is None:
            self._vector_store = InMemoryVectorStore()
        else:
            self._vector_store = vector_store

    @property
    def document_loader(self) -> DocumentLoaderProtocol:
        """Return the configured document loader."""

        return self._document_loader

    @property
    def text_chunker(self) -> TextChunker:
        """Return the configured text chunker."""

        return self._text_chunker

    @property
    def vector_store(self) -> InMemoryVectorStore:
        """Return the configured vector store."""

        return self._vector_store

    def ingest_pdf(
        self,
        file_path: PathLike,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentIngestionResult:
        """
        Load, chunk, embed, and store one local PDF.

        Args:
            file_path:
                Local path to one PDF file.

            metadata:
                Optional metadata added to every chunk.

        Returns:
            Typed summary of the completed ingestion.
        """

        validated_metadata = self._validate_metadata(
            metadata
        )

        document = self._document_loader.load(file_path)

        chunks = self._text_chunker.chunk_document(
            document=document,
            metadata=validated_metadata,
        )

        vector_documents = (
            self._convert_chunks_to_documents(chunks)
        )

        stored_documents = self._store_documents(
            vector_documents
        )

        return self._build_document_result(
            document=document,
            chunks=chunks,
            stored_documents=stored_documents,
            metadata=validated_metadata,
        )

    def ingest_pdfs(
        self,
        file_paths: Sequence[PathLike],
        metadata: Mapping[str, Any] | None = None,
        continue_on_error: bool = True,
    ) -> BatchIngestionResult:
        """
        Ingest multiple PDF files.

        Each PDF is handled as its own atomic vector-store batch.
        """

        validated_paths = self._validate_file_paths(
            file_paths
        )
        validated_metadata = self._validate_metadata(
            metadata
        )

        if not isinstance(continue_on_error, bool):
            raise TypeError(
                "continue_on_error must be a boolean."
            )

        successful_results: list[
            DocumentIngestionResult
        ] = []
        failures: list[IngestionFailure] = []

        for file_path in validated_paths:
            try:
                result = self.ingest_pdf(
                    file_path=file_path,
                    metadata=validated_metadata,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise

                failures.append(
                    IngestionFailure(
                        source_path=self._safe_path(
                            file_path
                        ),
                        error_type=type(exc).__name__,
                        message=(
                            str(exc)
                            or type(exc).__name__
                        ),
                    )
                )
            else:
                successful_results.append(result)

        return self._build_batch_result(
            total_files=len(validated_paths),
            successful_results=successful_results,
            failures=failures,
        )

    def ingest_directory(
        self,
        directory_path: PathLike,
        metadata: Mapping[str, Any] | None = None,
        recursive: bool = False,
        continue_on_error: bool = True,
    ) -> BatchIngestionResult:
        """
        Ingest every PDF from a local directory.

        Files are sorted to keep ingestion order deterministic.
        """

        if not isinstance(recursive, bool):
            raise TypeError(
                "recursive must be a boolean."
            )

        if not isinstance(continue_on_error, bool):
            raise TypeError(
                "continue_on_error must be a boolean."
            )

        directory = self._validate_directory_path(
            directory_path
        )

        pattern = "**/*" if recursive else "*"

        pdf_paths = sorted(
            (
                path
                for path in directory.glob(pattern)
                if path.is_file()
                and path.suffix.lower() == ".pdf"
            ),
            key=lambda path: str(path).lower(),
        )

        return self.ingest_pdfs(
            file_paths=pdf_paths,
            metadata=metadata,
            continue_on_error=continue_on_error,
        )

    @staticmethod
    def _convert_chunks_to_documents(
        chunks: Sequence[DocumentChunk],
    ) -> list[Document]:
        """
        Convert chunker results into vector-store documents.
        """

        documents: list[Document] = []

        for chunk in chunks:
            if not isinstance(chunk, DocumentChunk):
                raise TypeError(
                    "chunks must contain DocumentChunk instances."
                )

            metadata = dict(chunk.metadata)

            metadata.update(
                {
                    "chunk_id": chunk.chunk_id,
                    "source_filename": (
                        chunk.source_filename
                    ),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )

            documents.append(
                Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=metadata,
                )
            )

        return documents

    def _store_documents(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """
        Store documents through the existing vector-store interface.
        """

        if not documents:
            return []

        existing_ids = [
            document.id
            for document in documents
            if document.id in self._vector_store
        ]

        if existing_ids:
            raise DuplicateDocumentError(
                "One or more chunks have already been ingested: "
                + ", ".join(sorted(existing_ids))
            )

        try:
            return self._vector_store.add_documents(
                list(documents)
            )
        except VectorStoreError as exc:
            message = str(exc)

            if (
                "already exists" in message.lower()
                or "duplicate" in message.lower()
            ):
                raise DuplicateDocumentError(
                    message
                ) from exc

            raise IngestionError(
                "Unable to store document chunks: "
                f"{message}"
            ) from exc
        except Exception as exc:
            raise IngestionError(
                "Unable to complete document ingestion."
            ) from exc

    @staticmethod
    def _build_document_result(
        document: PDFDocument,
        chunks: Sequence[DocumentChunk],
        stored_documents: Sequence[Document],
        metadata: Mapping[str, Any],
    ) -> DocumentIngestionResult:
        """Build the result for one successfully ingested PDF."""

        return DocumentIngestionResult(
            source_path=document.source_path,
            source_filename=document.source_filename,
            total_pages=document.total_pages,
            empty_page_numbers=(
                document.empty_page_numbers
            ),
            chunks_created=len(chunks),
            vectors_stored=len(stored_documents),
            document_ids=tuple(
                stored_document.id
                for stored_document in stored_documents
            ),
            metadata=dict(metadata),
        )

    @staticmethod
    def _build_batch_result(
        total_files: int,
        successful_results: Sequence[
            DocumentIngestionResult
        ],
        failures: Sequence[IngestionFailure],
    ) -> BatchIngestionResult:
        """Build an aggregate result for multiple PDF files."""

        successful_tuple = tuple(
            successful_results
        )
        failure_tuple = tuple(failures)

        return BatchIngestionResult(
            successful_documents=successful_tuple,
            failures=failure_tuple,
            total_files=total_files,
            documents_ingested=len(
                successful_tuple
            ),
            documents_failed=len(
                failure_tuple
            ),
            total_pages=sum(
                result.total_pages
                for result in successful_tuple
            ),
            chunks_created=sum(
                result.chunks_created
                for result in successful_tuple
            ),
            vectors_stored=sum(
                result.vectors_stored
                for result in successful_tuple
            ),
        )

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy optional document metadata."""

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError(
                "metadata must be a mapping."
            )

        return dict(metadata)

    @staticmethod
    def _validate_file_paths(
        file_paths: Sequence[PathLike],
    ) -> tuple[PathLike, ...]:
        """Validate a sequence of PDF paths."""

        if isinstance(
            file_paths,
            (str, bytes, Path),
        ):
            raise TypeError(
                "file_paths must be a sequence of paths, "
                "not one path."
            )

        if not isinstance(file_paths, Sequence):
            raise TypeError(
                "file_paths must be a sequence."
            )

        validated_paths: list[PathLike] = []

        for file_path in file_paths:
            if not isinstance(
                file_path,
                (str, Path),
            ):
                raise TypeError(
                    "Each file path must be a string or "
                    "pathlib.Path."
                )

            if (
                isinstance(file_path, str)
                and not file_path.strip()
            ):
                raise ValueError(
                    "File paths cannot contain empty strings."
                )

            validated_paths.append(file_path)

        return tuple(validated_paths)

    @staticmethod
    def _validate_directory_path(
        directory_path: PathLike,
    ) -> Path:
        """Validate and resolve a local ingestion directory."""

        if not isinstance(
            directory_path,
            (str, Path),
        ):
            raise TypeError(
                "directory_path must be a string or "
                "pathlib.Path."
            )

        if (
            isinstance(directory_path, str)
            and not directory_path.strip()
        ):
            raise ValueError(
                "directory_path cannot be empty."
            )

        path = Path(
            directory_path
        ).expanduser()

        if not path.exists():
            raise DirectoryIngestionError(
                f"Directory does not exist: '{path}'."
            )

        if not path.is_dir():
            raise DirectoryIngestionError(
                f"Path is not a directory: '{path}'."
            )

        return path.resolve()

    @staticmethod
    def _safe_path(
        file_path: PathLike,
    ) -> Path:
        """
        Convert a failed input path to Path without requiring it to exist.
        """

        return Path(file_path).expanduser()


def ingest_pdf(
    file_path: PathLike,
    vector_store: InMemoryVectorStore | None = None,
    metadata: Mapping[str, Any] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> DocumentIngestionResult:
    """
    Convenience function for ingesting one local PDF.

    Args:
        file_path:
            Local PDF path.

        vector_store:
            Optional existing vector store.

        metadata:
            Optional metadata copied to every chunk.

        chunk_size:
            Maximum number of characters per chunk.

        chunk_overlap:
            Number of overlapping characters between chunks.

    Returns:
        Typed ingestion result.
    """

    service = PDFIngestionService(
        text_chunker=TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        vector_store=vector_store,
    )

    return service.ingest_pdf(
        file_path=file_path,
        metadata=metadata,
    )