"""
Text chunking utilities for the Finance Agentic AI System.

This module converts page-level PDF text from ``document_loader.py`` into
smaller overlapping text chunks suitable for a later embedding and ingestion
stage.

The module deliberately contains no:

* PDF-reading logic
* OCR logic
* Embedding generation
* Vector-store operations
* LLM calls
* Finance calculation logic

Chunking is performed independently for each PDF page. A chunk never combines
text from different pages, which preserves accurate page-level citations and
metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.rag.document_loader import PDFDocument, PDFPage


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class TextChunkerError(RuntimeError):
    """Base exception raised when document chunking fails."""


class InvalidChunkConfigurationError(TextChunkerError):
    """Raised when chunk-size or overlap configuration is invalid."""


@dataclass(frozen=True)
class DocumentChunk:
    """
    Represents one chunk created from a PDF page.

    Attributes:
        chunk_id:
            Deterministic unique identifier for the chunk.

        text:
            Text contained in this chunk.

        source_filename:
            Filename of the source PDF.

        page_number:
            One-based page number from which the chunk was created.

        chunk_index:
            Zero-based position of the chunk within its source page.

        start_char:
            Inclusive character offset within the normalized page text.

        end_char:
            Exclusive character offset within the normalized page text.

        metadata:
            Additional metadata that can later be stored in a vector database.
    """

    chunk_id: str
    text: str
    source_filename: str
    page_number: int
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str):
            raise TypeError("chunk_id must be a string.")

        cleaned_chunk_id = self.chunk_id.strip()

        if not cleaned_chunk_id:
            raise ValueError("chunk_id cannot be empty.")

        if not isinstance(self.text, str):
            raise TypeError("chunk text must be a string.")

        if not self.text:
            raise ValueError("chunk text cannot be empty.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = self.source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        if isinstance(self.page_number, bool) or not isinstance(
            self.page_number,
            int,
        ):
            raise TypeError("page_number must be an integer.")

        if self.page_number <= 0:
            raise ValueError("page_number must be greater than zero.")

        if isinstance(self.chunk_index, bool) or not isinstance(
            self.chunk_index,
            int,
        ):
            raise TypeError("chunk_index must be an integer.")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative.")

        if isinstance(self.start_char, bool) or not isinstance(
            self.start_char,
            int,
        ):
            raise TypeError("start_char must be an integer.")

        if self.start_char < 0:
            raise ValueError("start_char cannot be negative.")

        if isinstance(self.end_char, bool) or not isinstance(
            self.end_char,
            int,
        ):
            raise TypeError("end_char must be an integer.")

        if self.end_char <= self.start_char:
            raise ValueError(
                "end_char must be greater than start_char."
            )

        if self.end_char - self.start_char != len(self.text):
            raise ValueError(
                "Character offsets must match the chunk text length."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        object.__setattr__(self, "chunk_id", cleaned_chunk_id)
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


class TextChunker:
    """
    Split PDF page text into deterministic overlapping character chunks.

    Each page is chunked independently. Empty pages produce no chunks because
    there is no text to embed or retrieve.

    Args:
        chunk_size:
            Maximum number of characters in each chunk.

        chunk_overlap:
            Number of characters repeated between consecutive chunks.

    Example:
        For ``chunk_size=1000`` and ``chunk_overlap=200``:

        * Chunk 0 covers characters 0 to 1000
        * Chunk 1 covers characters 800 to 1800
        * Chunk 2 covers characters 1600 to 2600
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._validate_configuration(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        """Return the configured maximum chunk size."""

        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Return the configured character overlap."""

        return self._chunk_overlap

    @property
    def step_size(self) -> int:
        """
        Return the number of characters advanced between chunks.

        The step is calculated as:

        ``chunk_size - chunk_overlap``
        """

        return self._chunk_size - self._chunk_overlap

    def chunk_document(
        self,
        document: PDFDocument,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[DocumentChunk, ...]:
        """
        Split all non-empty pages in one PDF document.

        Args:
            document:
                Typed PDF document returned by ``PDFDocumentLoader``.

            metadata:
                Optional document-level metadata added to every chunk. Typical
                future values may include document category, financial year,
                business unit, version, or access permissions.

        Returns:
            Chunks in document order and page order.

        Raises:
            TypeError:
                If document is not a PDFDocument or metadata is invalid.
        """

        if not isinstance(document, PDFDocument):
            raise TypeError("document must be a PDFDocument.")

        document_metadata = self._validate_metadata(metadata)
        chunks: list[DocumentChunk] = []

        for page in document.pages:
            page_chunks = self.chunk_page(
                page=page,
                metadata=document_metadata,
            )
            chunks.extend(page_chunks)

        return tuple(chunks)

    def chunk_page(
        self,
        page: PDFPage,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[DocumentChunk, ...]:
        """
        Split one PDF page into overlapping chunks.

        Empty pages return an empty tuple. Character positions are relative to
        ``page.text``.

        Args:
            page:
                PDFPage produced by the document loader.

            metadata:
                Optional metadata added to each resulting chunk.

        Returns:
            Ordered chunks for the supplied page.
        """

        if not isinstance(page, PDFPage):
            raise TypeError("page must be a PDFPage.")

        chunk_metadata = self._validate_metadata(metadata)

        if page.is_empty:
            return ()

        return self.chunk_text(
            text=page.text,
            source_filename=page.source_filename,
            page_number=page.page_number,
            metadata=chunk_metadata,
        )

    def chunk_text(
        self,
        text: str,
        source_filename: str,
        page_number: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[DocumentChunk, ...]:
        """
        Split text associated with one source page.

        This method is useful for direct unit testing and for future document
        loaders that produce the same page-level information.

        Args:
            text:
                Page text to split.

            source_filename:
                Source PDF filename.

            page_number:
                One-based page number.

            metadata:
                Optional metadata copied into each chunk.

        Returns:
            Ordered tuple of DocumentChunk objects.
        """

        validated_text = self._validate_text(text)
        validated_filename = self._validate_source_filename(
            source_filename
        )
        validated_page_number = self._validate_page_number(
            page_number
        )
        base_metadata = self._validate_metadata(metadata)

        if not validated_text:
            return ()

        chunks: list[DocumentChunk] = []
        text_length = len(validated_text)
        start_char = 0
        chunk_index = 0

        while start_char < text_length:
            end_char = min(
                start_char + self._chunk_size,
                text_length,
            )

            chunk_text = validated_text[start_char:end_char]

            chunk_id = self._create_chunk_id(
                source_filename=validated_filename,
                page_number=validated_page_number,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                text=chunk_text,
            )

            resolved_metadata = self._build_chunk_metadata(
                base_metadata=base_metadata,
                chunk_id=chunk_id,
                source_filename=validated_filename,
                page_number=validated_page_number,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_filename=validated_filename,
                    page_number=validated_page_number,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=resolved_metadata,
                )
            )

            if end_char >= text_length:
                break

            start_char += self.step_size
            chunk_index += 1

        return tuple(chunks)

    @staticmethod
    def _validate_configuration(
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        """Validate chunk-size and overlap configuration."""

        if isinstance(chunk_size, bool) or not isinstance(
            chunk_size,
            int,
        ):
            raise TypeError("chunk_size must be an integer.")

        if chunk_size <= 0:
            raise InvalidChunkConfigurationError(
                "chunk_size must be greater than zero."
            )

        if isinstance(chunk_overlap, bool) or not isinstance(
            chunk_overlap,
            int,
        ):
            raise TypeError("chunk_overlap must be an integer.")

        if chunk_overlap < 0:
            raise InvalidChunkConfigurationError(
                "chunk_overlap cannot be negative."
            )

        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigurationError(
                "chunk_overlap must be smaller than chunk_size."
            )

    @staticmethod
    def _validate_text(text: str) -> str:
        """
        Validate page text.

        The loader already normalizes leading and trailing whitespace. This
        method performs the same normalization for direct ``chunk_text`` use.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        return text.strip()

    @staticmethod
    def _validate_source_filename(
        source_filename: str,
    ) -> str:
        """Validate and normalize the source filename."""

        if not isinstance(source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        return cleaned_filename

    @staticmethod
    def _validate_page_number(page_number: int) -> int:
        """Validate a one-based page number."""

        if isinstance(page_number, bool) or not isinstance(
            page_number,
            int,
        ):
            raise TypeError("page_number must be an integer.")

        if page_number <= 0:
            raise ValueError(
                "page_number must be greater than zero."
            )

        return page_number

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate and copy optional metadata."""

        if metadata is None:
            return {}

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        return dict(metadata)

    @staticmethod
    def _create_chunk_id(
        source_filename: str,
        page_number: int,
        chunk_index: int,
        start_char: int,
        end_char: int,
        text: str,
    ) -> str:
        """
        Create a deterministic chunk identifier.

        The identifier is derived from source and chunk content. The same
        document text and chunking configuration therefore produce the same
        identifiers across repeated runs.
        """

        identity_value = "|".join(
            (
                source_filename,
                str(page_number),
                str(chunk_index),
                str(start_char),
                str(end_char),
                text,
            )
        )

        digest = hashlib.sha256(
            identity_value.encode("utf-8")
        ).hexdigest()

        return f"chunk_{digest}"

    @staticmethod
    def _build_chunk_metadata(
        base_metadata: Mapping[str, Any],
        chunk_id: str,
        source_filename: str,
        page_number: int,
        chunk_index: int,
        start_char: int,
        end_char: int,
    ) -> dict[str, Any]:
        """
        Combine user metadata with required chunk metadata.

        Required system metadata overrides conflicting values supplied by the
        caller so that chunk records remain internally consistent.
        """

        metadata = dict(base_metadata)

        metadata.update(
            {
                "chunk_id": chunk_id,
                "source_filename": source_filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
            }
        )

        return metadata


def chunk_document(
    document: PDFDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[DocumentChunk, ...]:
    """
    Convenience function for chunking one PDF document.

    Args:
        document:
            PDFDocument returned by the document loader.

        chunk_size:
            Maximum characters in each chunk.

        chunk_overlap:
            Repeated characters between consecutive chunks.

        metadata:
            Optional metadata added to every chunk.

    Returns:
        Ordered tuple of document chunks.
    """

    chunker = TextChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return chunker.chunk_document(
        document=document,
        metadata=metadata,
    )