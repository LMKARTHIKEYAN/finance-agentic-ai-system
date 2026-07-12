"""
Local PDF document loader for the Finance Agentic AI System.

This module reads text-based PDF files from local storage and returns typed,
page-level results. It deliberately contains no chunking, embedding,
vector-store, LLM, or finance calculation logic.

OCR is not performed. A page whose extracted text is empty is retained and
marked as empty so that downstream ingestion code can decide how to handle it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pypdf import PdfReader
from pypdf.errors import PdfReadError


PathLike: TypeAlias = str | Path


class DocumentLoaderError(RuntimeError):
    """Base exception raised when a document cannot be loaded."""


class DocumentFileNotFoundError(DocumentLoaderError):
    """Raised when the requested PDF path does not exist."""


class InvalidDocumentError(DocumentLoaderError):
    """Raised when the requested path is not a readable PDF document."""


class EncryptedDocumentError(DocumentLoaderError):
    """Raised when a PDF is encrypted and cannot be opened without a password."""


@dataclass(frozen=True)
class PDFPage:
    """
    Text extracted from one PDF page.

    Attributes:
        page_number:
            One-based page number from the source PDF.
        text:
            Extracted page text. Empty pages contain an empty string.
        source_filename:
            Filename of the source PDF, without its directory path.
        is_empty:
            True when the page has no non-whitespace extracted text.
    """

    page_number: int
    text: str
    source_filename: str
    is_empty: bool

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or not isinstance(
            self.page_number,
            int,
        ):
            raise TypeError("page_number must be an integer.")

        if self.page_number <= 0:
            raise ValueError("page_number must be greater than zero.")

        if not isinstance(self.text, str):
            raise TypeError("page text must be a string.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = self.source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        if not isinstance(self.is_empty, bool):
            raise TypeError("is_empty must be a boolean.")

        cleaned_text = self.text.strip()
        expected_is_empty = not bool(cleaned_text)

        if self.is_empty != expected_is_empty:
            raise ValueError(
                "is_empty must match whether page text is empty."
            )

        object.__setattr__(self, "text", cleaned_text)
        object.__setattr__(
            self,
            "source_filename",
            cleaned_filename,
        )


@dataclass(frozen=True)
class PDFDocument:
    """
    Complete page-level result for one loaded PDF.

    Attributes:
        source_path:
            Resolved local path of the source PDF.
        source_filename:
            Filename of the source PDF.
        pages:
            Extracted pages in original PDF order.
        total_pages:
            Number of pages in the PDF.
        empty_page_numbers:
            One-based page numbers with no extractable text.
    """

    source_path: Path
    source_filename: str
    pages: tuple[PDFPage, ...]
    total_pages: int
    empty_page_numbers: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise TypeError("source_path must be a pathlib.Path.")

        if not isinstance(self.source_filename, str):
            raise TypeError("source_filename must be a string.")

        cleaned_filename = self.source_filename.strip()

        if not cleaned_filename:
            raise ValueError("source_filename cannot be empty.")

        if not isinstance(self.pages, tuple):
            raise TypeError("pages must be a tuple.")

        if not all(isinstance(page, PDFPage) for page in self.pages):
            raise TypeError("pages must contain only PDFPage instances.")

        if isinstance(self.total_pages, bool) or not isinstance(
            self.total_pages,
            int,
        ):
            raise TypeError("total_pages must be an integer.")

        if self.total_pages < 0:
            raise ValueError("total_pages cannot be negative.")

        if self.total_pages != len(self.pages):
            raise ValueError("total_pages must match the number of pages.")

        if not isinstance(self.empty_page_numbers, tuple):
            raise TypeError("empty_page_numbers must be a tuple.")

        derived_empty_pages = tuple(
            page.page_number for page in self.pages if page.is_empty
        )

        if self.empty_page_numbers != derived_empty_pages:
            raise ValueError(
                "empty_page_numbers must match empty pages."
            )

        expected_page_numbers = tuple(
            range(1, self.total_pages + 1)
        )
        actual_page_numbers = tuple(
            page.page_number for page in self.pages
        )

        if actual_page_numbers != expected_page_numbers:
            raise ValueError(
                "pages must use sequential one-based page numbers."
            )

        if any(
            page.source_filename != cleaned_filename
            for page in self.pages
        ):
            raise ValueError(
                "all pages must use the document source_filename."
            )

        object.__setattr__(
            self,
            "source_filename",
            cleaned_filename,
        )

    @property
    def has_empty_pages(self) -> bool:
        """Return whether one or more pages contain no extracted text."""

        return bool(self.empty_page_numbers)


class PDFDocumentLoader:
    """
    Load text from local PDF files page by page.

    The loader uses ``pypdf`` for text extraction and intentionally does not
    perform OCR. Scanned-image pages therefore remain present with
    ``is_empty=True`` when no text layer can be extracted.
    """

    def load(self, file_path: PathLike) -> PDFDocument:
        """
        Load one local PDF document.

        Args:
            file_path:
                Local path to a PDF file.

        Returns:
            A typed PDFDocument containing page-level extraction results.

        Raises:
            TypeError:
                If file_path is not a string or pathlib.Path.
            ValueError:
                If file_path is an empty string.
            DocumentFileNotFoundError:
                If the path does not exist.
            InvalidDocumentError:
                If the path is not a file, does not have a .pdf extension,
                cannot be parsed, or page text extraction fails.
            EncryptedDocumentError:
                If the PDF is encrypted and requires a password.
        """

        resolved_path = self._validate_path(file_path)
        source_filename = resolved_path.name
        reader = self._create_reader(resolved_path)
        self._validate_not_encrypted(reader, resolved_path)

        pages: list[PDFPage] = []

        try:
            for page_number, page in enumerate(
                reader.pages,
                start=1,
            ):
                extracted_text = page.extract_text()
                text = extracted_text if extracted_text is not None else ""
                cleaned_text = text.strip()

                pages.append(
                    PDFPage(
                        page_number=page_number,
                        text=cleaned_text,
                        source_filename=source_filename,
                        is_empty=not bool(cleaned_text),
                    )
                )
        except Exception as exc:
            raise InvalidDocumentError(
                f"Failed to extract text from PDF '{resolved_path}'."
            ) from exc

        page_tuple = tuple(pages)
        empty_page_numbers = tuple(
            page.page_number for page in page_tuple if page.is_empty
        )

        return PDFDocument(
            source_path=resolved_path,
            source_filename=source_filename,
            pages=page_tuple,
            total_pages=len(page_tuple),
            empty_page_numbers=empty_page_numbers,
        )

    @staticmethod
    def _validate_path(file_path: PathLike) -> Path:
        """Validate and resolve a local PDF path."""

        if not isinstance(file_path, (str, Path)):
            raise TypeError(
                "file_path must be a string or pathlib.Path."
            )

        if isinstance(file_path, str) and not file_path.strip():
            raise ValueError("file_path cannot be empty.")

        path = Path(file_path).expanduser()

        if not path.exists():
            raise DocumentFileNotFoundError(
                f"PDF file does not exist: '{path}'."
            )

        if not path.is_file():
            raise InvalidDocumentError(
                f"PDF path is not a file: '{path}'."
            )

        if path.suffix.lower() != ".pdf":
            raise InvalidDocumentError(
                f"Document must have a .pdf extension: '{path}'."
            )

        return path.resolve()

    @staticmethod
    def _create_reader(file_path: Path) -> PdfReader:
        """Create a pypdf reader and normalize parsing failures."""

        try:
            return PdfReader(str(file_path))
        except (PdfReadError, OSError, ValueError) as exc:
            raise InvalidDocumentError(
                f"Unable to read PDF '{file_path}'."
            ) from exc

    @staticmethod
    def _validate_not_encrypted(
        reader: PdfReader,
        file_path: Path,
    ) -> None:
        """Reject encrypted PDFs that require a password."""

        if not reader.is_encrypted:
            return

        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise EncryptedDocumentError(
                f"PDF is encrypted and requires a password: "
                f"'{file_path}'."
            ) from exc

        if decrypt_result == 0:
            raise EncryptedDocumentError(
                f"PDF is encrypted and requires a password: "
                f"'{file_path}'."
            )


def load_pdf(file_path: PathLike) -> PDFDocument:
    """Convenience function that loads one local PDF document."""

    return PDFDocumentLoader().load(file_path)