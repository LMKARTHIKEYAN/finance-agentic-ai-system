"""
Tests for the local PDF document loader.

These tests cover:

* PDFPage validation
* PDFDocument validation
* Local path validation
* Page-by-page extraction
* Empty-page detection
* Invalid and encrypted PDFs
* Convenience-function behaviour

The tests intentionally do not cover chunking, embeddings, vector stores,
LLMs, OCR, or finance calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfReader, PdfWriter

from src.rag.document_loader import (
    DocumentFileNotFoundError,
    EncryptedDocumentError,
    InvalidDocumentError,
    PDFDocument,
    PDFDocumentLoader,
    PDFPage,
    load_pdf,
)


class FakePage:
    """Simple test double for a pypdf page."""

    def __init__(
        self,
        text: str | None = "",
        extraction_error: Exception | None = None,
    ) -> None:
        self._text = text
        self._extraction_error = extraction_error

    def extract_text(self) -> str | None:
        """Return configured text or raise the configured error."""

        if self._extraction_error is not None:
            raise self._extraction_error

        return self._text


class FakeReader:
    """Simple test double for PdfReader."""

    def __init__(
        self,
        pages: list[FakePage],
        *,
        is_encrypted: bool = False,
        decrypt_result: int = 1,
        decrypt_error: Exception | None = None,
    ) -> None:
        self.pages = pages
        self.is_encrypted = is_encrypted
        self._decrypt_result = decrypt_result
        self._decrypt_error = decrypt_error

    def decrypt(self, password: str) -> int:
        """Return configured decryption result."""

        if self._decrypt_error is not None:
            raise self._decrypt_error

        return self._decrypt_result


def create_blank_pdf(
    file_path: Path,
    *,
    page_count: int = 1,
    encrypted_password: str | None = None,
) -> Path:
    """
    Create a temporary PDF using pypdf.

    Blank pages are sufficient for testing file handling, page counts,
    empty-page detection, and encryption behaviour.
    """

    writer = PdfWriter()

    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)

    if encrypted_password is not None:
        writer.encrypt(encrypted_password)

    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    return file_path


def create_valid_page(
    *,
    page_number: int = 1,
    text: str = "Budget assumptions",
    source_filename: str = "budget_assumptions.pdf",
) -> PDFPage:
    """Create a valid PDFPage instance for document tests."""

    return PDFPage(
        page_number=page_number,
        text=text,
        source_filename=source_filename,
        is_empty=not bool(text.strip()),
    )


def test_pdf_page_creates_valid_non_empty_page() -> None:
    """PDFPage should store valid page-level metadata."""

    page = PDFPage(
        page_number=1,
        text="  Forecast methodology  ",
        source_filename="forecast_methodology.pdf",
        is_empty=False,
    )

    assert page.page_number == 1
    assert page.text == "Forecast methodology"
    assert page.source_filename == "forecast_methodology.pdf"
    assert page.is_empty is False


def test_pdf_page_creates_valid_empty_page() -> None:
    """Whitespace-only page text should be normalized to an empty string."""

    page = PDFPage(
        page_number=2,
        text="   \n\t ",
        source_filename="finance_policy.pdf",
        is_empty=True,
    )

    assert page.page_number == 2
    assert page.text == ""
    assert page.is_empty is True


@pytest.mark.parametrize(
    "page_number",
    [0, -1],
)
def test_pdf_page_rejects_non_positive_page_number(
    page_number: int,
) -> None:
    """Page numbers must be one-based positive integers."""

    with pytest.raises(
        ValueError,
        match="page_number must be greater than zero",
    ):
        PDFPage(
            page_number=page_number,
            text="Text",
            source_filename="document.pdf",
            is_empty=False,
        )


@pytest.mark.parametrize(
    "page_number",
    [True, 1.5, "1", None],
)
def test_pdf_page_rejects_invalid_page_number_type(
    page_number: Any,
) -> None:
    """Page numbers must be integers and must not accept booleans."""

    with pytest.raises(
        TypeError,
        match="page_number must be an integer",
    ):
        PDFPage(
            page_number=page_number,
            text="Text",
            source_filename="document.pdf",
            is_empty=False,
        )


def test_pdf_page_rejects_empty_source_filename() -> None:
    """A page must retain a non-empty source filename."""

    with pytest.raises(
        ValueError,
        match="source_filename cannot be empty",
    ):
        PDFPage(
            page_number=1,
            text="Text",
            source_filename="   ",
            is_empty=False,
        )


def test_pdf_page_rejects_incorrect_empty_flag() -> None:
    """is_empty must agree with the normalized page text."""

    with pytest.raises(
        ValueError,
        match="is_empty must match",
    ):
        PDFPage(
            page_number=1,
            text="Actual page text",
            source_filename="document.pdf",
            is_empty=True,
        )


def test_pdf_document_creates_valid_document(
    tmp_path: Path,
) -> None:
    """PDFDocument should preserve valid page-level results."""

    source_path = (tmp_path / "budget_assumptions.pdf").resolve()

    pages = (
        create_valid_page(page_number=1, text="Budget assumptions"),
        create_valid_page(page_number=2, text=""),
    )

    document = PDFDocument(
        source_path=source_path,
        source_filename="budget_assumptions.pdf",
        pages=pages,
        total_pages=2,
        empty_page_numbers=(2,),
    )

    assert document.source_path == source_path
    assert document.source_filename == "budget_assumptions.pdf"
    assert document.pages == pages
    assert document.total_pages == 2
    assert document.empty_page_numbers == (2,)
    assert document.has_empty_pages is True


def test_pdf_document_reports_no_empty_pages(
    tmp_path: Path,
) -> None:
    """has_empty_pages should be false when every page contains text."""

    source_path = (tmp_path / "document.pdf").resolve()

    document = PDFDocument(
        source_path=source_path,
        source_filename="document.pdf",
        pages=(
            create_valid_page(
                page_number=1,
                text="Page one",
                source_filename="document.pdf",
            ),
        ),
        total_pages=1,
        empty_page_numbers=(),
    )

    assert document.has_empty_pages is False


def test_pdf_document_rejects_total_pages_mismatch(
    tmp_path: Path,
) -> None:
    """total_pages must equal the number of stored pages."""

    source_path = (tmp_path / "document.pdf").resolve()

    with pytest.raises(
        ValueError,
        match="total_pages must match",
    ):
        PDFDocument(
            source_path=source_path,
            source_filename="document.pdf",
            pages=(
                create_valid_page(
                    source_filename="document.pdf",
                ),
            ),
            total_pages=2,
            empty_page_numbers=(),
        )


def test_pdf_document_rejects_non_sequential_page_numbers(
    tmp_path: Path,
) -> None:
    """Document pages must use sequential one-based page numbers."""

    source_path = (tmp_path / "document.pdf").resolve()

    with pytest.raises(
        ValueError,
        match="sequential one-based page numbers",
    ):
        PDFDocument(
            source_path=source_path,
            source_filename="document.pdf",
            pages=(
                create_valid_page(
                    page_number=2,
                    source_filename="document.pdf",
                ),
            ),
            total_pages=1,
            empty_page_numbers=(),
        )


def test_pdf_document_rejects_incorrect_empty_page_numbers(
    tmp_path: Path,
) -> None:
    """Empty-page metadata must match the stored page results."""

    source_path = (tmp_path / "document.pdf").resolve()

    pages = (
        create_valid_page(
            page_number=1,
            text="",
            source_filename="document.pdf",
        ),
    )

    with pytest.raises(
        ValueError,
        match="empty_page_numbers must match",
    ):
        PDFDocument(
            source_path=source_path,
            source_filename="document.pdf",
            pages=pages,
            total_pages=1,
            empty_page_numbers=(),
        )


def test_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """The loader should raise a focused error for a missing PDF."""

    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(
        DocumentFileNotFoundError,
        match="does not exist",
    ):
        PDFDocumentLoader().load(missing_path)


def test_loader_rejects_directory_path(
    tmp_path: Path,
) -> None:
    """A directory must not be accepted as a PDF file."""

    directory_path = tmp_path / "documents"
    directory_path.mkdir()

    with pytest.raises(
        InvalidDocumentError,
        match="not a file",
    ):
        PDFDocumentLoader().load(directory_path)


def test_loader_rejects_non_pdf_extension(
    tmp_path: Path,
) -> None:
    """Only paths ending in .pdf should be accepted."""

    text_file = tmp_path / "finance_policy.txt"
    text_file.write_text("Finance policy", encoding="utf-8")

    with pytest.raises(
        InvalidDocumentError,
        match=r"\.pdf extension",
    ):
        PDFDocumentLoader().load(text_file)


def test_loader_rejects_empty_string_path() -> None:
    """An empty path string should fail before filesystem access."""

    with pytest.raises(
        ValueError,
        match="file_path cannot be empty",
    ):
        PDFDocumentLoader().load("   ")


@pytest.mark.parametrize(
    "invalid_path",
    [None, 123, 1.5, object()],
)
def test_loader_rejects_invalid_path_type(
    invalid_path: Any,
) -> None:
    """The loader should accept only strings and pathlib.Path values."""

    with pytest.raises(
        TypeError,
        match="string or pathlib.Path",
    ):
        PDFDocumentLoader().load(invalid_path)


def test_loader_reads_blank_pdf_page_by_page(
    tmp_path: Path,
) -> None:
    """A real blank PDF should retain every empty page."""

    pdf_path = create_blank_pdf(
        tmp_path / "finance_policy.pdf",
        page_count=3,
    )

    document = PDFDocumentLoader().load(pdf_path)

    assert document.source_path == pdf_path.resolve()
    assert document.source_filename == "finance_policy.pdf"
    assert document.total_pages == 3
    assert document.empty_page_numbers == (1, 2, 3)
    assert document.has_empty_pages is True

    assert tuple(
        page.page_number for page in document.pages
    ) == (1, 2, 3)

    assert all(page.text == "" for page in document.pages)
    assert all(page.is_empty for page in document.pages)

    assert all(
        page.source_filename == "finance_policy.pdf"
        for page in document.pages
    )


def test_loader_extracts_text_and_preserves_page_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The loader should preserve text, page order, and filename."""

    pdf_path = create_blank_pdf(
        tmp_path / "forecast_methodology.pdf",
    )

    fake_reader = FakeReader(
        pages=[
            FakePage("  Revenue forecast methodology  "),
            FakePage(None),
            FakePage("\n Cost assumptions \n"),
        ],
    )

    monkeypatch.setattr(
        PDFDocumentLoader,
        "_create_reader",
        staticmethod(lambda _: fake_reader),
    )

    document = PDFDocumentLoader().load(pdf_path)

    assert document.source_path == pdf_path.resolve()
    assert document.source_filename == "forecast_methodology.pdf"
    assert document.total_pages == 3
    assert document.empty_page_numbers == (2,)
    assert document.has_empty_pages is True

    assert document.pages[0] == PDFPage(
        page_number=1,
        text="Revenue forecast methodology",
        source_filename="forecast_methodology.pdf",
        is_empty=False,
    )

    assert document.pages[1] == PDFPage(
        page_number=2,
        text="",
        source_filename="forecast_methodology.pdf",
        is_empty=True,
    )

    assert document.pages[2] == PDFPage(
        page_number=3,
        text="Cost assumptions",
        source_filename="forecast_methodology.pdf",
        is_empty=False,
    )


def test_loader_rejects_corrupted_pdf(
    tmp_path: Path,
) -> None:
    """A file with a PDF extension but invalid contents should fail."""

    corrupted_pdf = tmp_path / "corrupted.pdf"
    corrupted_pdf.write_bytes(b"This is not a valid PDF file.")

    with pytest.raises(
        InvalidDocumentError,
        match="Unable to read PDF",
    ):
        PDFDocumentLoader().load(corrupted_pdf)


def test_loader_rejects_password_protected_pdf(
    tmp_path: Path,
) -> None:
    """A PDF requiring a non-empty password should be rejected."""

    encrypted_pdf = create_blank_pdf(
        tmp_path / "protected.pdf",
        encrypted_password="secret-password",
    )

    with pytest.raises(
        EncryptedDocumentError,
        match="requires a password",
    ):
        PDFDocumentLoader().load(encrypted_pdf)


def test_loader_accepts_pdf_decryptable_with_empty_password(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An encrypted PDF decryptable with an empty password may be loaded."""

    pdf_path = create_blank_pdf(
        tmp_path / "empty_password.pdf",
    )

    fake_reader = FakeReader(
        pages=[FakePage("Finance policy")],
        is_encrypted=True,
        decrypt_result=1,
    )

    monkeypatch.setattr(
        PDFDocumentLoader,
        "_create_reader",
        staticmethod(lambda _: fake_reader),
    )

    document = PDFDocumentLoader().load(pdf_path)

    assert document.total_pages == 1
    assert document.pages[0].text == "Finance policy"
    assert document.has_empty_pages is False


def test_loader_wraps_page_extraction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected extraction failures should become InvalidDocumentError."""

    pdf_path = create_blank_pdf(
        tmp_path / "extraction_failure.pdf",
    )

    fake_reader = FakeReader(
        pages=[
            FakePage(
                extraction_error=RuntimeError(
                    "Unable to extract page text."
                )
            ),
        ],
    )

    monkeypatch.setattr(
        PDFDocumentLoader,
        "_create_reader",
        staticmethod(lambda _: fake_reader),
    )

    with pytest.raises(
        InvalidDocumentError,
        match="Failed to extract text",
    ):
        PDFDocumentLoader().load(pdf_path)


def test_load_pdf_matches_document_loader(
    tmp_path: Path,
) -> None:
    """The convenience function should match direct loader usage."""

    pdf_path = create_blank_pdf(
        tmp_path / "budget_assumptions.pdf",
        page_count=2,
    )

    direct_result = PDFDocumentLoader().load(pdf_path)
    convenience_result = load_pdf(pdf_path)

    assert convenience_result == direct_result


def test_created_pdf_is_readable_by_pypdf(
    tmp_path: Path,
) -> None:
    """The temporary-PDF helper should create a genuine PDF file."""

    pdf_path = create_blank_pdf(
        tmp_path / "valid.pdf",
        page_count=2,
    )

    reader = PdfReader(str(pdf_path))

    assert len(reader.pages) == 2