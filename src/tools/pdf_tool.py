"""
PDF utility tool for the Finance Agentic AI System.

This module reads PDF files and extracts text from them.
It will mainly be used for budget assumptions, FP&A policies,
forecast rules, and RAG documents.
"""

from pathlib import Path

from pypdf import PdfReader


class PDFTool:
    """
    Tool for handling PDF file operations.

    This class validates PDF file paths and extracts readable text
    from PDF documents.
    """

    @staticmethod
    def validate_file_path(file_path: str | Path) -> Path:
        """
        Validate whether the given PDF file path exists.

        Args:
            file_path: Path of the PDF file.

        Returns:
            Path: Validated PDF file path.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File must be a PDF file: {path}")

        return path

    @staticmethod
    def extract_text(file_path: str | Path) -> str:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path of the PDF file.

        Returns:
            str: Extracted text from the PDF.

        Raises:
            ValueError: If no readable text is found in the PDF.
        """
        path = PDFTool.validate_file_path(file_path)
        reader = PdfReader(path)

        extracted_text = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text.strip())

        full_text = "\n\n".join(extracted_text)

        if not full_text.strip():
            raise ValueError(f"No readable text found in PDF: {path}")

        return full_text