"""
Document Extraction Layer.
Extracts structured text from PDF, DOCX, and TXT files while strictly preserving
real page numbers for PDFs and setting page=None for non-paged documents.
"""

import io
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document


def clean_text(text: str) -> str:
    """
    Lightweight text cleaning for academic documents.
    Normalizes excessive whitespace and redundant blank lines without
    destroying sentences, code blocks, or academic headings.
    """
    if not text:
        return ""
    # Replace non-breaking spaces and carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    # Replace sequences of 3 or more newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace excessive horizontal whitespace with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Strip leading and trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_pdf(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF document.
    Preserves 1-indexed page numbers. Never invents page numbers.
    """
    extracted_pages = []
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        if num_pages == 0:
            return []

        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1  # 1-indexed
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            
            cleaned = clean_text(raw_text)
            if cleaned:
                extracted_pages.append({
                    "filename": filename,
                    "page": page_num,
                    "text": cleaned
                })
    except Exception as e:
        raise ValueError(f"Failed to read PDF '{filename}': {str(e)}")

    return extracted_pages


def extract_docx(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a Word DOCX document.
    Preserves paragraph structure. Sets page to None (DOCX lacks fixed physical pages).
    """
    try:
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs.append(text)
        
        # Also extract table text if present
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        full_text = "\n\n".join(paragraphs)
        cleaned = clean_text(full_text)
        
        if not cleaned:
            return []

        return [{
            "filename": filename,
            "page": None,
            "text": cleaned
        }]
    except Exception as e:
        raise ValueError(f"Failed to read DOCX '{filename}': {str(e)}")


def extract_txt(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a plain text (TXT) document.
    Tries utf-8 first, with fallback to latin-1. Sets page to None.
    """
    try:
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")

        cleaned = clean_text(content)
        if not cleaned:
            return []

        return [{
            "filename": filename,
            "page": None,
            "text": cleaned
        }]
    except Exception as e:
        raise ValueError(f"Failed to read TXT '{filename}': {str(e)}")


def extract_document(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Routes an uploaded file to the appropriate extraction function based on its extension.
    Returns a list of structured records:
    [
        {"filename": "...", "page": 1, "text": "..."},
        ...
    ]
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return extract_pdf(file_bytes, filename)
    elif lower_name.endswith(".docx"):
        return extract_docx(file_bytes, filename)
    elif lower_name.endswith(".txt"):
        return extract_txt(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type for '{filename}'. Only PDF, DOCX, and TXT are supported.")
