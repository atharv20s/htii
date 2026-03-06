"""
PDF Parser — Extracts text and tables from digital PDF files.
Uses pdfplumber for high-quality extraction of Indian financial documents.
"""
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> Dict:
    """
    Parse a digital PDF file and extract text + tables from each page.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        Dict with keys:
            - "full_text": concatenated text from all pages
            - "pages": list of per-page dicts with "page_num", "text", "tables"
            - "metadata": file info
            - "is_scanned": True if the PDF appears to be scanned (no text layer)
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber is required. Install: pip install pdfplumber")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    result = {
        "full_text": "",
        "pages": [],
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
        },
        "is_scanned": False,
    }

    pages_with_no_text = 0

    try:
        with pdfplumber.open(file_path) as pdf:
            result["metadata"]["total_pages"] = len(pdf.pages)

            for i, page in enumerate(pdf.pages):
                page_data = {
                    "page_num": i + 1,
                    "text": "",
                    "tables": [],
                }

                # --- Extract text ---
                text = page.extract_text()
                if text and text.strip():
                    page_data["text"] = text.strip()
                else:
                    pages_with_no_text += 1

                # --- Extract tables ---
                try:
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Clean up table: replace None with empty string
                            cleaned_table = []
                            for row in table:
                                cleaned_row = [
                                    cell.strip() if cell else ""
                                    for cell in row
                                ]
                                cleaned_table.append(cleaned_row)
                            page_data["tables"].append(cleaned_table)
                except Exception as e:
                    logger.warning(f"Table extraction failed on page {i+1}: {e}")

                result["pages"].append(page_data)

            # Combine all text
            result["full_text"] = "\n\n".join(
                p["text"] for p in result["pages"] if p["text"]
            )

            # Detect if PDF is scanned (majority of pages have no text)
            total_pages = len(pdf.pages)
            if total_pages > 0 and (pages_with_no_text / total_pages) > 0.5:
                result["is_scanned"] = True
                logger.info(
                    f"PDF appears to be scanned ({pages_with_no_text}/{total_pages} "
                    f"pages have no text layer). Use OCR parser instead."
                )

    except Exception as e:
        logger.error(f"Failed to parse PDF '{file_path}': {e}")
        raise

    logger.info(
        f"Parsed PDF: {result['metadata']['file_name']} — "
        f"{result['metadata']['total_pages']} pages, "
        f"{len(result['full_text'])} characters extracted"
    )

    return result


def tables_to_text(tables: List[List[List[str]]]) -> str:
    """
    Convert extracted tables to a readable text format.
    Useful for feeding into LLM for analysis.
    """
    text_parts = []
    for t_idx, table in enumerate(tables):
        text_parts.append(f"\n--- Table {t_idx + 1} ---")
        for row in table:
            text_parts.append(" | ".join(row))
    return "\n".join(text_parts)
