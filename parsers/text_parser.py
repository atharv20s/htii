"""
Text Parser — Handles plain text files and qualitative notes from Credit Officers.
"""
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def parse_text_file(file_path: str) -> Dict:
    """
    Parse a plain text file (.txt, .md).

    Returns:
        Dict with "full_text", "metadata".
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    result = {
        "full_text": "",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "format": "text",
        },
    }

    try:
        # Try multiple encodings (Indian documents may use different encodings)
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    result["full_text"] = f.read().strip()
                result["metadata"]["encoding"] = encoding
                break
            except UnicodeDecodeError:
                continue

        if not result["full_text"]:
            # Fallback: read as binary and decode with errors ignored
            with open(file_path, "rb") as f:
                result["full_text"] = f.read().decode("utf-8", errors="ignore").strip()
            result["metadata"]["encoding"] = "utf-8 (lossy)"

    except Exception as e:
        logger.error(f"Text file parsing failed for '{file_path}': {e}")
        raise

    logger.info(
        f"Parsed text: {result['metadata']['file_name']} — "
        f"{len(result['full_text'])} characters"
    )
    return result


def parse_qualitative_notes(notes: str, officer_name: str = "Unknown") -> Dict:
    """
    Structure qualitative notes from a Credit Officer into the ingestion format.
    These are free-text observations from factory visits, management interviews, etc.

    Args:
        notes: Raw text notes from the credit officer.
        officer_name: Name of the credit officer.

    Returns:
        Dict with "full_text", "metadata", "note_type".
    """
    result = {
        "full_text": (
            f"Credit Officer Qualitative Assessment\n"
            f"Officer: {officer_name}\n"
            f"---\n"
            f"{notes.strip()}"
        ),
        "metadata": {
            "file_name": f"qualitative_notes_{officer_name.replace(' ', '_').lower()}",
            "format": "qualitative_note",
            "officer_name": officer_name,
            "source": "primary_due_diligence",
        },
        "note_type": _classify_note(notes),
    }

    logger.info(
        f"Parsed qualitative note from {officer_name} — "
        f"Type: {result['note_type']}"
    )
    return result


def _classify_note(notes: str) -> str:
    """
    Classify the qualitative note type based on content keywords.
    """
    notes_lower = notes.lower()

    if any(kw in notes_lower for kw in ["factory", "plant", "site visit", "capacity", "machinery"]):
        return "FACTORY_VISIT"
    elif any(kw in notes_lower for kw in ["management", "interview", "promoter", "director", "meeting"]):
        return "MANAGEMENT_INTERVIEW"
    elif any(kw in notes_lower for kw in ["market", "competition", "sector", "industry"]):
        return "MARKET_OBSERVATION"
    elif any(kw in notes_lower for kw in ["risk", "concern", "warning", "red flag", "issue"]):
        return "RISK_OBSERVATION"
    else:
        return "GENERAL_OBSERVATION"
