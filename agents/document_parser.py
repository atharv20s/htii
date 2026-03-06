"""
Document Parser Agent — The main ingestion orchestrator (Agent 1).
Takes a file, detects its type, routes to the correct parser,
extracts structured data via LLM, and stores embeddings in ChromaDB.
"""
import os
import sys
import logging
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SUPPORTED_EXTENSIONS,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
)

logger = logging.getLogger(__name__)


class DocumentParserAgent:
    """
    Agent 1: Document Parser & Ingestion Agent.

    Pipeline for each file:
        detect_type → parse → classify (LLM) → extract (LLM) → chunk → embed → store
    """

    def __init__(
        self,
        ollama_model: str = None,
        embedding_model: str = None,
        collection_name: str = None,
        use_llm: bool = True,
    ):
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        self.collection_name = collection_name or CHROMA_COLLECTION_NAME
        self.use_llm = use_llm  # Can disable LLM for testing without Ollama

        self.ingestion_log = []  # Track all processed files

    def ingest_file(self, file_path: str) -> Dict:
        """
        Main entry point: ingest a single file through the full pipeline.

        Args:
            file_path: Absolute path to the file.

        Returns:
            Dict with full ingestion results and status.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"INGESTING: {file_path}")
        logger.info(f"{'='*60}")

        result = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "status": "pending",
            "steps": {},
        }

        try:
            # ── Step 1: Detect file type ──
            file_type = self._detect_file_type(file_path)
            result["file_type"] = file_type
            result["steps"]["detect_type"] = {"status": "success", "type": file_type}
            logger.info(f"  [1/5] File type detected: {file_type}")

            # ── Step 2: Parse the file ──
            parsed = self._parse_file(file_path, file_type)
            full_text = parsed.get("full_text", "")
            result["steps"]["parse"] = {
                "status": "success",
                "text_length": len(full_text),
            }
            logger.info(f"  [2/5] Parsed: {len(full_text)} characters extracted")

            if not full_text.strip():
                result["status"] = "warning"
                result["message"] = "No text could be extracted from the file."
                logger.warning("  No text extracted — skipping further processing.")
                self.ingestion_log.append(result)
                return result

            # ── Step 3: Classify document type via LLM ──
            if self.use_llm:
                classification = self._classify_document(full_text)
            else:
                classification = {
                    "document_type": self._guess_type_from_extension(file_path),
                    "confidence": 0.5,
                    "reasoning": "Guessed from file extension (LLM disabled)",
                }
            document_type = classification.get("document_type", "OTHER")
            result["document_type"] = document_type
            result["steps"]["classify"] = {"status": "success", **classification}
            logger.info(f"  [3/5] Classified as: {document_type}")

            # ── Step 4: Extract structured data via LLM ──
            if self.use_llm:
                extracted_data = self._extract_data(full_text, document_type)
            else:
                extracted_data = {"note": "LLM extraction disabled"}
            result["extracted_data"] = extracted_data
            result["steps"]["extract"] = {"status": "success", "fields": len(extracted_data)}
            logger.info(f"  [4/5] Extracted {len(extracted_data)} fields")

            # ── Step 5: Chunk, embed, and store in ChromaDB ──
            storage_metadata = {
                "file_name": os.path.basename(file_path),
                "document_type": document_type,
                "source": "upload",
                "classification_confidence": classification.get("confidence", 0),
            }
            storage_result = self._store_in_vectordb(full_text, storage_metadata)
            result["steps"]["store"] = storage_result
            result["document_id"] = storage_result.get("document_id")
            logger.info(
                f"  [5/5] Stored {storage_result.get('chunk_count', 0)} chunks "
                f"in ChromaDB (ID: {storage_result.get('document_id', 'N/A')})"
            )

            result["status"] = "success"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.error(f"  INGESTION FAILED: {e}")

        self.ingestion_log.append(result)
        return result

    def ingest_directory(self, directory_path: str) -> List[Dict]:
        """
        Ingest all supported files from a directory.

        Args:
            directory_path: Path to directory containing files.

        Returns:
            List of ingestion results for each file.
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        results = []
        all_extensions = []
        for exts in SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(exts)

        for file_name in sorted(os.listdir(directory_path)):
            file_path = os.path.join(directory_path, file_name)

            if not os.path.isfile(file_path):
                continue

            ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
            if ext not in all_extensions:
                logger.info(f"Skipping unsupported file: {file_name}")
                continue

            result = self.ingest_file(file_path)
            results.append(result)

        logger.info(
            f"\nDirectory ingestion complete: {len(results)} files processed, "
            f"{sum(1 for r in results if r['status'] == 'success')} successful"
        )
        return results

    def ingest_qualitative_notes(self, notes: str, officer_name: str = "Unknown") -> Dict:
        """
        Ingest qualitative notes from a Credit Officer.

        Args:
            notes: Free-text observations.
            officer_name: Name of the credit officer.

        Returns:
            Ingestion result dict.
        """
        from parsers.text_parser import parse_qualitative_notes

        parsed = parse_qualitative_notes(notes, officer_name)

        metadata = {
            "file_name": parsed["metadata"]["file_name"],
            "document_type": "QUALITATIVE_NOTE",
            "source": "primary_due_diligence",
            "officer_name": officer_name,
            "note_type": parsed.get("note_type", "GENERAL"),
        }

        storage_result = self._store_in_vectordb(parsed["full_text"], metadata)

        result = {
            "file_name": metadata["file_name"],
            "document_type": "QUALITATIVE_NOTE",
            "note_type": parsed.get("note_type"),
            "status": "success",
            "storage": storage_result,
        }

        self.ingestion_log.append(result)
        logger.info(f"Ingested qualitative note from {officer_name}")
        return result

    def get_ingestion_summary(self) -> Dict:
        """Get a summary of all ingested documents."""
        from vector_store.chroma_store import get_collection_stats

        total = len(self.ingestion_log)
        success = sum(1 for r in self.ingestion_log if r.get("status") == "success")
        errors = sum(1 for r in self.ingestion_log if r.get("status") == "error")

        stats = get_collection_stats(self.collection_name, CHROMA_DB_DIR)

        return {
            "files_processed": total,
            "successful": success,
            "errors": errors,
            "vector_store": stats,
            "files": [
                {
                    "file": r.get("file_name", "N/A"),
                    "type": r.get("document_type", "N/A"),
                    "status": r.get("status", "N/A"),
                }
                for r in self.ingestion_log
            ],
        }

    # ────────────────────────────────────────────────────────
    # Private helper methods
    # ────────────────────────────────────────────────────────

    def _detect_file_type(self, file_path: str) -> str:
        """Detect the file type category from extension."""
        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

        for category, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return category

        return "unknown"

    def _parse_file(self, file_path: str, file_type: str) -> Dict:
        """Route to the correct parser based on file type."""

        if file_type == "pdf":
            from parsers.pdf_parser import parse_pdf
            result = parse_pdf(file_path)

            # If scanned, fall back to OCR
            if result.get("is_scanned", False):
                logger.info("  PDF is scanned — switching to OCR parser")
                from parsers.ocr_parser import parse_scanned_pdf
                result = parse_scanned_pdf(file_path)
            return result

        elif file_type == "image":
            from parsers.ocr_parser import parse_image_ocr
            return parse_image_ocr(file_path)

        elif file_type == "structured":
            ext = file_path.rsplit(".", 1)[-1].lower()
            if ext == "csv":
                from parsers.structured_parser import parse_csv
                return parse_csv(file_path)
            elif ext in ["xlsx", "xls"]:
                from parsers.structured_parser import parse_excel
                return parse_excel(file_path)
            elif ext == "json":
                from parsers.structured_parser import parse_json
                return parse_json(file_path)

        elif file_type == "text":
            from parsers.text_parser import parse_text_file
            return parse_text_file(file_path)

        # Fallback: try reading as text
        from parsers.text_parser import parse_text_file
        return parse_text_file(file_path)

    def _classify_document(self, text: str) -> Dict:
        """Classify document type using LLM."""
        from llm.extractor import classify_document
        return classify_document(text, self.ollama_model)

    def _extract_data(self, text: str, document_type: str) -> Dict:
        """Extract structured data using LLM."""
        from llm.extractor import extract_financial_data
        return extract_financial_data(text, document_type, self.ollama_model)

    def _store_in_vectordb(self, text: str, metadata: Dict) -> Dict:
        """Chunk, embed, and store in ChromaDB."""
        from vector_store.chroma_store import store_document
        return store_document(
            text=text,
            metadata=metadata,
            collection_name=self.collection_name,
            persist_dir=CHROMA_DB_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            embedding_model=self.embedding_model,
        )

    def _guess_type_from_extension(self, file_path: str) -> str:
        """Fallback document type guess when LLM is disabled."""
        name = os.path.basename(file_path).lower()
        if "gst" in name or "gstr" in name:
            return "GST_RETURN"
        elif "itr" in name or "tax" in name:
            return "ITR"
        elif "bank" in name or "statement" in name:
            return "BANK_STATEMENT"
        elif "annual" in name or "report" in name:
            return "ANNUAL_REPORT"
        elif "legal" in name or "notice" in name or "litigation" in name:
            return "LEGAL_NOTICE"
        elif "sanction" in name:
            return "SANCTION_LETTER"
        else:
            return "OTHER"
