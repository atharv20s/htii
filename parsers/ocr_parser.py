"""
OCR Parser — Extracts text from scanned PDFs and image files.
Uses pytesseract + Pillow for optical character recognition.
"""
import os
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def parse_image_ocr(file_path: str) -> Dict:
    """
    Extract text from an image file using OCR.

    Args:
        file_path: Path to image file (PNG, JPG, TIFF, BMP).

    Returns:
        Dict with "full_text", "metadata".
    """
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageEnhance
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required. "
            "Install: pip install pytesseract Pillow"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    result = {
        "full_text": "",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "source": "ocr",
        },
    }

    try:
        image = Image.open(file_path)

        # --- Preprocessing for better OCR accuracy ---
        # Convert to grayscale
        image = image.convert("L")
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        # --- Run OCR ---
        # Use English + Hindi (common in Indian financial docs)
        text = pytesseract.image_to_string(image, lang="eng")

        result["full_text"] = text.strip()
        result["metadata"]["image_size"] = f"{image.size[0]}x{image.size[1]}"

    except Exception as e:
        logger.error(f"OCR failed for '{file_path}': {e}")
        raise

    logger.info(
        f"OCR parsed: {result['metadata']['file_name']} — "
        f"{len(result['full_text'])} characters extracted"
    )
    return result


def parse_scanned_pdf(file_path: str) -> Dict:
    """
    Extract text from a scanned PDF by converting each page to an image
    and running OCR.

    Args:
        file_path: Path to the scanned PDF.

    Returns:
        Dict with "full_text", "pages", "metadata".
    """
    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image, ImageFilter, ImageEnhance
    except ImportError:
        raise ImportError(
            "PyMuPDF, pytesseract, and Pillow are required. "
            "Install: pip install PyMuPDF pytesseract Pillow"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    result = {
        "full_text": "",
        "pages": [],
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "source": "ocr",
        },
    }

    try:
        doc = fitz.open(file_path)
        result["metadata"]["total_pages"] = len(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to image at 300 DPI for good OCR quality
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")

            # Convert to PIL Image
            import io
            image = Image.open(io.BytesIO(img_data))

            # Preprocess
            image = image.convert("L")
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            image = image.filter(ImageFilter.SHARPEN)

            # OCR
            text = pytesseract.image_to_string(image, lang="eng")

            page_data = {
                "page_num": page_num + 1,
                "text": text.strip(),
            }
            result["pages"].append(page_data)

        doc.close()

        # Combine all page texts
        result["full_text"] = "\n\n".join(
            p["text"] for p in result["pages"] if p["text"]
        )

    except Exception as e:
        logger.error(f"Scanned PDF parsing failed for '{file_path}': {e}")
        raise

    logger.info(
        f"OCR parsed scanned PDF: {result['metadata']['file_name']} — "
        f"{result['metadata']['total_pages']} pages, "
        f"{len(result['full_text'])} characters extracted"
    )
    return result
