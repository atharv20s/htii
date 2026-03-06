"""
Structured Data Parser — Handles CSV, Excel, and JSON files.
Converts structured financial data (GST, ITR, Bank Statements) to text + DataFrames.
"""
import os
import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def parse_csv(file_path: str) -> Dict:
    """
    Parse a CSV file into a DataFrame and generate a text summary.

    Returns:
        Dict with "dataframe", "full_text", "summary", "metadata".
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install: pip install pandas")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    result = {
        "dataframe": None,
        "full_text": "",
        "summary": "",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "format": "csv",
        },
    }

    try:
        df = pd.read_csv(file_path)
        result["dataframe"] = df
        result["metadata"]["rows"] = len(df)
        result["metadata"]["columns"] = list(df.columns)

        # Generate text representation for LLM consumption
        result["full_text"] = _dataframe_to_text(df, os.path.basename(file_path))

        # Generate quick summary
        result["summary"] = (
            f"CSV file with {len(df)} rows and {len(df.columns)} columns. "
            f"Columns: {', '.join(df.columns[:15])}"
        )

    except Exception as e:
        logger.error(f"CSV parsing failed for '{file_path}': {e}")
        raise

    logger.info(f"Parsed CSV: {result['metadata']['file_name']} — {len(df)} rows")
    return result


def parse_excel(file_path: str) -> Dict:
    """
    Parse an Excel file (all sheets) into DataFrames and text.

    Returns:
        Dict with "sheets" (dict of sheet_name: DataFrame), "full_text", "metadata".
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas and openpyxl are required. "
            "Install: pip install pandas openpyxl"
        )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    result = {
        "sheets": {},
        "full_text": "",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "format": "excel",
        },
    }

    try:
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        result["metadata"]["sheet_names"] = sheet_names

        text_parts = []
        for sheet_name in sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            result["sheets"][sheet_name] = df

            text_parts.append(f"\n=== Sheet: {sheet_name} ===")
            text_parts.append(
                _dataframe_to_text(df, f"{os.path.basename(file_path)} - {sheet_name}")
            )

        result["full_text"] = "\n".join(text_parts)

    except Exception as e:
        logger.error(f"Excel parsing failed for '{file_path}': {e}")
        raise

    logger.info(
        f"Parsed Excel: {result['metadata']['file_name']} — "
        f"{len(sheet_names)} sheets"
    )
    return result


def parse_json(file_path: str) -> Dict:
    """
    Parse a JSON file into a Python dict and generate text representation.

    Returns:
        Dict with "data", "full_text", "metadata".
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    result = {
        "data": None,
        "full_text": "",
        "metadata": {
            "file_name": os.path.basename(file_path),
            "file_path": file_path,
            "file_size_kb": round(os.path.getsize(file_path) / 1024, 2),
            "format": "json",
        },
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result["data"] = data
        # Pretty-print JSON as text for LLM consumption
        result["full_text"] = (
            f"JSON Data from {os.path.basename(file_path)}:\n"
            f"{json.dumps(data, indent=2, ensure_ascii=False)}"
        )

    except Exception as e:
        logger.error(f"JSON parsing failed for '{file_path}': {e}")
        raise

    logger.info(f"Parsed JSON: {result['metadata']['file_name']}")
    return result


def _dataframe_to_text(df, source_name: str) -> str:
    """
    Convert a pandas DataFrame into LLM-friendly text format.
    Includes column names, data types, summary stats, and first rows.
    """
    import pandas as pd

    lines = [f"\nData from: {source_name}"]
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    lines.append(f"Columns: {', '.join(df.columns)}")

    # Data types
    lines.append("\nColumn Types:")
    for col in df.columns:
        lines.append(f"  - {col}: {df[col].dtype}")

    # Show first 20 rows as a table
    lines.append("\nData (first 20 rows):")
    lines.append(df.head(20).to_string(index=False))

    # Numeric summary
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        lines.append("\nNumeric Summary:")
        lines.append(df[numeric_cols].describe().to_string())

    # Detect possible financial columns (Indian context)
    financial_keywords = [
        "amount", "revenue", "sales", "tax", "gst", "igst", "cgst", "sgst",
        "credit", "debit", "balance", "turnover", "profit", "loss",
        "interest", "total", "value", "invoice", "cess",
    ]
    financial_cols = [
        col for col in df.columns
        if any(kw in col.lower() for kw in financial_keywords)
    ]
    if financial_cols:
        lines.append(f"\nDetected financial columns: {', '.join(financial_cols)}")
        for col in financial_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                lines.append(
                    f"  {col}: Total = {df[col].sum():,.2f}, "
                    f"Mean = {df[col].mean():,.2f}"
                )

    return "\n".join(lines)
