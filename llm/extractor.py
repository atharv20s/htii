"""
LLM Extractor — Uses Ollama to extract structured financial data from raw text.
Handles document classification and data extraction in Indian financial context.
"""
import json
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def get_ollama_client():
    """Get an Ollama client instance."""
    try:
        import ollama
        return ollama
    except ImportError:
        raise ImportError("ollama is required. Install: pip install ollama")


def classify_document(text: str, model: str = "llama3") -> Dict:
    """
    Use LLM to classify the type of financial document.

    Args:
        text: First ~2000 characters of the document.
        model: Ollama model name.

    Returns:
        Dict with "document_type", "confidence", "reasoning".
    """
    ollama = get_ollama_client()
    snippet = text[:2000]

    prompt = f"""You are a financial document classifier for Indian corporate credit analysis.

Classify this document into ONE of these categories:
- GST_RETURN (GSTR-1, GSTR-3B, GSTR-2A filings)
- ITR (Income Tax Return)
- BANK_STATEMENT (bank transaction records)
- ANNUAL_REPORT (company annual report)
- FINANCIAL_STATEMENT (balance sheet, P&L statement, cash flow statement)
- LEGAL_NOTICE (litigation documents, court orders, legal notices)
- SANCTION_LETTER (loan sanction letters from banks)
- BOARD_MINUTES (board meeting minutes)
- RATING_REPORT (credit rating agency reports - CRISIL, ICRA, CARE, etc.)
- SHAREHOLDING (shareholding pattern)
- QUALITATIVE_NOTE (qualitative observations from credit officers)
- OTHER (if none of the above)

Document text (first 2000 chars):
\"\"\"
{snippet}
\"\"\"

Respond ONLY with valid JSON in this exact format:
{{"document_type": "...", "confidence": 0.0, "reasoning": "..."}}
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        result = json.loads(response["message"]["content"])
        logger.info(f"Document classified as: {result.get('document_type', 'UNKNOWN')}")
        return result
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}. Defaulting to OTHER.")
        return {
            "document_type": "OTHER",
            "confidence": 0.0,
            "reasoning": f"Classification failed: {str(e)}",
        }


def extract_financial_data(text: str, document_type: str, model: str = "llama3") -> Dict:
    """
    Use LLM to extract structured financial data from document text.
    The extraction prompt is customized based on document type.

    Args:
        text: Full document text (will be truncated to ~6000 chars for LLM).
        document_type: Type of document (from classify_document).
        model: Ollama model name.

    Returns:
        Dict with extracted structured data.
    """
    ollama = get_ollama_client()

    # Truncate to avoid token limits
    truncated_text = text[:6000]

    # Get type-specific extraction prompt
    extraction_prompt = _get_extraction_prompt(document_type, truncated_text)

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": extraction_prompt}],
            format="json",
        )
        result = json.loads(response["message"]["content"])
        logger.info(f"Extracted {len(result)} fields from {document_type}")
        return result
    except Exception as e:
        logger.error(f"LLM extraction failed for {document_type}: {e}")
        return {"extraction_error": str(e), "raw_text_preview": text[:500]}


def _get_extraction_prompt(document_type: str, text: str) -> str:
    """Generate document-type-specific extraction prompts."""

    base_instruction = (
        "You are a financial data extraction expert specializing in Indian corporate documents. "
        "Extract ALL relevant financial information from the following document text. "
        "Use INR (Indian Rupees) for all monetary values. "
        "Return ONLY valid JSON.\n\n"
    )

    prompts = {
        "GST_RETURN": f"""{base_instruction}
Extract GST return data from this document:
\"\"\"{text}\"\"\"

Return JSON with these fields (use null if not found):
{{
    "gstin": "GSTIN number",
    "legal_name": "Business name",
    "tax_period": "Month/Quarter and Year",
    "return_type": "GSTR-1 / GSTR-3B / GSTR-2A",
    "total_taxable_value": 0.0,
    "total_igst": 0.0,
    "total_cgst": 0.0,
    "total_sgst": 0.0,
    "total_cess": 0.0,
    "total_tax_liability": 0.0,
    "itc_claimed": 0.0,
    "itc_eligible": 0.0,
    "net_tax_payable": 0.0,
    "filing_date": "date if available",
    "key_observations": ["list of notable observations"]
}}""",

        "ITR": f"""{base_instruction}
Extract Income Tax Return data from this document:
\"\"\"{text}\"\"\"

Return JSON with these fields (use null if not found):
{{
    "pan": "PAN number",
    "assessment_year": "e.g. 2024-25",
    "name": "Assessee name",
    "itr_form": "ITR-1/2/3/4/5/6",
    "total_income": 0.0,
    "gross_total_income": 0.0,
    "total_deductions": 0.0,
    "tax_payable": 0.0,
    "business_income": 0.0,
    "capital_gains": 0.0,
    "other_sources_income": 0.0,
    "depreciation": 0.0,
    "brought_forward_losses": 0.0,
    "key_observations": ["list of notable items"]
}}""",

        "BANK_STATEMENT": f"""{base_instruction}
Extract bank statement data from this document:
\"\"\"{text}\"\"\"

Return JSON with these fields (use null if not found):
{{
    "account_holder": "Name",
    "account_number": "Account number (mask middle digits)",
    "bank_name": "Bank name",
    "statement_period": "From - To dates",
    "opening_balance": 0.0,
    "closing_balance": 0.0,
    "total_credits": 0.0,
    "total_debits": 0.0,
    "average_monthly_balance": 0.0,
    "number_of_transactions": 0,
    "largest_credit": 0.0,
    "largest_debit": 0.0,
    "bounce_count": 0,
    "emi_payments_detected": [],
    "key_observations": ["patterns, large transactions, irregularities"]
}}""",

        "ANNUAL_REPORT": f"""{base_instruction}
Extract key data from this Annual Report:
\"\"\"{text}\"\"\"

Return JSON with these fields (use null if not found):
{{
    "company_name": "Name",
    "cin": "Corporate Identification Number",
    "financial_year": "e.g. 2023-24",
    "revenue": 0.0,
    "net_profit": 0.0,
    "ebitda": 0.0,
    "total_assets": 0.0,
    "total_liabilities": 0.0,
    "net_worth": 0.0,
    "total_debt": 0.0,
    "equity_share_capital": 0.0,
    "reserves_and_surplus": 0.0,
    "promoter_holding_pct": 0.0,
    "dividend_declared": 0.0,
    "auditor_name": "Auditor firm name",
    "auditor_opinion": "Qualified / Unqualified / Adverse",
    "related_party_transactions": [],
    "contingent_liabilities": 0.0,
    "key_risks_mentioned": [],
    "key_observations": ["important highlights"]
}}""",

        "FINANCIAL_STATEMENT": f"""{base_instruction}
Extract financial statement data from this document:
\"\"\"{text}\"\"\"

Return JSON with these fields (use null if not found):
{{
    "statement_type": "Balance Sheet / P&L / Cash Flow",
    "company_name": "Name",
    "period": "Financial year or quarter",
    "revenue_from_operations": 0.0,
    "other_income": 0.0,
    "total_income": 0.0,
    "cost_of_materials": 0.0,
    "employee_benefit_expense": 0.0,
    "depreciation": 0.0,
    "finance_costs": 0.0,
    "other_expenses": 0.0,
    "profit_before_tax": 0.0,
    "tax_expense": 0.0,
    "profit_after_tax": 0.0,
    "current_assets": 0.0,
    "non_current_assets": 0.0,
    "current_liabilities": 0.0,
    "non_current_liabilities": 0.0,
    "shareholders_equity": 0.0,
    "key_observations": []
}}""",

        "LEGAL_NOTICE": f"""{base_instruction}
Extract legal/litigation details from this document:
\"\"\"{text}\"\"\"

Return JSON with:
{{
    "case_type": "Civil / Criminal / Tax / NCLT / Arbitration / Other",
    "parties_involved": [],
    "court_name": "Name of court or tribunal",
    "case_number": "Case/filing number",
    "date_filed": "Date if available",
    "current_status": "Pending / Disposed / Appeal",
    "amount_in_dispute": 0.0,
    "summary": "Brief description of dispute",
    "risk_assessment": "High / Medium / Low",
    "key_observations": []
}}""",

        "SANCTION_LETTER": f"""{base_instruction}
Extract loan sanction details from this document:
\"\"\"{text}\"\"\"

Return JSON with:
{{
    "lender_name": "Bank/NBFC name",
    "borrower_name": "Company name",
    "facility_type": "Term Loan / Working Capital / CC / OD etc.",
    "sanctioned_amount": 0.0,
    "interest_rate": "Rate or formula",
    "tenure_months": 0,
    "collateral_details": [],
    "key_covenants": [],
    "sanction_date": "Date",
    "key_observations": []
}}""",
    }

    # Default prompt for unrecognized types
    default_prompt = f"""{base_instruction}
Extract all relevant financial and business information from this document:
\"\"\"{text}\"\"\"

Return JSON with ALL relevant fields you can identify including:
- Company/entity names
- Financial figures (revenue, profit, debt, etc.)
- Dates and periods
- Key observations and risk factors
- Any compliance or regulatory information
"""

    return prompts.get(document_type, default_prompt)


def generate_document_summary(text: str, document_type: str, model: str = "llama3") -> str:
    """
    Generate a concise summary of the document for the credit analysis context.

    Returns:
        A 3-5 sentence summary string.
    """
    ollama = get_ollama_client()
    snippet = text[:4000]

    prompt = f"""Summarize this {document_type} document in 3-5 sentences for a 
credit analyst. Focus on financial risks, key metrics, and anything unusual.

Document:
\"\"\"{snippet}\"\"\"

Return plain text summary (not JSON).
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Summary unavailable. Document type: {document_type}."
