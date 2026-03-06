"""
Intelli-Credit: Data Ingestion Pipeline — Test Runner
=====================================================
This script demonstrates the full ingestion pipeline:
  1. Creates sample test files (CSV, JSON, text)
  2. Ingests them through the Document Parser Agent
  3. Stores embeddings in ChromaDB
  4. Queries the vector store to verify retrieval

Usage:
  python run_ingestion.py              # Run with LLM (requires Ollama running)
  python run_ingestion.py --no-llm     # Run without LLM (for testing parsers + vector DB)
"""
import os
import sys
import json
import logging
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME
from agents.document_parser import DocumentParserAgent
from vector_store.chroma_store import query_documents, get_collection_stats, delete_collection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_sample_files():
    """Create sample financial files for testing the pipeline."""

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    created_files = []

    # ── Sample 1: GST Return (CSV) ──
    gst_csv_path = os.path.join(UPLOAD_DIR, "gstr3b_sample.csv")
    gst_csv_content = """GSTIN,Legal_Name,Tax_Period,Return_Type,Total_Taxable_Value,IGST,CGST,SGST,Total_Tax,ITC_Claimed,Net_Tax_Payable
27AAACB1234F1Z5,ABC Industries Pvt Ltd,Jan-2025,GSTR-3B,15000000,1200000,900000,900000,3000000,2500000,500000
27AAACB1234F1Z5,ABC Industries Pvt Ltd,Feb-2025,GSTR-3B,16500000,1320000,990000,990000,3300000,2750000,550000
27AAACB1234F1Z5,ABC Industries Pvt Ltd,Mar-2025,GSTR-3B,18000000,1440000,1080000,1080000,3600000,3000000,600000
"""
    with open(gst_csv_path, "w") as f:
        f.write(gst_csv_content)
    created_files.append(gst_csv_path)

    # ── Sample 2: Bank Statement (CSV) ──
    bank_csv_path = os.path.join(UPLOAD_DIR, "bank_statement_sample.csv")
    bank_csv_content = """Date,Description,Credit,Debit,Balance
01-01-2025,Opening Balance,,,5000000
05-01-2025,Sales Receipt from XYZ Corp,3500000,,8500000
10-01-2025,Purchase Payment to Raw Materials Ltd,,2000000,6500000
15-01-2025,EMI Payment - HDFC Term Loan,,450000,6050000
20-01-2025,GST Refund Received,250000,,6300000
25-01-2025,Salary Payments,,1200000,5100000
28-01-2025,Sales Receipt from PQR Industries,4200000,,9300000
30-01-2025,Electricity and Utilities,,180000,9120000
31-01-2025,Cheque Bounce - Customer Payment,,500000,8620000
"""
    with open(bank_csv_path, "w") as f:
        f.write(bank_csv_content)
    created_files.append(bank_csv_path)

    # ── Sample 3: Annual Report Summary (JSON) ──
    annual_report_path = os.path.join(UPLOAD_DIR, "annual_report_summary.json")
    annual_report_data = {
        "company_name": "ABC Industries Private Limited",
        "cin": "U28920MH2010PTC123456",
        "financial_year": "2024-25",
        "directors_report": {
            "revenue_growth": "15.2% YoY",
            "key_highlights": [
                "Expanded manufacturing capacity by 30%",
                "Entered 2 new export markets (UAE, Singapore)",
                "Employee count grew from 450 to 580",
            ],
            "risk_factors": [
                "Raw material price volatility in steel and copper",
                "Dependency on single large customer (35% of revenue)",
                "Pending litigation with former supplier (Rs 2.5 Crore)",
            ],
        },
        "financials": {
            "revenue": 150_00_00_000,
            "net_profit": 12_00_00_000,
            "ebitda": 22_00_00_000,
            "total_assets": 95_00_00_000,
            "total_debt": 45_00_00_000,
            "equity": 30_00_00_000,
            "current_ratio": 1.42,
            "debt_to_equity": 1.50,
            "interest_coverage": 3.2,
        },
        "auditor": {
            "firm": "Mehta & Associates, Chartered Accountants",
            "opinion": "Unqualified",
            "emphasis_of_matter": "Pending tax dispute of Rs 85 Lakhs for AY 2021-22",
        },
        "promoter_holding": 62.5,
    }
    with open(annual_report_path, "w") as f:
        json.dump(annual_report_data, f, indent=2, ensure_ascii=False)
    created_files.append(annual_report_path)

    # ── Sample 4: Credit Officer Qualitative Notes (TXT) ──
    notes_path = os.path.join(UPLOAD_DIR, "site_visit_notes.txt")
    notes_content = """Credit Officer Site Visit Report
Officer: Rajesh Kumar (Senior Credit Manager)
Date: 15th February 2025
Company: ABC Industries Private Limited
Location: Plot No. 45, MIDC Industrial Area, Pune, Maharashtra

FACTORY OBSERVATIONS:
- Factory is operating at approximately 65% capacity (3 of 5 production lines active)
- Two production lines were shut down for "maintenance" but appeared to have been idle for several months
- Inventory levels appeared high — finished goods warehouse was nearly full
- Modern machinery observed (CNC machines from 2022), well-maintained
- Workers appeared experienced and production quality looked good

MANAGEMENT MEETING:
- Met with MD Mr. Anil Sharma and CFO Mrs. Priya Deshmukh
- MD was confident about growth plans, mentioned new orders from Tata Motors pipeline
- CFO was evasive about the pending litigation with former supplier
- When asked about high inventory, management attributed it to "seasonal demand patterns"
- Plans to start 4th production line by Q3 2025

CONCERNS:
- The 2 idle production lines are a red flag despite "maintenance" claim
- High finished goods inventory could indicate demand slowdown
- CFO's reluctance to discuss litigation in detail is concerning
- Single customer dependency (Tata Motors) creates concentration risk

OVERALL ASSESSMENT: Moderate Risk — strong manufacturing setup but operational concerns exist
"""
    with open(notes_path, "w") as f:
        f.write(notes_content)
    created_files.append(notes_path)

    # ── Sample 5: Legal Notice (TXT) ──
    legal_path = os.path.join(UPLOAD_DIR, "legal_notice_sample.txt")
    legal_content = """LEGAL NOTICE

IN THE NATIONAL COMPANY LAW TRIBUNAL, MUMBAI BENCH
Company Petition No. CP/123/NCLT/MUM/2024

IN THE MATTER OF:
M/s RawMat Supplies Pvt Ltd                    ... Petitioner
vs
M/s ABC Industries Private Limited              ... Respondent

FACTS OF THE CASE:
1. The Petitioner supplied raw materials worth Rs 2,50,00,000 (Rupees Two Crores Fifty Lakhs)
   to the Respondent between April 2023 and September 2023.
2. The Respondent has failed to make payment despite multiple reminders and follow-ups.
3. Cheques issued by the Respondent on 15th October 2023 (Rs 1,00,00,000) and
   20th November 2023 (Rs 75,00,000) were both dishonoured.
4. Total outstanding amount including interest: Rs 2,85,00,000.

CURRENT STATUS: Pending hearing. Next date: 15th March 2025.
RISK LEVEL: HIGH — involves dishonoured cheques (Section 138 NI Act potential)
"""
    with open(legal_path, "w") as f:
        f.write(legal_content)
    created_files.append(legal_path)

    logger.info(f"Created {len(created_files)} sample files in: {UPLOAD_DIR}")
    return created_files


def run_test_pipeline(use_llm: bool = True):
    """Run the full ingestion pipeline on sample data."""

    print("\n" + "=" * 70)
    print("    INTELLI-CREDIT: Data Ingestion Pipeline Test")
    print("=" * 70)

    # ── Step 1: Clean previous test data ──
    print("\n[Step 1] Cleaning previous test data...")
    try:
        delete_collection(CHROMA_COLLECTION_NAME, CHROMA_DB_DIR)
        print("  ✓ Previous collection deleted")
    except Exception:
        print("  ✓ No previous collection to clean")

    # ── Step 2: Create sample files ──
    print("\n[Step 2] Creating sample test files...")
    sample_files = create_sample_files()
    for f in sample_files:
        print(f"  ✓ Created: {os.path.basename(f)}")

    # ── Step 3: Initialize the agent ──
    print(f"\n[Step 3] Initializing Document Parser Agent (LLM: {'ON' if use_llm else 'OFF'})...")
    agent = DocumentParserAgent(use_llm=use_llm)
    print("  ✓ Agent initialized")

    # ── Step 4: Ingest all files ──
    print("\n[Step 4] Ingesting files through the pipeline...")
    print("-" * 50)

    results = agent.ingest_directory(UPLOAD_DIR)

    print("-" * 50)
    print(f"\n  Files processed: {len(results)}")
    for r in results:
        status_icon = "✓" if r["status"] == "success" else "✗"
        print(f"  {status_icon} {r['file_name']} → {r.get('document_type', 'N/A')} [{r['status']}]")

    # ── Step 5: Ingest qualitative notes ──
    print("\n[Step 5] Ingesting qualitative credit officer notes...")
    note_result = agent.ingest_qualitative_notes(
        notes=(
            "Factory found operating at 40% capacity. "
            "Two production lines were idle. "
            "Management claims seasonal slowdown but inventory levels are very high. "
            "Recommend reducing proposed exposure by 25%."
        ),
        officer_name="Suresh Patel",
    )
    status_icon = "✓" if note_result["status"] == "success" else "✗"
    print(f"  {status_icon} Qualitative note ingested [{note_result['status']}]")

    # ── Step 6: Show ingestion summary ──
    print("\n[Step 6] Ingestion Summary")
    print("-" * 50)
    summary = agent.get_ingestion_summary()
    print(f"  Total files processed: {summary['files_processed']}")
    print(f"  Successful: {summary['successful']}")
    print(f"  Errors: {summary['errors']}")
    print(f"  Total chunks in ChromaDB: {summary['vector_store'].get('total_chunks', 'N/A')}")

    # ── Step 7: Test vector store queries ──
    print("\n[Step 7] Testing vector store queries...")
    print("-" * 50)

    test_queries = [
        "What is the GST revenue and tax liability?",
        "Tell me about any litigation or legal disputes",
        "What did the credit officer observe at the factory?",
        "What is the company's debt to equity ratio?",
        "Are there any cheque bounces in the bank statement?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")
        result = query_documents(
            query_text=query,
            n_results=2,
            collection_name=CHROMA_COLLECTION_NAME,
            persist_dir=CHROMA_DB_DIR,
        )

        if result["results"]:
            for i, r in enumerate(result["results"]):
                doc_type = r["metadata"].get("document_type", "N/A")
                text_preview = r["text"][:120].replace("\n", " ")
                distance = r.get("distance", "N/A")
                print(f"    → [{doc_type}] (dist: {distance:.4f}): {text_preview}...")
        else:
            print("    → No results found")

    # ── Done ──
    print("\n" + "=" * 70)
    print("  ✓ INGESTION PIPELINE TEST COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intelli-Credit Ingestion Pipeline Test")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without LLM (skip Ollama classification/extraction)",
    )
    args = parser.parse_args()

    run_test_pipeline(use_llm=not args.no_llm)
