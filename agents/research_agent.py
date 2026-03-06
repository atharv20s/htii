"""
Research Agent — LangGraph ReAct Agent with Tavily Deep Search & Groq LLM.

All investigation prompts are embedded directly inside the tools.
Architecture:
    - LangGraph ReAct agent with tool-calling
    - Tavily AI deep web search (9 specialized tools with Indian data source prompts)
    - ChromaDB vector store for uploaded document knowledge
    - Groq (Llama 3.3 70B) for reasoning and summarization
    - Human-in-the-loop: loops until user is satisfied
    - Final output saved to .txt file

Indian Data Sources Targeted:
    - MCA (Ministry of Corporate Affairs) filings
    - CRISIL, ICRA, CARE Ratings, India Ratings
    - Screener.in, Tijori Finance, Trendlyne, MoneyControl
    - eCourts India, NCLT/NCLAT
    - GST Network, Income Tax
    - RBI, NITI Aayog, World Steel Association
    - Economic Times, Business Standard, Mint
    - Zauba Corp, OpenCorporates
    - BSE, NSE, Nifty 50
    - CDP, MSCI ESG Research
"""
import os
import sys
import json
import logging
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TAVILY_API_KEY, CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)

MAX_TOOL_OUTPUT = 2500  # Max chars per tool — increased for richer financial data extraction


def _truncate(text: str, max_len: int = None) -> str:
    limit = max_len or MAX_TOOL_OUTPUT
    return text[:limit] + "\n...[truncated]" if len(text) > limit else text


# ============================================================
# Tools with Indian data source prompts embedded
# ============================================================

def _create_tools(company_name: str, promoter_names: List[str] = None, sector: str = None):
    """Create all 9 tools with deep Indian data source prompts baked in."""
    from langchain_core.tools import tool
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    promoters = promoter_names or []
    sector_name = sector or "corporate"

    # ── Tool 1: Company Profile (MCA + BSE/NSE) ──

    @tool
    def investigate_company_profile(query: str) -> str:
        """Investigate company profile using MCA filings, BSE/NSE, and financial news.

        Sources: Ministry of Corporate Affairs (MCA), BSE India, NSE India,
        Economic Times, Business Standard, Mint, MoneyControl.

        Extracts: company registration, director information, annual filings,
        financial statements, charges/loans on assets, BSE/NSE listing status,
        Nifty 50 membership, market cap, business model, operational scale.

        Input: search query about the company profile."""
        try:
            queries = [
                f"{company_name} {query} MCA filings director information registration India",
                f"{company_name} BSE NSE stock Nifty 50 market cap share price India",
            ]
            output = ""
            for q in queries:
                r = tavily_client.search(query=q, search_depth="advanced", max_results=2, include_answer=True)
                output += f"Summary: {r.get('answer', 'N/A')}\n"
                for s in r.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:150]}\n"
            doc_results = _search_docs(f"{company_name} business profile operations")
            output += f"\nDOCS:\n{doc_results}"
            return _truncate(output)
        except Exception as e:
            return f"Company investigation failed: {e}"

    # ── Tool 2: Financials (Screener + Trendlyne + Tijori) ──

    @tool
    def investigate_financials(query: str) -> str:
        """Analyze financials using Screener.in, Tijori Finance, Trendlyne, and documents.

        Sources: Screener.in, Tijori Finance, Trendlyne, MoneyControl,
        uploaded ITR, GST Returns, Bank Statements, Annual Reports.

        Extracts: revenue growth, debt levels, ROE, profit margins,
        debt-to-equity ratio, current ratio, interest coverage,
        cash flow patterns, working capital trends, related party transactions.

        Input: financial query about the company."""
        try:
            doc_results = _search_docs(f"{company_name} {query} revenue profit tax income balance sheet")
            queries = [
                f"{company_name} {query} revenue net profit EBITDA FY2024 FY2023 FY2022 screener.in India crores",
                f"{company_name} debt equity ratio current ratio interest coverage DSCR ROE ROCE working capital India",
            ]
            output = f"DOCS:\n{doc_results}\n"
            for q in queries:
                web = tavily_client.search(
                    query=q, search_depth="advanced", max_results=2, include_answer=True,
                )
                output += f"WEB: {web.get('answer', 'N/A')}\n"
                for s in web.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:200]}\n"
            return _truncate(output)
        except Exception as e:
            return f"Financial investigation failed: {e}"

    # ── Tool 3: Credit Ratings (CRISIL + ICRA + CARE) ──

    @tool
    def investigate_credit_rating(query: str) -> str:
        """Search credit ratings from Indian rating agencies — critical for banks.

        Sources: CRISIL, ICRA, CARE Ratings, India Ratings and Research.

        Extracts: credit rating (AAA, AA+, etc.), rating outlook (stable/positive/negative),
        downgrade history, debt profile, rating rationale, long-term vs short-term ratings.
        Banks rely heavily on these for credit decisions.

        Input: query about credit rating or debt profile."""
        try:
            queries = [
                f"{company_name} {query} credit rating CRISIL ICRA outlook 2025",
                f"{company_name} rating downgrade upgrade CARE India Ratings debt profile",
            ]
            output = ""
            for q in queries:
                r = tavily_client.search(query=q, search_depth="advanced", max_results=2, include_answer=True)
                output += f"Summary: {r.get('answer', 'N/A')}\n"
                for s in r.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:150]}\n"
            return _truncate(output)
        except Exception as e:
            return f"Credit rating search failed: {e}"

    # ── Tool 4: Litigation (eCourts + NCLT) ──

    @tool
    def investigate_litigation(query: str) -> str:
        """Search legal disputes from eCourts India, NCLT/NCLAT, and regulatory bodies.

        Sources: eCourts India, National Company Law Tribunal (NCLT/NCLAT),
        Supreme Court, High Courts, Enforcement Directorate (ED), CBI, SFIO.

        Checks: insolvency/CIRP cases, contract disputes, tax litigation,
        regulatory violations, cheque bounce (Section 138 NI Act),
        wilful defaulter classification, fraud allegations, ED investigations.

        Input: query about litigation or legal matters."""
        try:
            queries = [
                f'"{company_name}" {query} case ecourts NCLT litigation India',
                f'"{company_name}" wilful defaulter NPA cheque bounce ED investigation',
            ]
            for name in promoters[:1]:
                queries.append(f'"{name}" fraud court case arrested investigation India')
            output = ""
            for q in queries[:2]:
                r = tavily_client.search(query=q, search_depth="advanced", max_results=2, include_answer=True)
                output += f"Summary: {r.get('answer', 'N/A')}\n"
                for s in r.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:150]}\n"
            doc_results = _search_docs("litigation legal dispute court notice")
            output += f"\nDOCS:\n{doc_results}"
            return _truncate(output)
        except Exception as e:
            return f"Litigation search failed: {e}"

    # ── Tool 5: Sector Benchmarks (RBI + NITI Aayog + Industry Bodies) ──

    @tool
    def investigate_sector_benchmark(query: str) -> str:
        """Compare company with sector peers using RBI, NITI Aayog, and industry data.

        Sources: Reserve Bank of India (RBI), NITI Aayog, World Steel Association,
        Ministry of Steel, SAIL, Screener peer comparison, industry bodies.

        Extracts: sector growth forecast, average revenue & profit margins,
        production capacity vs peers, market share, export trends,
        currency risks, supply chain vulnerabilities, economic headwinds,
        government policies, tariffs, import/export regulations.

        Input: query about sector comparison or benchmarking."""
        try:
            queries = [
                f"{sector_name} industry outlook India 2025 growth forecast RBI NITI Aayog",
                f"{company_name} peer comparison {sector_name} average revenue profit margin India",
            ]
            output = ""
            for q in queries:
                r = tavily_client.search(query=q, search_depth="advanced", max_results=2, include_answer=True)
                output += f"Summary: {r.get('answer', 'N/A')}\n"
                for s in r.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:150]}\n"
            return _truncate(output)
        except Exception as e:
            return f"Sector benchmark failed: {e}"

    # ── Tool 6: Regulatory Compliance (GST Network + MCA + RBI) ──

    @tool
    def investigate_regulatory_compliance(query: str) -> str:
        """Check compliance using GST Network, MCA, RBI circulars, and SEBI.

        Sources: GST Network (GSTN), MCA (ROC filings), RBI circulars,
        SEBI regulations, Income Tax department.

        Checks: GST compliance news, GST notice, tax disputes,
        GSTR-2A vs GSTR-3B reconciliation, income tax filing regularity,
        MCA annual returns & financial statements filing status,
        environmental or labor law compliance, sector-specific rules.

        Input: query about regulatory or compliance matters."""
        try:
            queries = [
                f"{company_name} {query} GST notice tax dispute compliance India",
                f"{company_name} MCA filing ROC compliance annual return India",
            ]
            output = ""
            for q in queries:
                r = tavily_client.search(query=q, search_depth="advanced", max_results=2, include_answer=True)
                output += f"Summary: {r.get('answer', 'N/A')}\n"
                for s in r.get("results", [])[:2]:
                    output += f"- {s['title']}: {s['content'][:150]}\n"
            doc_results = _search_docs("GST compliance tax filing regulatory")
            output += f"DOCS:\n{doc_results}"
            return _truncate(output)
        except Exception as e:
            return f"Regulatory search failed: {e}"

    # ── Tool 7: Promoter Background (Zauba Corp + OpenCorporates) ──

    @tool
    def investigate_promoter_background(query: str) -> str:
        """Investigate promoters/directors using Zauba Corp, OpenCorporates, and news.

        Sources: Zauba Corp, OpenCorporates, MCA DIN lookup,
        Economic Times, Business Standard, Mint, court records.

        Extracts: director history and past companies, DIN details,
        legal disputes or criminal cases, financial credibility,
        political connections, related party transactions,
        media coverage, public reputation.

        Input: query about promoters or directors."""
        try:
            output = ""
            for name in (promoters if promoters else [company_name + " promoter director"]):
                queries = [
                    f'"{name}" {query} director companies history Zauba Corp India',
                    f'"{name}" fraud investigation court case legal issues India',
                ]
                for q in queries:
                    r = tavily_client.search(query=q, search_depth="basic", max_results=2, include_answer=True)
                    output += f"Summary: {r.get('answer', 'N/A')}\n"
                    for s in r.get("results", [])[:2]:
                        output += f"- {s['title']}: {s['content'][:150]}\n"
            return _truncate(output)
        except Exception as e:
            return f"Promoter investigation failed: {e}"

    # ── Tool 8: Document Knowledge (Vector DB) ──

    @tool
    def search_document_knowledge(query: str) -> str:
        """Search uploaded financial documents in the vector database.

        Searches ALL ingested docs: ITR, GST Returns (GSTR-1/3B/2A),
        Bank Statements, Annual Reports, Legal Notices, Sanction Letters,
        Credit Officer Site Visit Notes, Balance Sheet, P&L.

        Use for: cross-document comparison, detecting mismatched revenue numbers,
        finding missing disclosures, comparing GST vs ITR vs bank statements.

        Input: question about the company's financial documents."""
        return _search_docs(query)

    # ── Tool 9: Irregularity Detection + ESG + Hypothesis Testing ──

    @tool
    def detect_irregularities(query: str) -> str:
        """Detect irregularities, ESG risks, and test hypotheses across all sources.

        Combines document search + web to find:
        - Revenue mismatch: GST vs Bank Statement vs ITR
        - Abnormal growth not supported by market
        - Circular trading patterns, unusual ITC claims
        - Cash flow manipulation signs
        - ESG violations (CDP, MSCI ESG Research data)
        - Environmental violations, sustainability risks
        - Credit rating vs actual financial health mismatch

        Also performs:
        HYPOTHESIS TESTING — form, search evidence, CONFIRM/DENY/INCONCLUSIVE, score 0-1
        CONTRADICTION DETECTION — cross-check documents vs web vs news

        Input: what kind of irregularity to investigate."""
        try:
            doc_results = _search_docs(f"{company_name} {query} revenue income GST bank inconsistency")
            web = tavily_client.search(
                query=f"{company_name} {query} irregularity fraud GST notice ESG violation India",
                search_depth="advanced", max_results=2, include_answer=True,
            )
            output = f"DOCS:\n{doc_results}\n"
            output += f"WEB: {web.get('answer', 'N/A')}\n"
            for s in web.get("results", [])[:2]:
                output += f"- {s['title']}: {s['content'][:150]}\n"
            return _truncate(output)
        except Exception as e:
            return f"Irregularity detection failed: {e}"

    return [
        investigate_company_profile,
        investigate_financials,
        investigate_credit_rating,
        investigate_litigation,
        investigate_sector_benchmark,
        investigate_regulatory_compliance,
        investigate_promoter_background,
        search_document_knowledge,
        detect_irregularities,
    ]


def _search_docs(query: str) -> str:
    """Helper: search vector DB for document chunks."""
    try:
        from vector_store.chroma_store import query_documents
        results = query_documents(
            query_text=query, n_results=5,
            collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_DB_DIR,
        )
        if not results.get("results"):
            return "No relevant documents found."
        output = ""
        for i, r in enumerate(results["results"][:3], 1):
            doc_type = r["metadata"].get("document_type", "N/A")
            file_name = r["metadata"].get("file_name", "N/A")
            output += f"[{i}] {file_name} ({doc_type}): {r['text'][:200]}\n"
        return _truncate(output)
    except Exception as e:
        return f"Document search error: {e}"


# ============================================================
# LangGraph ReAct Agent with all prompts embedded
# ============================================================

def create_research_agent(company_name: str, promoter_names: List[str] = None, sector: str = None):
    """Create the LangGraph ReAct Research Agent."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage
    from langchain_setup import get_llm

    llm = get_llm()
    tools = _create_tools(company_name, promoter_names, sector)

    promoter_text = f"Promoters/Directors: {', '.join(promoter_names)}" if promoter_names else "Promoters: Not specified"
    sector_text = f"Sector: {sector}" if sector else "Sector: Not specified"

    system_prompt = SystemMessage(content=f"""You are an AI Credit Research Agent for Indian corporate lending.

COMPANY: {company_name}
{promoter_text}
{sector_text}

You behave like a senior credit analyst at an Indian bank evaluating a loan application.
You MUST extract SPECIFIC NUMBERS, not vague summaries. You investigate using:
1. Uploaded financial documents (vector DB) — ITR, GST Returns, Bank Statements, Annual Reports
2. Deep web search (Tavily AI) targeting Indian data sources
3. Sector benchmarks and regulatory databases

You have 9 specialized tools:
1. investigate_company_profile — MCA filings, BSE/NSE listing, company registration
2. investigate_financials — Screener.in, Trendlyne ratios, document cross-check
3. investigate_credit_rating — CRISIL, ICRA, CARE Ratings, India Ratings
4. investigate_litigation — eCourts India, NCLT, ED investigations
5. investigate_sector_benchmark — RBI, NITI Aayog, industry body reports
6. investigate_regulatory_compliance — GST Network, MCA ROC, RBI circulars
7. investigate_promoter_background — Zauba Corp, OpenCorporates, DIN lookup
8. search_document_knowledge — Search uploaded ITR/GST/Bank statements
9. detect_irregularities — ESG (CDP/MSCI), hypothesis testing, contradiction detection

MANDATORY RESEARCH STRATEGY (execute ALL 9 steps):
1. search_document_knowledge — Read internal financial documents FIRST
2. investigate_company_profile — MCA filings + BSE/NSE stock data
3. investigate_financials — Screener ratios + cross-check with documents
4. investigate_credit_rating — CRISIL/ICRA rating and outlook
5. investigate_litigation — eCourts + NCLT cases
6. investigate_sector_benchmark — RBI + industry comparison
7. investigate_regulatory_compliance — GST compliance + MCA filings
8. investigate_promoter_background — Zauba Corp + director history
9. detect_irregularities — Hypothesis testing + ESG + contradiction detection

CRITICAL RULES:
- ALWAYS extract EXACT NUMBERS: revenue in crores, D/E ratio, ROE %, DSCR, current ratio, interest coverage ratio, profit margins, EBITDA, net profit, cash flow figures
- If a number is found, ALWAYS include the year/period it belongs to
- Compare year-over-year: FY22 vs FY23 vs FY24 minimum
- Cross-verify: if GST revenue says X but ITR says Y, FLAG the mismatch
- For each conclusion provide confidence score 0-1 WITH justification for the score
- If info is missing or unavailable, explicitly state "DATA NOT FOUND" — never skip silently
- Explain reasoning behind every conclusion with source citations

FINAL REPORT FORMAT (follow EXACTLY):

COMPANY OVERVIEW
────────────────
Company Name: [full legal name]
CIN: [if found]
Incorporation Date: [date]
Registered Address: [address]
Business Model: [1-2 sentences]
Sector: [sector]
Key Products/Services: [list]
Promoters/Directors: [names with DIN if found]
BSE/NSE: [listing status, stock code, Nifty 50 membership]
Market Cap: [amount in crores]
Number of Employees: [if found]

CREDIT RATING
─────────────
Rating Agency: [name]
Long-Term Rating: [e.g., CARE AA+ / CRISIL AA]
Short-Term Rating: [e.g., CRISIL A1+]
Outlook: [Stable / Positive / Negative]
Last Rating Action: [upgrade/downgrade/reaffirmed] on [date]
Rating Rationale: [key reasons]
Downgrade History: [any past downgrades with dates]
Confidence: [0-1] — [why this score]

FINANCIAL ANALYSIS
──────────────────
 Revenue (3-year trend):
  - FY [year]: ₹[X] crores
  - FY [year]: ₹[X] crores
  - FY [year]: ₹[X] crores
 Net Profit (3-year trend):
  - FY [year]: ₹[X] crores
  - FY [year]: ₹[X] crores
  - FY [year]: ₹[X] crores
 EBITDA Margin: [X]%
 Net Profit Margin: [X]%
 Debt-to-Equity Ratio: [X]
 Current Ratio: [X]
 Interest Coverage Ratio: [X]
 DSCR (Debt Service Coverage Ratio): [X]
 Return on Equity (ROE): [X]%
 Return on Capital Employed (ROCE): [X]%
 Working Capital Cycle: [X] days
 Cash Flow from Operations: ₹[X] crores
 Free Cash Flow: ₹[X] crores
 Related Party Transactions: [summary if found]
 Document Cross-Check: [any mismatch between uploaded docs and web data]
Confidence: [0-1] — [why this score]

SECTOR BENCHMARK
────────────────
Industry: [sector name]
Sector Growth Rate: [X]% CAGR
Company vs Sector Average:
  - Revenue Growth: Company [X]% vs Sector [Y]%
  - Profit Margin: Company [X]% vs Sector [Y]%
  - D/E Ratio: Company [X] vs Sector avg [Y]
Key Peers: [peer names and brief comparison]
Sector Outlook: [positive/neutral/negative with reasons]
Government Policy Impact: [relevant policies affecting sector]
Confidence: [0-1] — [why this score]

PROMOTER BACKGROUND
───────────────────
Promoter Name(s): [names]
DIN: [if found]
Other Directorships: [list of companies]
Past Defaults/Legal Issues: [any found]
Political Connections: [if any]
Related Party Concerns: [if any]
Public Reputation: [summary from news]
Wilful Defaulter Status: [Yes/No]
Confidence: [0-1] — [why this score]

REGULATORY RISK
───────────────
GST Compliance: [compliant/issues found]
  - GSTR Filing Status: [regular/delayed]
  - Any GST Notices: [details]
MCA Filing Status: [up to date / delayed]
Income Tax Disputes: [details if any]
SEBI Violations: [details if any]
Environmental Compliance: [any violations]
Labor Law Compliance: [any violations]
Risk Level: LOW / MEDIUM / HIGH
Confidence: [0-1] — [why this score]

LITIGATION RISK
───────────────
Active Cases: [count and brief description]
  - Case 1: [parties, court, status, amount involved]
  - Case 2: [parties, court, status, amount involved]
NCLT/IBC Proceedings: [details if any]
ED/CBI Investigations: [details if any]
Cheque Bounce Cases (Sec 138 NI Act): [count if any]
Total Litigation Exposure: ₹[X] crores (estimated)
Risk Level: LOW / MEDIUM / HIGH
Confidence: [0-1] — [why this score]

IRREGULARITIES & ESG
─────────────────────
Financial Irregularities Found:
  - [list each with evidence, e.g., "GST revenue ₹X vs ITR declared ₹Y — mismatch of ₹Z"]
ESG Risks:
  - Environmental: [violations, pollution cases, carbon footprint issues]
  - Social: [labor disputes, safety incidents, community conflicts]
  - Governance: [board independence, related party issues, whistle-blower cases]
Contradictions Detected:
  - [document vs web data mismatches]
Confidence: [0-1] — [why this score]

HYPOTHESIS TESTING
──────────────────
Test each of these hypotheses with evidence:

H1: Revenue Consistency
  Hypothesis: "Revenue reported in GST returns matches ITR and bank statements"
  Result: CONFIRMED / DENIED / INCONCLUSIVE
  Evidence: [specific numbers and sources]

H2: Debt Sustainability
  Hypothesis: "The company can service its debt obligations (DSCR > 1.2)"
  Result: CONFIRMED / DENIED / INCONCLUSIVE
  Evidence: [DSCR value, interest coverage, cash flow data]

H3: Growth Authenticity
  Hypothesis: "Company's reported growth is consistent with sector trends"
  Result: CONFIRMED / DENIED / INCONCLUSIVE
  Evidence: [company growth vs sector growth comparison]

H4: Promoter Integrity
  Hypothesis: "Promoters have no history of defaults, fraud, or wilful default"
  Result: CONFIRMED / DENIED / INCONCLUSIVE
  Evidence: [findings from background check]

H5: Regulatory Standing
  Hypothesis: "The company is compliant with GST, MCA, and tax regulations"
  Result: CONFIRMED / DENIED / INCONCLUSIVE
  Evidence: [compliance status, any notices or disputes]

FINAL CREDIT INSIGHT
────────────────────
Overall Risk: LOW / MEDIUM / HIGH
Key Strengths:
  1. [strength]
  2. [strength]
  3. [strength]
Key Concerns:
  1. [concern]
  2. [concern]
  3. [concern]
Credit Opinion: [3-4 sentence detailed recommendation on lending decision]
Recommended Loan Terms: [if applicable — tenure, collateral, conditions]
Confidence: [0-1] — [why this score]
""")

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


# ============================================================
# Human-in-the-loop runner
# ============================================================

def run_research_with_human_loop(
    company_name: str,
    promoter_names: List[str] = None,
    sector: str = None,
):
    """Run the Research Agent with human-in-the-loop."""
    agent = create_research_agent(company_name, promoter_names, sector)

    initial_query = (
        f"Perform a comprehensive credit investigation on '{company_name}'. "
        f"Follow ALL 9 steps of the mandatory research strategy — do NOT skip any tool. "
        f"STEP 1: Search uploaded documents for revenue, profit, GST, bank statement data. "
        f"STEP 2: Investigate company profile — get CIN, directors, MCA registration. "
        f"STEP 3: Get EXACT financial numbers — revenue in crores for FY22/FY23/FY24, D/E ratio, ROE, DSCR, current ratio, interest coverage, EBITDA margin, net profit margin. "
        f"STEP 4: Get credit rating from CRISIL/ICRA/CARE — exact rating, outlook, last action date. "
        f"STEP 5: Search eCourts/NCLT for active litigation — count cases, estimate exposure in crores. "
        f"STEP 6: Compare with sector peers — company growth vs sector average growth, margins vs sector. "
        f"STEP 7: Check GST compliance, MCA filings, tax disputes. "
        f"STEP 8: Investigate promoter/director backgrounds — other companies, legal issues, wilful defaulter status. "
        f"STEP 9: Detect irregularities — cross-check GST revenue vs ITR vs bank statements, test all 5 hypotheses. "
        f"IMPORTANT: Extract SPECIFIC NUMBERS with years, not vague descriptions. "
        f"Produce the final report in the EXACT required format with all sections filled."
    )

    print(f"\n{'='*60}")
    print(f"  🔍 RESEARCH AGENT — LangGraph ReAct")
    print(f"  Company: {company_name}")
    if promoter_names:
        print(f"  Promoters: {', '.join(promoter_names)}")
    if sector:
        print(f"  Sector: {sector}")
    print(f"  LLM: Groq (Llama 3.3 70B)")
    print(f"  Search: Tavily AI (Deep Search)")
    print(f"  Sources: MCA, CRISIL, Screener, eCourts, GST, BSE/NSE")
    print(f"{'='*60}")

    all_research = []
    current_query = initial_query
    iteration = 1

    while True:
        print(f"\n  🤖 Research iteration {iteration}...")
        print(f"  {'-'*50}")

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": current_query}]}
            )

            final_answer = ""
            for msg in result.get("messages", []):
                if hasattr(msg, 'content') and msg.content:
                    final_answer = msg.content

            print(f"\n  📋 RESEARCH FINDINGS (Iteration {iteration}):")
            print(f"  {'='*55}")
            print(f"\n{final_answer}")
            print(f"\n  {'='*55}")

            all_research.append({
                "iteration": iteration,
                "query": current_query,
                "findings": final_answer,
            })

        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"\n  ❌ {error_msg}")
            all_research.append({
                "iteration": iteration,
                "query": current_query,
                "findings": error_msg,
            })

        # Human-in-the-loop
        print(f"\n  {'─'*50}")
        print("  Options:")
        print("    [1] Satisfied — save and finish")
        print("    [2] Research more (type your follow-up)")
        print("    [3] Quit without saving")

        user_input = input("\n  Your choice > ").strip()

        if user_input == "1":
            output_path = _save_research_report(company_name, all_research)
            _store_research_in_vectordb(company_name, all_research)
            print(f"\n  ✅ Research saved to: {output_path}")
            print(f"  ✅ Research stored in vector database")
            break
        elif user_input == "3":
            print("\n  ❌ Exiting without saving.")
            break
        else:
            if user_input == "2":
                current_query = input("  Type your follow-up question > ").strip()
            else:
                current_query = user_input
            iteration += 1

    return all_research


def _save_research_report(company_name: str, research_iterations: List[Dict]) -> str:
    """Save complete research to .txt file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = company_name.replace(" ", "_").lower()
    filename = f"research_{safe_name}_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  INTELLI-CREDIT RESEARCH REPORT\n")
        f.write(f"  Company: {company_name}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Agent: LangGraph ReAct + Tavily + Groq\n")
        f.write(f"  Sources: MCA, CRISIL/ICRA/CARE, Screener, eCourts, GST, BSE/NSE\n")
        f.write(f"{'='*70}\n\n")

        for item in research_iterations:
            f.write(f"\n{'─'*70}\n")
            f.write(f"  ITERATION {item['iteration']}\n")
            f.write(f"{'─'*70}\n\n")
            f.write(f"{item['findings']}\n\n")

        f.write(f"\n{'='*70}\n")
        f.write(f"  END OF RESEARCH REPORT\n")
        f.write(f"{'='*70}\n")

    return filepath


def _store_research_in_vectordb(company_name: str, research_iterations: List[Dict]):
    """Store research in vector DB for other agents."""
    try:
        from vector_store.chroma_store import store_document
        all_text = "\n\n".join(item["findings"] for item in research_iterations)
        if all_text.strip():
            store_document(
                text=all_text,
                metadata={
                    "file_name": f"research_{company_name.replace(' ', '_').lower()}",
                    "document_type": "RESEARCH_REPORT",
                    "source": "tavily_deep_research",
                    "company": company_name,
                    "research_date": datetime.now().isoformat(),
                },
                collection_name=CHROMA_COLLECTION_NAME,
                persist_dir=CHROMA_DB_DIR,
            )
    except Exception as e:
        logger.warning(f"Could not store research: {e}")
