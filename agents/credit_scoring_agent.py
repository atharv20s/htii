"""
Credit Scoring Agent — LangGraph ReAct Agent that scores credit risk.

Takes research report output and produces:
    - Category-wise scores (0-100) across 7 dimensions
    - Weighted overall credit score
    - Indian credit rating (AAA → D)
    - Red flags and lending recommendation

Architecture:
    - LangGraph ReAct agent with tool-calling
    - Groq LLM (Llama 3.3 70B) for analysis
    - ChromaDB for document cross-checks
    - Structured JSON output from LLM for reliable scoring
"""
import os
import sys
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ============================================================
# Scoring Configuration
# ============================================================

SCORING_CATEGORIES = {
    "financial_health": {
        "name": "Financial Health",
        "weight": 0.30,
        "description": "Revenue trends, profitability, debt ratios, cash flow, DSCR",
        "prompt": """Evaluate the company's FINANCIAL HEALTH from the research report.

Score 0-100 based on:
- Revenue trend (growing = higher score, declining = lower)
- Profitability (net profit margin, EBITDA margin)
- Debt-to-Equity ratio (< 1 = good, > 2 = risky)
- DSCR / Interest Coverage (> 1.5 = safe, < 1 = danger)
- Current Ratio (> 1.5 = healthy, < 1 = liquidity risk)
- ROE / ROCE (> 15% = strong, < 5% = weak)
- Cash flow from operations (positive = good, negative = bad)
- Working capital adequacy

Scoring guide:
- 90-100: Exceptional financials, growing revenue, low debt, strong cash flow
- 70-89: Good financials with minor concerns
- 50-69: Average, some debt concerns or flat growth
- 30-49: Weak financials, high debt, or declining revenue
- 0-29: Severe financial distress, negative cash flow, unsustainable debt""",
    },
    "credit_rating": {
        "name": "Credit Rating",
        "weight": 0.15,
        "description": "Agency ratings, outlook, downgrade history",
        "prompt": """Evaluate the company's CREDIT RATING status from the research report.

Score 0-100 based on:
- Current credit rating from CRISIL/ICRA/CARE/India Ratings
- Rating outlook (Stable/Positive = higher, Negative = lower)
- Any recent downgrades (penalize heavily)
- Long-term vs short-term rating consistency
- Rating rationale strength

Scoring guide:
- 90-100: AAA/AA+ rating with stable/positive outlook, no downgrades
- 70-89: AA/A+ rating, stable outlook
- 50-69: BBB+/BBB rating or recent downgrade from higher
- 30-49: BB or below, negative outlook
- 0-29: D rating, multiple downgrades, or wilful defaulter
- If NO rating exists: score 40 (uncertainty penalty)""",
    },
    "promoter_background": {
        "name": "Promoter Background",
        "weight": 0.15,
        "description": "Promoter integrity, past defaults, legal issues, reputation",
        "prompt": """Evaluate PROMOTER/DIRECTOR BACKGROUND from the research report.

Score 0-100 based on:
- Promoter track record (successful businesses = good)
- Any past defaults or loan restructuring
- Wilful defaulter status (if yes = score below 20)
- Criminal cases or fraud allegations against promoters
- Political connections (can be risk or benefit)
- Related party transaction concerns
- Media reputation and public perception
- Director disqualifications by MCA

Scoring guide:
- 90-100: Clean track record, reputable promoters, no legal issues
- 70-89: Generally clean with minor concerns
- 50-69: Some concerns but no criminal issues
- 30-49: Significant concerns, ongoing investigations
- 0-29: Wilful defaulter, fraud cases, arrested directors""",
    },
    "regulatory_compliance": {
        "name": "Regulatory Compliance",
        "weight": 0.10,
        "description": "GST, MCA, SEBI, tax compliance status",
        "prompt": """Evaluate REGULATORY COMPLIANCE from the research report.

Score 0-100 based on:
- GST filing regularity (GSTR-1, GSTR-3B on time)
- Any GST notices or demands
- MCA annual return filing status
- Income tax disputes or demands
- SEBI compliance (for listed companies)
- Environmental/labor law compliance
- Sector-specific regulatory adherence

Scoring guide:
- 90-100: Fully compliant, no notices, regular filings
- 70-89: Mostly compliant, minor delays
- 50-69: Some notices or disputes under resolution
- 30-49: Multiple regulatory issues, large tax demands
- 0-29: Serious violations, prosecution, or penalties""",
    },
    "litigation_risk": {
        "name": "Litigation Risk",
        "weight": 0.10,
        "description": "Court cases, NCLT proceedings, ED investigations",
        "prompt": """Evaluate LITIGATION RISK from the research report.

Score 0-100 based on (INVERT — higher score = LOWER risk):
- Number of active court cases
- Total litigation exposure (amount in crores)
- NCLT/IBC proceedings (if any = major red flag)
- ED/CBI investigations
- Cheque bounce cases (Sec 138 NI Act)
- Nature of cases (civil vs criminal)
- Wilful defaulter classification

Scoring guide:
- 90-100: No significant litigation, clean record
- 70-89: Minor civil cases, low exposure
- 50-69: Some active cases but manageable exposure
- 30-49: Significant litigation, NCLT proceedings
- 0-29: ED investigation, criminal cases, wilful defaulter""",
    },
    "sector_position": {
        "name": "Sector & Market Position",
        "weight": 0.10,
        "description": "Sector outlook, peer comparison, market position",
        "prompt": """Evaluate SECTOR & MARKET POSITION from the research report.

Score 0-100 based on:
- Industry growth outlook (growing sector = higher)
- Company's market share vs peers
- Revenue growth vs sector average
- Profit margins vs sector average
- Competitive advantages (market leader bonus)
- Government policy tailwinds or headwinds
- Export dependence and currency risks
- Supply chain resilience

Scoring guide:
- 90-100: Market leader in growing sector, outperforming peers
- 70-89: Strong position, growing sector
- 50-69: Average position, stable sector
- 30-49: Below average, declining sector or losing market share
- 0-29: Weak position in declining sector, no competitive edge""",
    },
    "esg_irregularities": {
        "name": "ESG & Irregularities",
        "weight": 0.10,
        "description": "Environmental, social, governance risks, data mismatches",
        "prompt": """Evaluate ESG RISKS & IRREGULARITIES from the research report.

Score 0-100 based on (INVERT — higher score = FEWER problems):
- Environmental violations or pollution cases
- Social issues (labor disputes, safety incidents)
- Governance concerns (board independence, whistle-blower cases)
- Financial irregularities detected (revenue mismatches)
- GST vs ITR vs bank statement inconsistencies
- Circular trading or unusual ITC claims
- Related party transaction anomalies
- Contradiction between documents and web sources

Scoring guide:
- 90-100: No irregularities, strong ESG practices
- 70-89: Minor ESG concerns, no financial irregularities
- 50-69: Some ESG issues or minor data inconsistencies
- 30-49: Significant irregularities or ESG violations
- 0-29: Major fraud indicators, serious data mismatches""",
    },
}

# Score to Indian credit rating mapping
SCORE_TO_RATING = [
    (95, "AAA",  "Very Low"),
    (90, "AA+",  "Very Low"),
    (85, "AA",   "Low"),
    (80, "AA-",  "Low"),
    (75, "A+",   "Moderate-Low"),
    (70, "A",    "Moderate-Low"),
    (65, "BBB+", "Moderate"),
    (60, "BBB",  "Moderate"),
    (55, "BBB-", "Moderate"),
    (50, "BB+",  "Moderate-High"),
    (45, "BB",   "Moderate-High"),
    (40, "B+",   "High"),
    (35, "B",    "High"),
    (30, "CCC",  "Very High"),
    (0,  "D",    "Default Risk"),
]


def _get_rating(score: float) -> tuple:
    """Map numerical score to credit rating and risk level."""
    for threshold, rating, risk in SCORE_TO_RATING:
        if score >= threshold:
            return rating, risk
    return "D", "Default Risk"


def _get_lending_decision(score: float, red_flags: list) -> dict:
    """Determine lending recommendation based on score and red flags."""
    critical_flags = [f for f in red_flags if f.get("severity") == "CRITICAL"]

    if score >= 70 and not critical_flags:
        decision = "APPROVE"
        detail = "Strong credit profile. Standard lending terms recommended."
    elif score >= 50 and len(critical_flags) <= 1:
        decision = "CONDITIONAL APPROVE"
        conditions = []
        if score < 60:
            conditions.append("Higher collateral requirement (1.5x coverage)")
        if any("promoter" in f.get("area", "").lower() for f in red_flags):
            conditions.append("Personal guarantee from promoters")
        if any("litigation" in f.get("area", "").lower() for f in red_flags):
            conditions.append("Litigation indemnity clause")
        if any("regulatory" in f.get("area", "").lower() for f in red_flags):
            conditions.append("Quarterly compliance certificate")
        if not conditions:
            conditions.append("Enhanced monitoring with quarterly review")
        detail = "Conditional approval with: " + "; ".join(conditions)
    else:
        decision = "REJECT"
        reasons = [f["description"] for f in critical_flags[:3]] if critical_flags else ["Overall credit score below acceptable threshold"]
        detail = "Rejected due to: " + "; ".join(reasons)

    return {"decision": decision, "detail": detail}


# ============================================================
# LangGraph ReAct Agent for Credit Scoring
# ============================================================

def create_scoring_agent(company_name: str, research_report: str):
    """Create the LangGraph ReAct Credit Scoring Agent."""
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage
    from langchain_core.tools import tool
    from langchain_setup import get_llm

    llm = get_llm()

    # ── Tool 1: Analyze a scoring category ──
    @tool
    def analyze_category(category_key: str) -> str:
        """Analyze and score a specific credit category from the research report.

        Valid category keys: financial_health, credit_rating, promoter_background,
        regulatory_compliance, litigation_risk, sector_position, esg_irregularities

        Returns a JSON with score (0-100), justification, and red flags.
        Input: one of the category key names listed above."""
        if category_key not in SCORING_CATEGORIES:
            return json.dumps({"error": f"Invalid category. Use one of: {list(SCORING_CATEGORIES.keys())}"})

        cat = SCORING_CATEGORIES[category_key]
        scoring_prompt = f"""You are a senior credit analyst scoring a company's creditworthiness.

CATEGORY: {cat['name']}
WEIGHT: {cat['weight'] * 100}%

{cat['prompt']}

RESEARCH REPORT TO ANALYZE:
\"\"\"
{research_report[:6000]}
\"\"\"

You MUST respond with ONLY valid JSON (no markdown, no extra text):
{{
    "category": "{category_key}",
    "score": <integer 0-100>,
    "justification": "<2-3 sentences explaining the score with specific data points from the report>",
    "red_flags": [
        {{"description": "<specific concern>", "severity": "CRITICAL or WARNING or INFO", "area": "{cat['name']}"}}
    ],
    "data_points": ["<key fact 1>", "<key fact 2>", "<key fact 3>"]
}}

If data is missing for this category, score 40-50 (uncertainty) and note it in justification."""

        try:
            response = llm.invoke(scoring_prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            # Try to parse JSON from response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                content = content.rsplit("```", 1)[0] if "```" in content else content
                content = content.strip()
            result = json.loads(content)
            # Clamp score to 0-100
            result["score"] = max(0, min(100, int(result.get("score", 50))))
            return json.dumps(result, indent=2)
        except json.JSONDecodeError:
            return json.dumps({
                "category": category_key,
                "score": 50,
                "justification": f"LLM response could not be parsed. Raw: {content[:200]}",
                "red_flags": [],
                "data_points": [],
            })
        except Exception as e:
            return json.dumps({
                "category": category_key,
                "score": 50,
                "justification": f"Analysis failed: {str(e)}",
                "red_flags": [],
                "data_points": [],
            })

    # ── Tool 2: Cross-check with uploaded documents ──
    @tool
    def cross_check_documents(query: str) -> str:
        """Search uploaded financial documents to verify or cross-check scoring data.

        Use this to verify specific claims from the research report against
        the actual uploaded documents (ITR, GST returns, bank statements, etc.).

        Input: specific question to verify against documents."""
        try:
            from vector_store.chroma_store import query_documents
            results = query_documents(
                query_text=query, n_results=3,
                collection_name=CHROMA_COLLECTION_NAME, persist_dir=CHROMA_DB_DIR,
            )
            if not results.get("results"):
                return "No relevant documents found for cross-checking."
            output = ""
            for i, r in enumerate(results["results"][:3], 1):
                doc_type = r["metadata"].get("document_type", "N/A")
                file_name = r["metadata"].get("file_name", "N/A")
                output += f"[{i}] {file_name} ({doc_type}): {r['text'][:300]}\n"
            return output
        except Exception as e:
            return f"Document cross-check failed: {e}"

    # ── Tool 3: Calculate final weighted score ──
    @tool
    def calculate_final_score(category_scores_json: str) -> str:
        """Calculate the final weighted credit score from all category scores.

        Input: JSON string with category scores, e.g.:
        {"financial_health": 75, "credit_rating": 80, "promoter_background": 70, ...}

        Returns the final score, rating, and risk level."""
        try:
            scores = json.loads(category_scores_json)
            weighted_total = 0.0
            total_weight = 0.0

            breakdown = []
            for key, cat in SCORING_CATEGORIES.items():
                score = scores.get(key, 50)  # default 50 if missing
                weighted = score * cat["weight"]
                weighted_total += weighted
                total_weight += cat["weight"]
                breakdown.append({
                    "category": cat["name"],
                    "score": score,
                    "weight": f"{cat['weight']*100:.0f}%",
                    "weighted_score": round(weighted, 1),
                })

            overall = round(weighted_total / total_weight, 1) if total_weight > 0 else 50
            rating, risk_level = _get_rating(overall)

            return json.dumps({
                "overall_score": overall,
                "credit_rating": rating,
                "risk_level": risk_level,
                "breakdown": breakdown,
            }, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Calculation failed: {e}"})

    tools = [analyze_category, cross_check_documents, calculate_final_score]

    system_prompt = SystemMessage(content=f"""You are an AI Credit Scoring Agent for Indian corporate lending.

COMPANY: {company_name}

You have received a detailed research report about this company. Your job is to
SCORE the company's creditworthiness across 7 categories and produce a final credit score.

You have 3 tools:
1. analyze_category — Score a specific category (call this for EACH of the 7 categories)
2. cross_check_documents — Verify data against uploaded financial documents
3. calculate_final_score — Compute the weighted final score

MANDATORY EXECUTION PLAN:
1. Call analyze_category for "financial_health"
2. Call analyze_category for "credit_rating"
3. Call analyze_category for "promoter_background"
4. Call analyze_category for "regulatory_compliance"
5. Call analyze_category for "litigation_risk"
6. Call analyze_category for "sector_position"
7. Call analyze_category for "esg_irregularities"
8. Call cross_check_documents to verify any suspicious findings
9. Call calculate_final_score with all 7 scores

After getting all results, produce the FINAL CREDIT SCORECARD in this EXACT format:

════════════════════════════════════════════════════
  INTELLI-CREDIT SCORECARD
  Company: {company_name}
  Date: [today's date]
════════════════════════════════════════════════════

OVERALL SCORE: [X]/100
CREDIT RATING: [rating]
RISK LEVEL: [risk level]

CATEGORY BREAKDOWN
──────────────────────────────────────────────────
  #  Category                  Score   Weight   Weighted
  1. Financial Health          [XX]    30%      [XX.X]
  2. Credit Rating             [XX]    15%      [XX.X]
  3. Promoter Background       [XX]    15%      [XX.X]
  4. Regulatory Compliance     [XX]    10%      [XX.X]
  5. Litigation Risk           [XX]    10%      [XX.X]
  6. Sector & Market Position  [XX]    10%      [XX.X]
  7. ESG & Irregularities      [XX]    10%      [XX.X]
──────────────────────────────────────────────────

DETAILED JUSTIFICATION
──────────────────────
[For each category, 2-3 lines explaining the score with data points]

RED FLAGS
─────────
[List all red flags found, with severity CRITICAL/WARNING/INFO]

KEY DATA POINTS
───────────────
[List the most important financial facts extracted]

LENDING RECOMMENDATION
──────────────────────
Decision: APPROVE / CONDITIONAL APPROVE / REJECT
Conditions: [if conditional, list conditions]
Confidence: [0-1]

IMPORTANT RULES:
- You MUST call analyze_category for ALL 7 categories — do NOT skip any
- Be harsh but fair — do not inflate scores
- Back every score with specific data from the report
- If data is missing, score 40-50 and flag it
""")

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


# ============================================================
# Runner with output saving
# ============================================================

def run_credit_scoring(
    company_name: str,
    research_report: str,
    interactive: bool = True,
) -> Dict:
    """Run the Credit Scoring Agent on a research report.

    Args:
        company_name: Name of the company
        research_report: Full text of the research report
        interactive: If True, show human-in-the-loop options

    Returns:
        Dict with scoring results
    """
    agent = create_scoring_agent(company_name, research_report)

    scoring_query = (
        f"Score the creditworthiness of '{company_name}' using the research report provided. "
        f"Call analyze_category for ALL 7 categories one by one: "
        f"financial_health, credit_rating, promoter_background, "
        f"regulatory_compliance, litigation_risk, sector_position, esg_irregularities. "
        f"Then call cross_check_documents to verify key financial claims. "
        f"Then call calculate_final_score with all 7 scores. "
        f"Finally, produce the COMPLETE credit scorecard in the required format."
    )

    print(f"\n{'='*60}")
    print(f"  📊 CREDIT SCORING AGENT")
    print(f"  Company: {company_name}")
    print(f"  LLM: Groq (Llama 3.3 70B)")
    print(f"  Categories: 7 | Weights: Financial 30%, Rating 15%,")
    print(f"              Promoter 15%, Regulatory 10%, Litigation 10%,")
    print(f"              Sector 10%, ESG 10%")
    print(f"{'='*60}")

    print(f"\n  🤖 Analyzing research report and scoring categories...")
    print(f"  {'-'*50}")

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": scoring_query}]}
        )

        final_answer = ""
        for msg in result.get("messages", []):
            if hasattr(msg, 'content') and msg.content:
                final_answer = msg.content

        print(f"\n  📋 CREDIT SCORECARD:")
        print(f"  {'='*55}")
        print(f"\n{final_answer}")
        print(f"\n  {'='*55}")

    except Exception as e:
        final_answer = f"Scoring failed: {e}"
        print(f"\n  ❌ {final_answer}")

    scoring_result = {
        "company": company_name,
        "timestamp": datetime.now().isoformat(),
        "scorecard": final_answer,
        "research_report_length": len(research_report),
    }

    if interactive:
        print(f"\n  {'─'*50}")
        print("  Options:")
        print("    [1] Save scorecard and finish")
        print("    [2] Re-score with adjustments (type feedback)")
        print("    [3] Quit without saving")

        user_input = input("\n  Your choice > ").strip()

        if user_input == "1":
            output_path = _save_scoring_report(company_name, scoring_result)
            _store_scoring_in_vectordb(company_name, scoring_result)
            print(f"\n  ✅ Scorecard saved to: {output_path}")
            print(f"  ✅ Scorecard stored in vector database")
        elif user_input == "3":
            print("\n  ❌ Exiting without saving.")
        else:
            feedback = user_input if user_input != "2" else input("  Your feedback > ").strip()
            print(f"\n  🔄 Re-scoring is not yet implemented. Save current results with [1].")
    else:
        output_path = _save_scoring_report(company_name, scoring_result)
        _store_scoring_in_vectordb(company_name, scoring_result)
        print(f"\n  ✅ Scorecard auto-saved to: {output_path}")

    return scoring_result


def _save_scoring_report(company_name: str, scoring_result: Dict) -> str:
    """Save the credit scorecard to a .txt file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = company_name.replace(" ", "_").lower()
    filename = f"scorecard_{safe_name}_{timestamp}.txt"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  INTELLI-CREDIT SCORECARD\n")
        f.write(f"  Company: {company_name}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  Agent: LangGraph ReAct + Groq\n")
        f.write(f"  Scoring: 7-Category Weighted Model\n")
        f.write(f"{'='*70}\n\n")
        f.write(scoring_result.get("scorecard", "No scorecard generated."))
        f.write(f"\n\n{'='*70}\n")
        f.write(f"  END OF SCORECARD\n")
        f.write(f"{'='*70}\n")

    return filepath


def _store_scoring_in_vectordb(company_name: str, scoring_result: Dict):
    """Store scorecard in vector DB for downstream agents (report generator)."""
    try:
        from vector_store.chroma_store import store_document
        scorecard = scoring_result.get("scorecard", "")
        if scorecard.strip():
            store_document(
                text=scorecard,
                metadata={
                    "file_name": f"scorecard_{company_name.replace(' ', '_').lower()}",
                    "document_type": "CREDIT_SCORECARD",
                    "source": "credit_scoring_agent",
                    "company": company_name,
                    "scoring_date": datetime.now().isoformat(),
                },
                collection_name=CHROMA_COLLECTION_NAME,
                persist_dir=CHROMA_DB_DIR,
            )
    except Exception as e:
        logger.warning(f"Could not store scorecard: {e}")


# ============================================================
# Standalone execution
# ============================================================

if __name__ == "__main__":
    import glob

    print("\n  ╔══════════════════════════════════════════════╗")
    print("  ║  Intelli-Credit Scoring Agent               ║")
    print("  ╚══════════════════════════════════════════════╝\n")

    # Find existing research reports
    report_files = glob.glob(os.path.join(OUTPUT_DIR, "research_*.txt")) + \
                   glob.glob(os.path.join(OUTPUT_DIR, "test_research_*.txt"))

    if report_files:
        print("  Found research reports:")
        for i, f in enumerate(report_files, 1):
            print(f"    [{i}] {os.path.basename(f)}")
        print(f"    [{len(report_files)+1}] Enter company name manually")

        choice = input("\n  Select report > ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(report_files):
                filepath = report_files[idx]
                with open(filepath, "r", encoding="utf-8") as f:
                    research_text = f.read()

                # Extract company name from filename
                basename = os.path.basename(filepath)
                # e.g., "research_tata_steel_20260306.txt" or "test_research_tata_steel.txt"
                name_part = basename.replace("test_research_", "").replace("research_", "")
                name_part = name_part.rsplit("_", 1)[0] if "_20" in name_part else name_part.replace(".txt", "")
                company_name = name_part.replace("_", " ").title()

                confirm = input(f"  Company: {company_name}. Correct? (y/n) > ").strip()
                if confirm.lower() != "y":
                    company_name = input("  Enter correct company name > ").strip()

                run_credit_scoring(company_name, research_text)
            else:
                company_name = input("  Enter company name > ").strip()
                report_path = input("  Enter path to research report > ").strip()
                with open(report_path, "r", encoding="utf-8") as f:
                    research_text = f.read()
                run_credit_scoring(company_name, research_text)
        except (ValueError, IndexError):
            print("  Invalid choice.")
    else:
        print("  No research reports found in data/output/")
        print("  Run the research agent first to generate a report.")
