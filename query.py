"""
Quick Query Tool — Ask questions using LangChain RetrievalQA with Groq LLM.

Usage:
  python query.py "What are the litigation risks?"
  python query.py "What is the total income of ORBIT EXPORTS?"
  python query.py                          # Interactive mode
"""
import sys
import os
import logging

logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR


def ask(question: str, n_results: int = 5):
    """Query the vector store and generate an answer using Groq LLM."""

    print(f"\n{'='*60}")
    print(f"  YOUR QUESTION: {question}")
    print(f"{'='*60}")

    # Get retrieved documents
    from vector_store.chroma_store import query_documents, get_collection_stats

    stats = get_collection_stats(CHROMA_COLLECTION_NAME, CHROMA_DB_DIR)
    print(f"  Searching {stats['total_chunks']} chunks in vector database...\n")

    results = query_documents(
        query_text=question,
        n_results=n_results,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_dir=CHROMA_DB_DIR,
    )

    if not results["results"]:
        print("  ❌ No results found!")
        return

    # Display retrieved chunks
    print("  📄 RELEVANT DOCUMENTS FOUND:")
    print(f"  {'-'*55}")

    retrieved_texts = []
    for i, r in enumerate(results["results"], 1):
        doc_type = r["metadata"].get("document_type", "N/A")
        file_name = r["metadata"].get("file_name", "N/A")
        text = r["text"].strip()
        display_text = text[:300]
        if len(text) > 300:
            display_text += "..."

        print(f"\n  [{i}] Source: {file_name} | Type: {doc_type}")
        print(f"      ┌{'─'*50}")
        for line in display_text.split("\n")[:8]:
            if line.strip():
                print(f"      │ {line.strip()}")
        print(f"      └{'─'*50}")

        retrieved_texts.append(text)

    # Generate answer using Groq LLM
    print(f"\n  {'='*55}")
    print(f"  🤖 AI-GENERATED ANSWER (Groq — Llama 3.3 70B):")
    print(f"  {'='*55}")

    answer = _generate_answer(question, retrieved_texts)
    print(f"\n  {answer}\n")


def _generate_answer(question: str, chunks: list) -> str:
    """Use Groq LLM to generate a natural language answer from retrieved chunks."""
    try:
        from langchain_setup import get_llm

        llm = get_llm()
        context = "\n\n---\n\n".join(chunks[:5])

        prompt = f"""Based on the following document excerpts, answer the question clearly.
If the documents don't contain enough information, say so.

DOCUMENTS:
\"\"\"
{context[:4000]}
\"\"\"

QUESTION: {question}

ANSWER (be specific, cite numbers and facts from the documents):"""

        response = llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        return content.strip()

    except Exception as e:
        return f"Could not generate answer: {e}"


if __name__ == "__main__":
    args = sys.argv[1:]

    if args:
        question = " ".join(args)
        ask(question)
    else:
        print("\n  ╔══════════════════════════════════════════╗")
        print("  ║  Intelli-Credit Query Tool (Groq LLM)   ║")
        print("  ║  Type your question, 'quit' to exit      ║")
        print("  ╚══════════════════════════════════════════╝\n")
        while True:
            question = input("  Ask > ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                print("  Goodbye!")
                break
            if question:
                ask(question)
