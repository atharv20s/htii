"""
LangChain Setup — Central configuration for LLM, embeddings, and vector store.
Uses Groq for fast cloud LLM inference, sentence-transformers for embeddings.
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    EMBEDDING_MODEL, CHROMA_DB_DIR, CHROMA_COLLECTION_NAME,
)

logger = logging.getLogger(__name__)

_llm = None
_embeddings = None
_vectorstore = None


def get_llm(temperature: float = 0.1):
    """
    Get the LLM — uses Groq (fast cloud inference with Llama 3.3 70B).
    Falls back to Ollama if Groq is unavailable.
    """
    global _llm
    if _llm is None:
        try:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(
                model=GROQ_MODEL,
                api_key=GROQ_API_KEY,
                temperature=temperature,
            )
            # Quick test
            _llm.invoke("test")
            logger.info(f"LLM initialized: Groq ({GROQ_MODEL})")
        except Exception as e:
            logger.warning(f"Groq unavailable ({e}), falling back to Ollama")
            from langchain_ollama import ChatOllama
            _llm = ChatOllama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=temperature,
            )
            logger.info(f"LLM initialized: Ollama ({OLLAMA_MODEL})")
    return _llm


def get_embeddings(model_name: str = None):
    """Get LangChain-compatible embeddings using sentence-transformers."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name or EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        logger.info(f"Embeddings initialized: {model_name or EMBEDDING_MODEL}")
    return _embeddings


def get_vectorstore(collection_name: str = None, persist_dir: str = None):
    """Get LangChain-compatible ChromaDB vector store."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        _vectorstore = Chroma(
            collection_name=collection_name or CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=persist_dir or CHROMA_DB_DIR,
        )
        logger.info("ChromaDB vector store initialized")
    return _vectorstore


def get_retriever(search_k: int = 5, filter_dict: dict = None):
    """Get a LangChain retriever from the vector store."""
    vs = get_vectorstore()
    search_kwargs = {"k": search_k}
    if filter_dict:
        search_kwargs["filter"] = filter_dict
    return vs.as_retriever(search_kwargs=search_kwargs)
