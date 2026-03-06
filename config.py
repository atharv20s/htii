"""
Central configuration for the Intelli-Credit pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress HuggingFace warnings and disable model downloads at import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# ============================================================
# Project Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Groq API Configuration (Fast cloud LLM)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ============================================================
# Tavily AI Configuration (Deep web search)
# ============================================================
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# ============================================================
# Local LLM (Ollama) — fallback if Groq is unavailable
# ============================================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "tinyllama"

# ============================================================
# Embedding Model Configuration
# ============================================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ============================================================
# ChromaDB Configuration
# ============================================================
CHROMA_COLLECTION_NAME = "intelli_credit_docs"

# ============================================================
# Text Chunking Configuration
# ============================================================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ============================================================
# Document Classification Labels
# ============================================================
DOCUMENT_TYPES = [
    "GST_RETURN", "ITR", "BANK_STATEMENT", "ANNUAL_REPORT",
    "FINANCIAL_STATEMENT", "LEGAL_NOTICE", "SANCTION_LETTER",
    "BOARD_MINUTES", "RATING_REPORT", "SHAREHOLDING",
    "QUALITATIVE_NOTE", "RESEARCH_REPORT", "OTHER",
]

# ============================================================
# Supported File Extensions
# ============================================================
SUPPORTED_EXTENSIONS = {
    "pdf": ["pdf"],
    "image": ["png", "jpg", "jpeg", "tiff", "bmp"],
    "structured": ["csv", "xlsx", "xls", "json"],
    "text": ["txt", "md", "doc"],
}

# Set environment variables for LangChain tools
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
