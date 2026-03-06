"""
ChromaDB Vector Store — Handles embedding, storage, and retrieval of document chunks.
Uses sentence-transformers for embeddings and ChromaDB for persistence.
"""
import os
import uuid
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Global instances (singleton pattern)
_chroma_client = None
_embedding_model = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load the sentence-transformers embedding model (cached)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}...")
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install: pip install sentence-transformers"
            )
    return _embedding_model


def _get_chroma_client(persist_dir: str):
    """Get or create a persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        try:
            import chromadb
            from chromadb.config import Settings
            logger.info(f"Initializing ChromaDB at: {persist_dir}")
            _chroma_client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB initialized successfully.")
        except ImportError:
            raise ImportError("chromadb is required. Install: pip install chromadb")
    return _chroma_client


def get_or_create_collection(
    collection_name: str = "intelli_credit_docs",
    persist_dir: str = None,
):
    """
    Get or create a ChromaDB collection.

    Args:
        collection_name: Name of the collection.
        persist_dir: Directory for ChromaDB persistence.

    Returns:
        ChromaDB collection object.
    """
    if persist_dir is None:
        from config import CHROMA_DB_DIR
        persist_dir = CHROMA_DB_DIR

    client = _get_chroma_client(persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "Intelli-Credit document embeddings"},
    )
    logger.info(
        f"Collection '{collection_name}' ready — {collection.count()} documents stored"
    )
    return collection


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    Uses sentence-aware splitting to avoid breaking mid-sentence.

    Args:
        text: Full document text.
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    # Split by sentences first (rough sentence splitting)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk_size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep overlap from the end of current chunk
            if overlap > 0:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # If no sentence splitting worked (e.g., no periods), fall back to character splitting
    if len(chunks) <= 1 and len(text) > chunk_size:
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts: List of text strings to embed.
        model_name: Sentence-transformers model name.

    Returns:
        List of embedding vectors (each is a list of floats).
    """
    if not texts:
        return []

    model = _get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def store_document(
    text: str,
    metadata: Dict,
    collection_name: str = "intelli_credit_docs",
    persist_dir: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict:
    """
    The main ingestion function: chunk text, generate embeddings, store in ChromaDB.

    Args:
        text: Full document text.
        metadata: Document metadata (file_name, document_type, etc.).
        collection_name: ChromaDB collection name.
        persist_dir: ChromaDB persistence directory.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        embedding_model: Sentence-transformers model name.

    Returns:
        Dict with storage results: chunk_count, document_ids, status.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided — skipping storage.")
        return {"status": "skipped", "reason": "empty text", "chunk_count": 0}

    # 1) Chunk the text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return {"status": "skipped", "reason": "no chunks generated", "chunk_count": 0}

    logger.info(f"Created {len(chunks)} chunks from document: {metadata.get('file_name', 'unknown')}")

    # 2) Generate embeddings
    embeddings = generate_embeddings(chunks, embedding_model)

    # 3) Prepare IDs and metadata for each chunk
    doc_id = str(uuid.uuid4())[:8]
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

    chunk_metadatas = []
    for i, chunk in enumerate(chunks):
        chunk_meta = {
            "document_id": doc_id,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "chunk_length": len(chunk),
            "file_name": metadata.get("file_name", "unknown"),
            "document_type": metadata.get("document_type", "OTHER"),
            "source": metadata.get("source", "upload"),
        }
        # Add any extra metadata (flatten non-dict/non-list values)
        for key, value in metadata.items():
            if key not in chunk_meta and isinstance(value, (str, int, float, bool)):
                chunk_meta[key] = value
        chunk_metadatas.append(chunk_meta)

    # 4) Store in ChromaDB
    collection = get_or_create_collection(collection_name, persist_dir)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=chunk_metadatas,
    )

    logger.info(
        f"Stored {len(chunks)} chunks in ChromaDB — "
        f"Document: {metadata.get('file_name', 'unknown')} (ID: {doc_id})"
    )

    return {
        "status": "success",
        "document_id": doc_id,
        "chunk_count": len(chunks),
        "chunk_ids": ids,
        "collection": collection_name,
    }


def query_documents(
    query_text: str,
    n_results: int = 5,
    collection_name: str = "intelli_credit_docs",
    persist_dir: str = None,
    filter_metadata: Optional[Dict] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Dict:
    """
    Query the vector store to find relevant document chunks.

    Args:
        query_text: The search query.
        n_results: Number of results to return.
        collection_name: ChromaDB collection name.
        persist_dir: ChromaDB persistence directory.
        filter_metadata: Optional metadata filter (e.g., {"document_type": "GST_RETURN"}).
        embedding_model: Sentence-transformers model name.

    Returns:
        Dict with "results" list, each containing "text", "metadata", "distance".
    """
    collection = get_or_create_collection(collection_name, persist_dir)

    # Generate query embedding
    query_embedding = generate_embeddings([query_text], embedding_model)[0]

    # Build query params
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": min(n_results, collection.count()) if collection.count() > 0 else 1,
    }
    if filter_metadata:
        query_params["where"] = filter_metadata

    try:
        results = collection.query(**query_params)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"results": [], "error": str(e)}

    # Format results
    formatted = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            formatted.append({
                "text": doc,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
                "id": results["ids"][0][i] if results["ids"] else None,
            })

    logger.info(f"Query returned {len(formatted)} results for: '{query_text[:50]}...'")
    return {"results": formatted, "query": query_text}


def get_collection_stats(
    collection_name: str = "intelli_credit_docs",
    persist_dir: str = None,
) -> Dict:
    """Get statistics about the vector store collection."""
    collection = get_or_create_collection(collection_name, persist_dir)
    count = collection.count()

    stats = {
        "collection_name": collection_name,
        "total_chunks": count,
    }

    if count > 0:
        # Get sample of metadata to see document types
        sample = collection.peek(min(count, 10))
        if sample and sample["metadatas"]:
            doc_types = set()
            file_names = set()
            for meta in sample["metadatas"]:
                doc_types.add(meta.get("document_type", "unknown"))
                file_names.add(meta.get("file_name", "unknown"))
            stats["document_types_sample"] = list(doc_types)
            stats["file_names_sample"] = list(file_names)

    return stats


def delete_collection(
    collection_name: str = "intelli_credit_docs",
    persist_dir: str = None,
):
    """Delete an entire collection (for cleanup/reset)."""
    if persist_dir is None:
        from config import CHROMA_DB_DIR
        persist_dir = CHROMA_DB_DIR

    client = _get_chroma_client(persist_dir)
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Could not delete collection '{collection_name}': {e}")
