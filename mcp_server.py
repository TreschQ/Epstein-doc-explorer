#!/usr/bin/env python3
"""
MCP Server for Epstein Document RAG.

Provides semantic search + keyword search over the document corpus,
with RAG capabilities for answering questions.

Supports both SQLite (local dev) and PostgreSQL (production).
Set DATABASE_URL environment variable to use PostgreSQL.
"""

import os
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
SQLITE_PATH = Path(__file__).parent / "document_analysis.db"
MODEL_NAME = "all-MiniLM-L6-v2"

# PostgreSQL support (lazy import)
_pg_pool = None

# Initialize MCP server
mcp = FastMCP("epstein-docs")

# Global model (lazy loaded)
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_db():
    """Get database connection (SQLite or PostgreSQL)."""
    if DATABASE_URL:
        return _get_postgres_conn()
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def _get_postgres_conn():
    """Get PostgreSQL connection."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(DATABASE_URL)
    conn.cursor_factory = psycopg2.extras.RealDictCursor
    return conn


def _is_postgres():
    """Check if using PostgreSQL."""
    return DATABASE_URL is not None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search(query: str, limit: int = 10) -> list[dict]:
    """
    Search documents by semantic similarity.
    Returns documents ranked by cosine similarity to the query.
    """
    model = get_model()
    conn = get_db()

    # Embed the query
    query_embedding = model.encode([query])[0].astype(np.float32)

    # Get all document embeddings
    cursor = conn.cursor()
    cursor.execute("""
        SELECT e.doc_id, e.embedding, d.paragraph_summary, d.one_sentence_summary,
               d.category, d.date_range_earliest, d.date_range_latest
        FROM document_embeddings e
        JOIN documents d ON e.doc_id = d.doc_id
    """)

    results = []
    for row in cursor:
        # Handle both SQLite Row and PostgreSQL dict
        if hasattr(row, 'keys'):
            row_dict = dict(row)
        else:
            row_dict = row

        # Decode embedding from blob/bytea
        embedding_data = row_dict["embedding"]
        if isinstance(embedding_data, memoryview):
            embedding_data = bytes(embedding_data)
        doc_embedding = np.frombuffer(embedding_data, dtype=np.float32)
        similarity = cosine_similarity(query_embedding, doc_embedding)

        results.append({
            "doc_id": row_dict["doc_id"],
            "similarity": similarity,
            "summary": row_dict["paragraph_summary"] or row_dict["one_sentence_summary"],
            "category": row_dict["category"],
            "date_range": f"{row_dict['date_range_earliest'] or '?'} - {row_dict['date_range_latest'] or '?'}"
        })

    # Sort by similarity and return top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    cursor.close()
    conn.close()

    return results[:limit]


def keyword_search(keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Search documents by keywords in full text.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Build search query - PostgreSQL uses ILIKE, SQLite uses LIKE
    if _is_postgres():
        conditions = " AND ".join(["full_text ILIKE %s" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords] + [limit]
        query = f"""
            SELECT doc_id, paragraph_summary, one_sentence_summary, category,
                   date_range_earliest, date_range_latest
            FROM documents
            WHERE {conditions}
            LIMIT %s
        """
    else:
        conditions = " AND ".join(["full_text LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords] + [limit]
        query = f"""
            SELECT doc_id, paragraph_summary, one_sentence_summary, category,
                   date_range_earliest, date_range_latest
            FROM documents
            WHERE {conditions}
            LIMIT ?
        """

    cursor.execute(query, params)

    results = []
    for row in cursor:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        results.append({
            "doc_id": row_dict["doc_id"],
            "summary": row_dict["paragraph_summary"] or row_dict["one_sentence_summary"],
            "category": row_dict["category"],
            "date_range": f"{row_dict['date_range_earliest'] or '?'} - {row_dict['date_range_latest'] or '?'}"
        })

    cursor.close()
    conn.close()
    return results


def get_document_text(doc_id: str) -> str | None:
    """Get full text of a document."""
    conn = get_db()
    cursor = conn.cursor()

    if _is_postgres():
        cursor.execute("SELECT full_text FROM documents WHERE doc_id = %s", [doc_id])
    else:
        cursor.execute("SELECT full_text FROM documents WHERE doc_id = ?", [doc_id])

    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        return row_dict["full_text"]
    return None


def get_relationships_for_actor(actor: str, limit: int = 50) -> list[dict]:
    """Get relationships involving an actor."""
    conn = get_db()
    cursor = conn.cursor()

    if _is_postgres():
        cursor.execute("""
            SELECT t.doc_id, t.timestamp, t.actor, t.action, t.target, t.location,
                   t.explicit_topic, t.implicit_topic
            FROM rdf_triples t
            LEFT JOIN entity_aliases a ON t.actor = a.original_name
            WHERE t.actor ILIKE %s OR t.target ILIKE %s
               OR a.canonical_name ILIKE %s
            ORDER BY t.timestamp
            LIMIT %s
        """, [f"%{actor}%", f"%{actor}%", f"%{actor}%", limit])
    else:
        cursor.execute("""
            SELECT t.doc_id, t.timestamp, t.actor, t.action, t.target, t.location,
                   t.explicit_topic, t.implicit_topic
            FROM rdf_triples t
            LEFT JOIN entity_aliases a ON t.actor = a.original_name
            WHERE t.actor LIKE ? OR t.target LIKE ?
               OR a.canonical_name LIKE ?
            ORDER BY t.timestamp
            LIMIT ?
        """, [f"%{actor}%", f"%{actor}%", f"%{actor}%", limit])

    results = []
    for row in cursor:
        row_dict = dict(row) if hasattr(row, 'keys') else row
        results.append({
            "doc_id": row_dict["doc_id"],
            "timestamp": row_dict["timestamp"],
            "actor": row_dict["actor"],
            "action": row_dict["action"],
            "target": row_dict["target"],
            "location": row_dict["location"],
            "topic": row_dict["explicit_topic"] or row_dict["implicit_topic"]
        })

    cursor.close()
    conn.close()
    return results


# ============== MCP Tools ==============

@mcp.tool()
def search_documents(query: str, limit: int = 10) -> str:
    """
    Search documents using semantic similarity.

    Args:
        query: Natural language search query (e.g., "meetings with politicians")
        limit: Maximum number of results to return (default 10)

    Returns:
        List of relevant documents with summaries and similarity scores.
    """
    results = semantic_search(query, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def search_by_keywords(keywords: str, limit: int = 10) -> str:
    """
    Search documents by keywords in full text.

    Args:
        keywords: Comma-separated keywords (e.g., "island, flight, 2005")
        limit: Maximum number of results to return (default 10)

    Returns:
        List of documents containing all keywords.
    """
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = keyword_search(keyword_list, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_document(doc_id: str) -> str:
    """
    Get the full text of a specific document.

    Args:
        doc_id: Document ID (e.g., "gov.uscourts.nysd.447706.195.0")

    Returns:
        Full document text.
    """
    text = get_document_text(doc_id)
    if text:
        return text
    return f"Document not found: {doc_id}"


@mcp.tool()
def search_actor(actor_name: str, limit: int = 50) -> str:
    """
    Get all relationships involving a specific actor/person.

    Args:
        actor_name: Name of the person (e.g., "Bill Clinton", "Ghislaine Maxwell")
        limit: Maximum number of relationships to return

    Returns:
        List of relationships (actor -> action -> target) involving this person.
    """
    results = get_relationships_for_actor(actor_name, limit)
    return json.dumps(results, indent=2)


@mcp.tool()
def get_stats() -> str:
    """
    Get database statistics.

    Returns:
        Total documents, relationships, actors, and categories.
    """
    conn = get_db()

    stats = {
        "total_documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "total_relationships": conn.execute("SELECT COUNT(*) FROM rdf_triples").fetchone()[0],
        "total_actors": conn.execute("SELECT COUNT(DISTINCT actor) FROM rdf_triples").fetchone()[0],
        "total_embeddings": conn.execute("SELECT COUNT(*) FROM document_embeddings").fetchone()[0],
        "categories": []
    }

    cursor = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC
    """)
    stats["categories"] = [{"name": row[0], "count": row[1]} for row in cursor]

    conn.close()
    return json.dumps(stats, indent=2)


@mcp.tool()
def hybrid_search(query: str, keywords: str = "", limit: int = 10) -> str:
    """
    Combined semantic + keyword search for best results.

    Args:
        query: Natural language query for semantic search
        keywords: Optional comma-separated keywords to filter results
        limit: Maximum results to return

    Returns:
        Documents matching both semantic similarity and keywords.
    """
    # Get semantic results
    semantic_results = semantic_search(query, limit * 3)

    if not keywords:
        return json.dumps(semantic_results[:limit], indent=2)

    # Filter by keywords
    keyword_list = [k.strip().lower() for k in keywords.split(",")]

    filtered = []
    for result in semantic_results:
        text = get_document_text(result["doc_id"])
        if text:
            text_lower = text.lower()
            if all(kw in text_lower for kw in keyword_list):
                result["full_text_preview"] = text[:500] + "..." if len(text) > 500 else text
                filtered.append(result)

        if len(filtered) >= limit:
            break

    return json.dumps(filtered, indent=2)


@mcp.tool()
def answer_question(question: str, num_docs: int = 5) -> str:
    """
    Retrieve relevant context for answering a question about the Epstein documents.

    This tool searches for the most relevant documents and returns them as context.
    Use this when you need to answer questions about the document corpus.

    Args:
        question: The question to answer
        num_docs: Number of documents to retrieve for context (default 5)

    Returns:
        Relevant document excerpts that can be used to answer the question.
    """
    # Search for relevant documents
    results = semantic_search(question, num_docs)

    context_parts = []
    for i, result in enumerate(results, 1):
        doc_text = get_document_text(result["doc_id"])
        preview = doc_text[:2000] if doc_text else "No text available"

        context_parts.append(f"""
--- Document {i} (similarity: {result['similarity']:.3f}) ---
Doc ID: {result['doc_id']}
Category: {result['category']}
Date Range: {result['date_range']}
Summary: {result['summary']}

Excerpt:
{preview}
""")

    return "\n".join(context_parts)


if __name__ == "__main__":
    mcp.run()
