#!/usr/bin/env python3
"""
MCP Server for Epstein Document RAG.

Provides semantic search + keyword search over the document corpus,
with RAG capabilities for answering questions.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Any
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "document_analysis.db"
MODEL_NAME = "all-MiniLM-L6-v2"

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


def get_db() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


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
    cursor = conn.execute("""
        SELECT e.doc_id, e.embedding, d.paragraph_summary, d.one_sentence_summary,
               d.category, d.date_range_earliest, d.date_range_latest
        FROM document_embeddings e
        JOIN documents d ON e.doc_id = d.doc_id
    """)

    results = []
    for row in cursor:
        # Decode embedding from blob
        doc_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
        similarity = cosine_similarity(query_embedding, doc_embedding)

        results.append({
            "doc_id": row["doc_id"],
            "similarity": similarity,
            "summary": row["paragraph_summary"] or row["one_sentence_summary"],
            "category": row["category"],
            "date_range": f"{row['date_range_earliest'] or '?'} - {row['date_range_latest'] or '?'}"
        })

    # Sort by similarity and return top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    conn.close()

    return results[:limit]


def keyword_search(keywords: list[str], limit: int = 10) -> list[dict]:
    """
    Search documents by keywords in full text.
    """
    conn = get_db()

    # Build search query
    conditions = " AND ".join(["full_text LIKE ?" for _ in keywords])
    params = [f"%{kw}%" for kw in keywords]

    cursor = conn.execute(f"""
        SELECT doc_id, paragraph_summary, one_sentence_summary, category,
               date_range_earliest, date_range_latest
        FROM documents
        WHERE {conditions}
        LIMIT ?
    """, params + [limit])

    results = []
    for row in cursor:
        results.append({
            "doc_id": row["doc_id"],
            "summary": row["paragraph_summary"] or row["one_sentence_summary"],
            "category": row["category"],
            "date_range": f"{row['date_range_earliest'] or '?'} - {row['date_range_latest'] or '?'}"
        })

    conn.close()
    return results


def get_document_text(doc_id: str) -> str | None:
    """Get full text of a document."""
    conn = get_db()
    cursor = conn.execute("SELECT full_text FROM documents WHERE doc_id = ?", [doc_id])
    row = cursor.fetchone()
    conn.close()
    return row["full_text"] if row else None


def get_relationships_for_actor(actor: str, limit: int = 50) -> list[dict]:
    """Get relationships involving an actor."""
    conn = get_db()

    cursor = conn.execute("""
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
        results.append({
            "doc_id": row["doc_id"],
            "timestamp": row["timestamp"],
            "actor": row["actor"],
            "action": row["action"],
            "target": row["target"],
            "location": row["location"],
            "topic": row["explicit_topic"] or row["implicit_topic"]
        })

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
