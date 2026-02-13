#!/usr/bin/env python3
"""
Generate embeddings for all documents in the Epstein database.
Uses sentence-transformers with a model optimized for semantic search.
"""

import sqlite3
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

DB_PATH = Path(__file__).parent / "document_analysis.db"
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
BATCH_SIZE = 64


def create_embeddings_table(conn: sqlite3.Connection):
    """Create the document_embeddings table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB,
            text_source TEXT,
            model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_embeddings_doc_id ON document_embeddings(doc_id)")
    conn.commit()


def get_documents_to_embed(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Get all documents that don't have embeddings yet."""
    cursor = conn.execute("""
        SELECT d.doc_id,
               COALESCE(d.paragraph_summary, d.one_sentence_summary, '') as text
        FROM documents d
        LEFT JOIN document_embeddings e ON d.doc_id = e.doc_id
        WHERE e.doc_id IS NULL
        AND (d.paragraph_summary IS NOT NULL OR d.one_sentence_summary IS NOT NULL)
    """)
    return cursor.fetchall()


def embed_documents(docs: list[tuple[str, str]], model: SentenceTransformer, conn: sqlite3.Connection):
    """Generate and store embeddings for documents."""

    doc_ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]

    print(f"Generating embeddings for {len(docs)} documents...")

    # Process in batches with progress bar
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch_ids = doc_ids[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]

        # Generate embeddings
        embeddings = model.encode(batch_texts, show_progress_bar=False)

        # Store in database
        for doc_id, embedding in zip(batch_ids, embeddings):
            # Store as binary blob (more efficient than JSON)
            embedding_bytes = embedding.astype(np.float32).tobytes()
            conn.execute(
                "INSERT OR REPLACE INTO document_embeddings (doc_id, embedding, text_source, model) VALUES (?, ?, ?, ?)",
                (doc_id, embedding_bytes, "paragraph_summary", MODEL_NAME)
            )

        conn.commit()


def main():
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    create_embeddings_table(conn)

    docs = get_documents_to_embed(conn)
    print(f"Found {len(docs)} documents without embeddings")

    if docs:
        embed_documents(docs, model, conn)
        print("Done!")
    else:
        print("All documents already have embeddings.")

    # Verify
    count = conn.execute("SELECT COUNT(*) FROM document_embeddings").fetchone()[0]
    print(f"Total embeddings in database: {count}")

    conn.close()


if __name__ == "__main__":
    main()
