#!/usr/bin/env python3
"""
Migrate embeddings to PGVector for efficient semantic search.

This script:
1. Adds the pgvector extension to PostgreSQL
2. Creates a vector column for embeddings
3. Migrates existing BYTEA embeddings to PGVector format
4. Creates vector index for fast similarity search

Usage:
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
    python migrate_to_pgvector.py
"""

import os
import sys
import numpy as np
from tqdm import tqdm

import psycopg2
from psycopg2.extras import execute_values

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimensions


def get_postgres_conn():
    """Get PostgreSQL connection from DATABASE_URL."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    return psycopg2.connect(url)


def add_pgvector_extension(conn):
    """Add pgvector extension to PostgreSQL."""
    print("Adding pgvector extension...")
    cursor = conn.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    print("✓ pgvector extension added")


def add_vector_column(conn):
    """Add vector column to document_embeddings table."""
    print("Adding vector column...")
    cursor = conn.cursor()

    # First, check if pgvector extension is available
    cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if not cursor.fetchone():
        raise Exception("pgvector extension is not installed. Cannot proceed with migration.")

    # Add vector column
    try:
        cursor.execute(f"""
            ALTER TABLE document_embeddings
            ADD COLUMN IF NOT EXISTS embedding_vector vector({EMBEDDING_DIM})
        """)
        conn.commit()
        print("✓ embedding_vector column added/verified")
    except Exception as e:
        print(f"Warning: Could not add embedding_vector column: {e}")
        conn.rollback()

    # Add tsvector column for full-text search
    try:
        cursor.execute("""
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS text_search_vector tsvector
        """)
        conn.commit()
        print("✓ text_search_vector column added/verified")
    except Exception as e:
        print(f"Warning: Could not add text_search_vector column: {e}")
        conn.rollback()

    # Create GIN index for full-text search
    try:
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_text_search
            ON documents USING GIN(text_search_vector)
        """)
        conn.commit()
        print("✓ GIN index created/verified")
    except Exception as e:
        print(f"Warning: Could not create GIN index: {e}")
        conn.rollback()

    # Check if text_search_vector needs to be populated
    cursor.execute("""
        SELECT COUNT(*) FROM documents
        WHERE text_search_vector IS NULL
    """)
    null_count = cursor.fetchone()[0]

    if null_count > 0:
        print(f"Updating {null_count} documents with text search vectors...")
        cursor.execute("""
            UPDATE documents
            SET text_search_vector = to_tsvector('english',
                COALESCE(paragraph_summary, '') || ' ' ||
                COALESCE(one_sentence_summary, '') || ' ' ||
                COALESCE(full_text, '')
            )
            WHERE text_search_vector IS NULL
        """)
        conn.commit()
        print("✓ Text search vectors updated")
    else:
        print("✓ Text search vectors already populated")

    # Create trigger to update tsvector on document changes
    try:
        cursor.execute("""
            DROP TRIGGER IF EXISTS tsvector_update ON documents
        """)
        cursor.execute("""
            CREATE TRIGGER tsvector_update
            BEFORE INSERT OR UPDATE ON documents
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger(
                text_search_vector, 'pg_catalog.english',
                paragraph_summary, one_sentence_summary, full_text
            )
        """)
        conn.commit()
        print("✓ Trigger created/verified")
    except Exception as e:
        print(f"Warning: Could not create trigger: {e}")
        conn.rollback()

    print("✓ Vector and text search columns added")


def migrate_embeddings(conn, batch_size=500):
    """Migrate existing BYTEA embeddings to vector format."""
    print("Migrating embeddings to PGVector format...")

    cursor = conn.cursor()

    # Get embeddings that need migration
    cursor.execute("""
        SELECT doc_id, embedding
        FROM document_embeddings
        WHERE embedding IS NOT NULL AND embedding_vector IS NULL
    """)
    rows = cursor.fetchall()
    total = len(rows)

    if total == 0:
        print("No embeddings to migrate")
        return

    print(f"Converting {total} embeddings...")

    batch = []
    update_cursor = conn.cursor()

    for doc_id, embedding_blob in tqdm(rows, desc="Converting embeddings"):
        # Decode embedding from BYTEA
        if isinstance(embedding_blob, memoryview):
            embedding_bytes = bytes(embedding_blob)
        else:
            embedding_bytes = embedding_blob

        # Convert to numpy array and then to vector string format
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
        vector_str = str(embedding_array.tolist())

        batch.append((vector_str, doc_id))

        if len(batch) >= batch_size:
            update_cursor.executemany("""
                UPDATE document_embeddings
                SET embedding_vector = %s::vector
                WHERE doc_id = %s
            """, batch)
            conn.commit()
            batch = []

    # Process remaining
    if batch:
        update_cursor.executemany("""
            UPDATE document_embeddings
            SET embedding_vector = %s::vector
            WHERE doc_id = %s
        """, batch)
        conn.commit()

    print(f"✓ Migrated {total} embeddings to PGVector")


def create_vector_index(conn):
    """Create HNSW index for fast vector similarity search."""
    print("Creating vector index for similarity search...")
    cursor = conn.cursor()

    # Drop old index if exists
    cursor.execute("DROP INDEX IF EXISTS idx_embeddings_vector")

    # Create HNSW index with cosine distance
    cursor.execute(f"""
        CREATE INDEX idx_embeddings_vector
        ON document_embeddings
        USING hnsw (embedding_vector vector_cosine_ops)
    """)
    conn.commit()
    print("✓ Vector index created (HNSW with cosine distance)")


def main():
    print("=" * 60)
    print("PGVector Migration")
    print("=" * 60)
    print()

    conn = get_postgres_conn()

    try:
        add_pgvector_extension(conn)
        add_vector_column(conn)
        migrate_embeddings(conn)
        create_vector_index(conn)

        print()
        print("=" * 60)
        print("✓ PGVector migration complete!")
        print("=" * 60)

        # Show stats
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM document_embeddings WHERE embedding_vector IS NOT NULL")
        vector_count = cursor.fetchone()[0]
        print(f"Total vector embeddings: {vector_count}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
