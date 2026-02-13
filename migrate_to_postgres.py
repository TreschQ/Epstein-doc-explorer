#!/usr/bin/env python3
"""
Migrate SQLite database to PostgreSQL.

Usage:
    # Set DATABASE_URL environment variable first
    export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
    python migrate_to_postgres.py
"""

import os
import sqlite3
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

SQLITE_PATH = Path(__file__).parent / "document_analysis.db"
BATCH_SIZE = 500


def get_postgres_conn():
    """Get PostgreSQL connection from DATABASE_URL."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    return psycopg2.connect(url)


def create_postgres_schema(pg_conn):
    """Create PostgreSQL schema matching SQLite structure."""
    cursor = pg_conn.cursor()

    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            one_sentence_summary TEXT NOT NULL,
            paragraph_summary TEXT NOT NULL,
            date_range_earliest TEXT,
            date_range_latest TEXT,
            category TEXT NOT NULL,
            content_tags JSONB,
            analysis_timestamp TEXT NOT NULL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cache_read_tokens INTEGER,
            cost_usd REAL,
            error TEXT,
            full_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # RDF triples table (no FK constraint - some orphan triples exist in source data)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rdf_triples (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            timestamp TEXT,
            actor TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT NOT NULL,
            location TEXT,
            actor_likely_type TEXT,
            triple_tags JSONB,
            explicit_topic TEXT,
            implicit_topic TEXT,
            sequence_order INTEGER NOT NULL,
            top_cluster_ids JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Entity aliases table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_aliases (
            original_name TEXT PRIMARY KEY,
            canonical_name TEXT NOT NULL,
            reasoning TEXT,
            hop_distance_from_principal INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT DEFAULT 'llm_dedupe'
        )
    """)

    # Canonical entities table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS canonical_entities (
            canonical_name TEXT PRIMARY KEY,
            hop_distance_from_principal INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Document embeddings - store as bytea (binary)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_embeddings (
            doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
            embedding BYTEA,
            text_source TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tag embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tag_embeddings (
            tag TEXT PRIMARY KEY,
            embedding JSONB NOT NULL,
            model TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rdf_triples_doc_id ON rdf_triples(doc_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rdf_triples_actor ON rdf_triples(actor)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rdf_triples_target ON rdf_triples(target)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rdf_triples_timestamp ON rdf_triples(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical ON entity_aliases(canonical_name)")

    pg_conn.commit()
    print("✓ PostgreSQL schema created")


def migrate_documents(sqlite_conn, pg_conn):
    """Migrate documents table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM documents")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT doc_id, file_path, one_sentence_summary, paragraph_summary,
               date_range_earliest, date_range_latest, category, content_tags,
               analysis_timestamp, input_tokens, output_tokens, cache_read_tokens,
               cost_usd, error, full_text
        FROM documents
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating documents"):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO documents (doc_id, file_path, one_sentence_summary, paragraph_summary,
                    date_range_earliest, date_range_latest, category, content_tags,
                    analysis_timestamp, input_tokens, output_tokens, cache_read_tokens,
                    cost_usd, error, full_text)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO documents (doc_id, file_path, one_sentence_summary, paragraph_summary,
                date_range_earliest, date_range_latest, category, content_tags,
                analysis_timestamp, input_tokens, output_tokens, cache_read_tokens,
                cost_usd, error, full_text)
            VALUES %s
            ON CONFLICT (doc_id) DO NOTHING
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} documents")


def migrate_rdf_triples(sqlite_conn, pg_conn):
    """Migrate RDF triples table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM rdf_triples")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT doc_id, timestamp, actor, action, target, location,
               actor_likely_type, triple_tags, explicit_topic, implicit_topic,
               sequence_order, top_cluster_ids
        FROM rdf_triples
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating RDF triples"):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO rdf_triples (doc_id, timestamp, actor, action, target, location,
                    actor_likely_type, triple_tags, explicit_topic, implicit_topic,
                    sequence_order, top_cluster_ids)
                VALUES %s
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO rdf_triples (doc_id, timestamp, actor, action, target, location,
                actor_likely_type, triple_tags, explicit_topic, implicit_topic,
                sequence_order, top_cluster_ids)
            VALUES %s
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} RDF triples")


def migrate_entity_aliases(sqlite_conn, pg_conn):
    """Migrate entity aliases table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM entity_aliases")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT original_name, canonical_name, reasoning, hop_distance_from_principal
        FROM entity_aliases
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating entity aliases"):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO entity_aliases (original_name, canonical_name, reasoning, hop_distance_from_principal)
                VALUES %s
                ON CONFLICT (original_name) DO NOTHING
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO entity_aliases (original_name, canonical_name, reasoning, hop_distance_from_principal)
            VALUES %s
            ON CONFLICT (original_name) DO NOTHING
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} entity aliases")


def migrate_canonical_entities(sqlite_conn, pg_conn):
    """Migrate canonical entities table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM canonical_entities")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT canonical_name, hop_distance_from_principal
        FROM canonical_entities
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating canonical entities"):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO canonical_entities (canonical_name, hop_distance_from_principal)
                VALUES %s
                ON CONFLICT (canonical_name) DO NOTHING
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO canonical_entities (canonical_name, hop_distance_from_principal)
            VALUES %s
            ON CONFLICT (canonical_name) DO NOTHING
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} canonical entities")


def migrate_document_embeddings(sqlite_conn, pg_conn):
    """Migrate document embeddings table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM document_embeddings")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT doc_id, embedding, text_source, model
        FROM document_embeddings
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating embeddings"):
        doc_id, embedding_blob, text_source, model = row
        # Convert blob to psycopg2 Binary
        batch.append((doc_id, psycopg2.Binary(embedding_blob) if embedding_blob else None, text_source, model))

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO document_embeddings (doc_id, embedding, text_source, model)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO document_embeddings (doc_id, embedding, text_source, model)
            VALUES %s
            ON CONFLICT (doc_id) DO NOTHING
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} document embeddings")


def migrate_tag_embeddings(sqlite_conn, pg_conn):
    """Migrate tag embeddings table."""
    cursor = sqlite_conn.execute("SELECT COUNT(*) FROM tag_embeddings")
    total = cursor.fetchone()[0]

    cursor = sqlite_conn.execute("""
        SELECT tag, embedding, model
        FROM tag_embeddings
    """)

    pg_cursor = pg_conn.cursor()
    batch = []

    for row in tqdm(cursor, total=total, desc="Migrating tag embeddings"):
        batch.append(row)

        if len(batch) >= BATCH_SIZE:
            execute_values(pg_cursor, """
                INSERT INTO tag_embeddings (tag, embedding, model)
                VALUES %s
                ON CONFLICT (tag) DO NOTHING
            """, batch)
            pg_conn.commit()
            batch = []

    if batch:
        execute_values(pg_cursor, """
            INSERT INTO tag_embeddings (tag, embedding, model)
            VALUES %s
            ON CONFLICT (tag) DO NOTHING
        """, batch)
        pg_conn.commit()

    print(f"✓ Migrated {total} tag embeddings")


def main():
    print("=" * 50)
    print("SQLite → PostgreSQL Migration")
    print("=" * 50)

    if not SQLITE_PATH.exists():
        print(f"ERROR: SQLite database not found: {SQLITE_PATH}")
        sys.exit(1)

    print(f"Source: {SQLITE_PATH}")
    print(f"Target: {os.environ.get('DATABASE_URL', 'NOT SET')[:50]}...")
    print()

    # Connect to databases
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_conn = get_postgres_conn()

    try:
        # Create schema
        create_postgres_schema(pg_conn)

        # Migrate tables in order (respecting foreign keys)
        migrate_documents(sqlite_conn, pg_conn)
        migrate_rdf_triples(sqlite_conn, pg_conn)
        migrate_entity_aliases(sqlite_conn, pg_conn)
        migrate_canonical_entities(sqlite_conn, pg_conn)
        migrate_document_embeddings(sqlite_conn, pg_conn)
        migrate_tag_embeddings(sqlite_conn, pg_conn)

        print()
        print("=" * 50)
        print("✓ Migration complete!")
        print("=" * 50)

    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
