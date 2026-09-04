"""PostgreSQL pgvector adapter — drop-in replacement for DeepLake.

Guideline §6: 'Pluggable vector store backends: pgvector for teams already
running Postgres, Qdrant for dedicated vector DB deployments.'

Usage: set VECTOR_STORE_CONFIG__PROVIDER=pgvector and
       VECTOR_STORE_CONFIG__CONNECTION_STRING=postgresql://... in .env

Requires: pip install asyncpg pgvector
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import BaseVectorStore, DocumentElement, RetrievedChunk
from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)

_INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS rag_chunks_{table_suffix} (
    id          SERIAL PRIMARY KEY,
    text        TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL,
    source_doc  TEXT NOT NULL,
    page_number INTEGER,
    chunk_id    TEXT,
    tenant_id   TEXT NOT NULL DEFAULT 'default',
    element_type TEXT,
    metadata    JSONB DEFAULT '{{}}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS rag_chunks_{table_suffix}_embedding_idx
    ON rag_chunks_{table_suffix}
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX IF NOT EXISTS rag_chunks_{table_suffix}_tenant_idx
    ON rag_chunks_{table_suffix} (tenant_id);
"""


class PGVectorAdapter(BaseVectorStore):
    """pgvector-backed vector store with per-tenant row-level isolation."""

    def __init__(self) -> None:
        self._cfg = get_config().vector_store_config
        self._pool = None
        self._table = self._cfg.collection_name.replace("-", "_")
        self._dim = self._cfg.embedding_dim

    @property
    def name(self) -> str:
        return "pgvector"

    def _check_deps(self) -> bool:
        try:
            import asyncpg  # noqa: F401

            return True
        except ImportError:
            logger.warning("asyncpg_not_installed", detail="pip install asyncpg pgvector")
            return False

    async def initialize(self, tenant_id: Optional[str] = None) -> None:
        if not self._check_deps():
            return
        try:
            import asyncpg

            conn_str = self._cfg.connection_string
            if not conn_str:
                logger.error(
                    "pgvector_no_connection_string",
                    detail="Set VECTOR_STORE_CONFIG__CONNECTION_STRING",
                )
                return
            self._pool = await asyncpg.create_pool(conn_str, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INIT_SQL.format(table_suffix=self._table, dim=self._dim)
                )
            logger.info("pgvector_initialized", table=self._table, dim=self._dim)
        except Exception as exc:
            logger.error("pgvector_init_failed", error=str(exc))

    async def upsert(
        self,
        elements: List[DocumentElement],
        embeddings: List[List[float]],
        tenant_id: Optional[str] = None,
    ) -> None:
        if not self._pool:
            return
        tid = tenant_id or "default"
        try:
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    f"""INSERT INTO rag_chunks_{self._table}
                         (text, embedding, source_doc, page_number, chunk_id,
                          tenant_id, element_type, metadata)
                         VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8)""",
                    [
                        (
                            e.text,
                            "[" + ",".join(str(x) for x in vec) + "]",
                            e.source_document,
                            e.page_number,
                            e.content_hash,
                            tid,
                            e.type,
                            json.dumps(e.metadata),
                        )
                        for e, vec in zip(elements, embeddings, strict=True)
                    ],
                )
            logger.info(
                "pgvector_upsert_complete", num_chunks=len(elements), tenant_id=tid
            )
        except Exception as exc:
            logger.error("pgvector_upsert_failed", error=str(exc))
            raise

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not self._pool:
            return []
        tid = tenant_id or "default"
        vec_str = "[" + ",".join(str(x) for x in query_vector) + "]"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    f"""SELECT text, source_doc, page_number, chunk_id,
                              1 - (embedding <=> $1::vector) AS score
                         FROM rag_chunks_{self._table}
                         WHERE tenant_id = $2
                         ORDER BY embedding <=> $1::vector
                         LIMIT $3""",
                    vec_str,
                    tid,
                    top_k,
                )
            return [
                RetrievedChunk(
                    text=row["text"],
                    score=float(row["score"]),
                    source_document=row["source_doc"],
                    page_number=row["page_number"],
                    chunk_id=row["chunk_id"],
                )
                for row in rows
            ]
        except Exception as exc:
            logger.error("pgvector_search_failed", error=str(exc))
            return []

    async def delete(self, doc_ids: List[str], tenant_id: Optional[str] = None) -> None:
        if not self._pool:
            return
        tid = tenant_id or "default"
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"DELETE FROM rag_chunks_{self._table} "
                    f"WHERE tenant_id = $1 AND chunk_id = ANY($2::text[])",
                    tid,
                    doc_ids,
                )
            logger.info("pgvector_delete_complete", doc_ids=doc_ids, tenant_id=tid)
        except Exception as exc:
            logger.error("pgvector_delete_failed", error=str(exc))
