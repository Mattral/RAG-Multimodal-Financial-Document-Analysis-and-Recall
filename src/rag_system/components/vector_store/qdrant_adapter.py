"""Qdrant vector store adapter — production-ready async implementation.

Guideline §7: 'Qdrant for dedicated vector DB deployments.'

Usage:
    VECTOR_STORE_CONFIG__PROVIDER=qdrant
    VECTOR_STORE_CONFIG__QDRANT_URL=http://localhost:6333

Requires: pip install qdrant-client>=1.9.0
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import structlog

from src.rag_system.components.base import BaseVectorStore, DocumentElement, RetrievedChunk
from src.rag_system.config import get_config

logger = structlog.get_logger(__name__)


class QdrantAdapter(BaseVectorStore):
    """Qdrant vector store with per-tenant collection isolation and HNSW indexing."""

    def __init__(self) -> None:
        self._cfg = get_config().vector_store_config
        self._client = None
        self._async_client = None

    @property
    def name(self) -> str:
        return "qdrant"

    def _collection_name(self, tenant_id: Optional[str]) -> str:
        base = self._cfg.collection_name
        if tenant_id and tenant_id != "default":
            return f"{base}_{tenant_id}"
        return base

    def _get_url(self) -> str:
        return (
            os.environ.get("VECTOR_STORE_CONFIG__QDRANT_URL")
            or getattr(self._cfg, "qdrant_url", None)
            or "http://localhost:6333"
        )

    def _get_api_key(self) -> Optional[str]:
        return os.environ.get("VECTOR_STORE_CONFIG__QDRANT_API_KEY") or getattr(
            self._cfg, "qdrant_api_key", None
        )

    def _check_deps(self) -> bool:
        try:
            from qdrant_client import QdrantClient  # noqa: F401

            return True
        except ImportError:
            logger.error(
                "qdrant_client_not_installed",
                detail="pip install qdrant-client>=1.9.0",
            )
            return False

    def _get_async_client(self):
        if self._async_client is None:
            from qdrant_client import AsyncQdrantClient

            self._async_client = AsyncQdrantClient(
                url=self._get_url(), api_key=self._get_api_key(), timeout=60
            )
        return self._async_client

    async def initialize(self, tenant_id: Optional[str] = None) -> None:
        if not self._check_deps():
            return
        try:
            await self._ensure_collection(tenant_id)
            logger.info(
                "qdrant_initialized", collection=self._collection_name(tenant_id)
            )
        except Exception as exc:
            logger.error("qdrant_init_failed", error=str(exc))
            raise

    async def _ensure_collection(self, tenant_id: Optional[str]) -> None:
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams

        client = self._get_async_client()
        collection = self._collection_name(tenant_id)
        dim = self._cfg.embedding_dim
        existing = {c.name for c in (await client.get_collections()).collections}
        if collection not in existing:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(
                    m=16, ef_construct=200, full_scan_threshold=10000
                ),
            )
            logger.info(
                "qdrant_collection_created", collection=collection, dim=dim
            )

    async def upsert(
        self,
        elements: List[DocumentElement],
        embeddings: List[List[float]],
        tenant_id: Optional[str] = None,
    ) -> None:
        if not self._check_deps() or not elements:
            return
        from qdrant_client.models import PointStruct

        await self._ensure_collection(tenant_id)
        client = self._get_async_client()
        collection = self._collection_name(tenant_id)
        points = [
            PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL, elem.content_hash or str(i)
                    )
                ),
                vector=vec,
                payload={
                    "text": elem.text,
                    "source_document": elem.source_document,
                    "page_number": elem.page_number,
                    "chunk_type": elem.type,
                    "content_hash": elem.content_hash,
                    "tenant_id": tenant_id or "default",
                    "metadata": elem.metadata or {},
                },
            )
            for i, (elem, vec) in enumerate(zip(elements, embeddings, strict=True))
        ]
        for i in range(0, len(points), 256):
            await client.upsert(
                collection_name=collection, points=points[i : i + 256]
            )
        logger.info(
            "qdrant_upsert_complete", num_points=len(points), tenant_id=tenant_id
        )

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        if not self._check_deps():
            return []
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        client = self._get_async_client()
        collection = self._collection_name(tenant_id)
        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            if conditions:
                qdrant_filter = Filter(must=conditions)
        try:
            hits = await client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:
            logger.error("qdrant_search_failed", error=str(exc))
            return []
        return [
            RetrievedChunk(
                text=h.payload.get("text", ""),
                score=float(h.score),
                source_document=h.payload.get("source_document", "unknown"),
                page_number=h.payload.get("page_number"),
                chunk_id=h.payload.get("content_hash"),
                metadata=h.payload.get("metadata", {}),
            )
            for h in hits
        ]

    async def delete(self, doc_ids: List[str], tenant_id: Optional[str] = None) -> None:
        if not self._check_deps():
            return
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = self._get_async_client()
        collection = self._collection_name(tenant_id)
        try:
            await client.delete(
                collection_name=collection,
                points_selector=Filter(
                    must=[FieldCondition(key="content_hash", match=MatchAny(any=doc_ids))]
                ),
            )
        except Exception as exc:
            logger.error("qdrant_delete_failed", error=str(exc))

    async def health_check(self) -> None:
        await self._get_async_client().get_collections()
