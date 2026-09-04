"""Abstract Base Classes and Protocols for pluggable RAG components.

Every major component has a well-defined interface so implementations
can be swapped via config without touching pipeline code.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Shared data models
# ---------------------------------------------------------------------------


class DocumentElement(BaseModel):
    """Structured document element with full provenance metadata."""

    model_config = ConfigDict(frozen=True)

    type: str  # "text" | "table" | "graph" | "image"
    text: str
    source_document: str
    page_number: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None  # {x0, y0, x1, y1}
    ingest_timestamp: Optional[str] = None
    content_hash: Optional[str] = None
    tenant_id: Optional[str] = None
    doc_version: Optional[str] = None
    metadata: Dict[str, Any] = {}


class RetrievedChunk(BaseModel):
    """A retrieved document chunk with score and provenance."""

    text: str
    score: float
    source_document: str
    page_number: Optional[int] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class GeneratedAnswer(BaseModel):
    """LLM-generated answer with citations and audit trail."""

    answer: str
    citations: List[RetrievedChunk] = []
    model_used: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    hallucination_score: Optional[float] = None  # 0=grounded, 1=hallucinated
    tenant_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Abstract Base Classes
# ---------------------------------------------------------------------------


class BaseParser(abc.ABC):
    """Parse raw documents (PDF, DOCX, HTML) into DocumentElements."""

    @abc.abstractmethod
    async def parse(self, file_path: str, tenant_id: Optional[str] = None) -> List[DocumentElement]:
        """Parse a single document."""
        ...

    @abc.abstractmethod
    async def parse_batch(
        self, file_paths: List[str], tenant_id: Optional[str] = None
    ) -> List[DocumentElement]:
        """Parse multiple documents concurrently."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this parser implementation."""
        ...


class BaseVisionDescriber(abc.ABC):
    """Describe images/charts from financial documents."""

    @abc.abstractmethod
    async def describe(
        self, image_path: str, source_document: str, tenant_id: Optional[str] = None
    ) -> Optional[DocumentElement]:
        """Describe a single image; returns None if no visual content found."""
        ...

    @abc.abstractmethod
    async def describe_batch(
        self,
        image_paths: List[str],
        source_document: str,
        tenant_id: Optional[str] = None,
    ) -> List[DocumentElement]:
        """Describe multiple images concurrently."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name of this vision model."""
        ...


class BaseEmbedder(abc.ABC):
    """Embed text into dense vectors."""

    @abc.abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts; returns list of float vectors."""
        ...

    @abc.abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        ...

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


class BaseVectorStore(abc.ABC):
    """Persist and retrieve dense embeddings."""

    @abc.abstractmethod
    async def initialize(self, tenant_id: Optional[str] = None) -> None:
        """Initialize/connect to the vector store."""
        ...

    @abc.abstractmethod
    async def upsert(
        self,
        elements: List[DocumentElement],
        embeddings: List[List[float]],
        tenant_id: Optional[str] = None,
    ) -> None:
        """Insert or update document vectors."""
        ...

    @abc.abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Dense vector similarity search."""
        ...

    @abc.abstractmethod
    async def delete(
        self,
        doc_ids: List[str],
        tenant_id: Optional[str] = None,
    ) -> None:
        """Delete documents by ID (for GDPR/CCPA compliance)."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


class BaseRetriever(abc.ABC):
    """Retrieve relevant chunks for a query (may combine multiple sources)."""

    @abc.abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-k relevant chunks."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


class BaseReranker(abc.ABC):
    """Rerank retrieved chunks for relevance."""

    @abc.abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        """Rerank and return top_n most relevant chunks."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


class BaseGenerator(abc.ABC):
    """Generate answers from context chunks."""

    @abc.abstractmethod
    async def generate(
        self,
        query: str,
        context: List[RetrievedChunk],
        tenant_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> GeneratedAnswer:
        """Generate a grounded answer from context."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


class BaseEvaluator(abc.ABC):
    """Evaluate RAG pipeline quality."""

    @abc.abstractmethod
    async def evaluate(
        self,
        query: str,
        answer: GeneratedAnswer,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """Return a dict of metric_name → score."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name."""
        ...


# ---------------------------------------------------------------------------
# Protocols for duck-typing flexibility
# ---------------------------------------------------------------------------


@runtime_checkable
class Cacheable(Protocol):
    """Components that support cache invalidation."""

    async def clear_cache(self) -> None:
        """Clear any cached data."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Components that expose a health check."""

    async def health_check(self) -> Dict[str, Any]:
        """Return health check result."""
        ...
