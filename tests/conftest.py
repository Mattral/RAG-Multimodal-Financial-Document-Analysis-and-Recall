"""Shared pytest fixtures and configuration for all test suites.

Provides:
  - Mock API keys in environment
  - In-memory pipeline fixture (no external deps)
  - Sample documents and elements
  - Async event loop configuration
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Environment setup before any imports ─────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-tests")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("VECTOR_STORE_CONFIG__PROVIDER", "memory")
os.environ.setdefault("CACHE_CONFIG__BACKEND", "memory")
os.environ.setdefault("LOGGING_CONFIG__LEVEL", "WARNING")
os.environ.setdefault("SECURITY_CONFIG__ENABLE_PII_REDACTION", "false")
os.environ.setdefault("SECURITY_CONFIG__ENABLE_GUARDRAILS", "false")


# ── Async event loop ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ── Config reset between tests ─────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def reset_config():
    from src.rag_system.config import reset_config as _reset

    _reset()
    yield
    _reset()


# ── Shared data fixtures ──────────────────────────────────────────────────────
@pytest.fixture
def sample_document_element():
    from src.rag_system.components.base import DocumentElement

    return DocumentElement(
        type="text",
        text="Tesla's Q3 2023 revenue was $23.35 billion, up 9% year-over-year.",
        source_document="tesla_10q_q3_2023.pdf",
        page_number=4,
        content_hash="abc123def456",
        tenant_id="test_tenant",
        metadata={"parser": "unstructured", "section": "Financial Results"},
    )


@pytest.fixture
def sample_table_element():
    from src.rag_system.components.base import DocumentElement

    return DocumentElement(
        type="table",
        text="| Metric | Q3 2023 | Q3 2022 | YoY |\n|---|---|---|---|\n| Revenue | $23.35B | $21.45B | +9% |\n| Gross Margin | 17.9% | 25.1% | -720bps |",
        source_document="tesla_10q_q3_2023.pdf",
        page_number=5,
        content_hash="table456abc",
        tenant_id="test_tenant",
    )


@pytest.fixture
def sample_graph_element():
    from src.rag_system.components.base import DocumentElement

    return DocumentElement(
        type="graph",
        text="Bar chart titled 'Quarterly Revenue 2020-2023'. X-axis: quarters Q1-Q4. Y-axis: Revenue in billions USD. Q3 2023 bar shows $23.35B. Clear upward trend from $6B in Q1 2020 to $23.35B [...]",
        source_document="tesla_investor_deck.pdf",
        page_number=8,
        content_hash="graph789xyz",
        tenant_id="test_tenant",
        metadata={"vision_model": "gpt-4o", "image_path": "/tmp/chart.png"},
    )


@pytest.fixture
def sample_chunks():
    from src.rag_system.components.base import RetrievedChunk

    return [
        RetrievedChunk(
            text="Q3 revenue was $23.35B.",
            score=0.92,
            source_document="tesla_10q_q3_2023.pdf",
            page_number=4,
            chunk_id="c1",
        ),
        RetrievedChunk(
            text="Gross margin was 17.9% in Q3 2023.",
            score=0.87,
            source_document="tesla_10q_q3_2023.pdf",
            page_number=5,
            chunk_id="c2",
        ),
        RetrievedChunk(
            text="Vehicle deliveries reached 435,059 units in Q3 2023.",
            score=0.81,
            source_document="tesla_10q_q3_2023.pdf",
            page_number=3,
            chunk_id="c3",
        ),
    ]


@pytest.fixture
def sample_generated_answer(sample_chunks):
    from src.rag_system.components.base import GeneratedAnswer

    return GeneratedAnswer(
        answer="Tesla's Q3 2023 revenue was $23.35 billion [Source: tesla_10q_q3_2023.pdf, Page 4], with a gross margin of 17.9% [Source: tesla_10q_q3_2023.pdf, Page 5].",
        citations=sample_chunks[:2],
        model_used="gpt-4o-mini",
        prompt_tokens=450,
        completion_tokens=85,
        estimated_cost_usd=0.000119,
        latency_ms=1342.5,
        tenant_id="test_tenant",
    )


# ── Mock component fixtures ───────────────────────────────────────────────────
@pytest.fixture
def mock_openai_embedder():
    embedder = MagicMock()
    embedder.name = "openai/text-embedding-3-small"
    embedder.dimension = 1536
    embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]] * 1536)
    embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3] * 512)
    return embedder


@pytest.fixture
def mock_generator(sample_generated_answer):
    gen = MagicMock()
    gen.name = "mock_generator"
    gen.generate = AsyncMock(return_value=sample_generated_answer)
    return gen


@pytest.fixture
def mock_parser(sample_document_element, sample_table_element):
    parser = MagicMock()
    parser.name = "mock_unstructured"
    parser.parse = AsyncMock(return_value=[sample_document_element])
    parser.parse_batch = AsyncMock(
        return_value=[sample_document_element, sample_table_element]
    )
    return parser


@pytest.fixture
async def in_memory_vector_store():
    from src.rag_system.components.vector_store import InMemoryVectorStore

    store = InMemoryVectorStore()
    await store.initialize(tenant_id="test_tenant")
    return store


@pytest.fixture
def bm25_index(sample_chunks):
    from src.rag_system.components.retriever import BM25Index

    idx = BM25Index()
    idx.build(sample_chunks)
    return idx


@pytest.fixture
async def minimal_pipeline():
    """Minimal pipeline with in-memory components — no external deps."""
    from src.rag_system.components.vector_store import InMemoryVectorStore
    from src.rag_system.pipeline import RAGPipeline

    store = InMemoryVectorStore()
    await store.initialize("test_tenant")

    return RAGPipeline(vector_store=store)


# ── tmp_path helpers ──────────────────────────────────────────────────────────
@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a dummy PDF file path for testing (does not parse)."""
    pdf_file = tmp_path / "test_financial_report.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 dummy content for testing")
    return str(pdf_file)
