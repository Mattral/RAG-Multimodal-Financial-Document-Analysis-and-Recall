"""Tests for src/rag_system/sdk/__init__.py — previously 0% covered.

Covers the RAGPipeline SDK wrapper via mocking the internal pipeline.
"""
from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from src.rag_system.sdk import RAGPipeline, get_config


def _inner():
    p = MagicMock()
    p.ingest = AsyncMock(return_value={"status": "success", "num_chunks": 5})
    p.query = AsyncMock(return_value={"status": "success", "answer": "Revenue was $23B."})
    p.health_check = AsyncMock(return_value={"status": "healthy"})
    return p


class TestRAGPipelineInit:
    def test_init_sets_pipeline_and_tenant(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="acme")
        assert sdk._pipeline is inner
        assert sdk._default_tenant == "acme"

    def test_init_default_tenant(self):
        assert RAGPipeline(_inner())._default_tenant == "default"


class TestRAGPipelineCreate:
    @pytest.mark.asyncio
    async def test_create_returns_rag_pipeline(self):
        inner = _inner()
        with patch("src.rag_system.pipeline.RAGPipeline.create", AsyncMock(return_value=inner)):
            sdk = await RAGPipeline.create(tenant_id="acme")
        assert isinstance(sdk, RAGPipeline)
        assert sdk._default_tenant == "acme"

    @pytest.mark.asyncio
    async def test_create_default_tenant(self):
        inner = _inner()
        with patch("src.rag_system.pipeline.RAGPipeline.create", AsyncMock(return_value=inner)):
            sdk = await RAGPipeline.create()
        assert sdk._default_tenant == "default"


class TestRAGPipelineIngest:
    @pytest.mark.asyncio
    async def test_ingest_list_delegates_to_inner(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="acme")
        result = await sdk.ingest(["report.pdf"])
        inner.ingest.assert_called_once_with(
            file_paths=["report.pdf"], tenant_id="acme", process_vision=True
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_ingest_single_string_converted_to_list(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="default")
        await sdk.ingest("report.pdf")
        call_args = inner.ingest.call_args
        assert call_args.kwargs["file_paths"] == ["report.pdf"]

    @pytest.mark.asyncio
    async def test_ingest_tenant_override(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="default")
        await sdk.ingest(["f.pdf"], tenant_id="override")
        assert inner.ingest.call_args.kwargs["tenant_id"] == "override"

    @pytest.mark.asyncio
    async def test_ingest_vision_flag(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        await sdk.ingest(["f.pdf"], process_vision=False)
        assert inner.ingest.call_args.kwargs["process_vision"] is False


class TestRAGPipelineQuery:
    @pytest.mark.asyncio
    async def test_query_delegates_to_inner(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="acme")
        result = await sdk.query("What was revenue?")
        inner.query.assert_called_once_with(
            query_text="What was revenue?", tenant_id="acme", top_k=5, filters=None
        )
        assert result["answer"] == "Revenue was $23B."

    @pytest.mark.asyncio
    async def test_query_tenant_override(self):
        inner = _inner()
        sdk = RAGPipeline(inner, tenant_id="default")
        await sdk.query("q", tenant_id="override")
        assert inner.query.call_args.kwargs["tenant_id"] == "override"

    @pytest.mark.asyncio
    async def test_query_top_k_passed_through(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        await sdk.query("q", top_k=10)
        assert inner.query.call_args.kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_query_filters_passed_through(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        await sdk.query("q", filters={"doc_type": "10-K"})
        assert inner.query.call_args.kwargs["filters"] == {"doc_type": "10-K"}


class TestRAGPipelineSyncWrappers:
    def test_query_sync_runs_coroutine(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        result = sdk.query_sync("What was revenue?")
        assert result["answer"] == "Revenue was $23B."

    def test_ingest_sync_runs_coroutine(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        result = sdk.ingest_sync(["report.pdf"])
        assert result["status"] == "success"

    def test_ingest_sync_single_string(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        sdk.ingest_sync("report.pdf")
        assert inner.ingest.call_args.kwargs["file_paths"] == ["report.pdf"]


class TestRAGPipelineHealth:
    @pytest.mark.asyncio
    async def test_health_delegates_to_inner(self):
        inner = _inner()
        sdk = RAGPipeline(inner)
        result = await sdk.health()
        inner.health_check.assert_called_once()
        assert result["status"] == "healthy"


class TestSDKExports:
    def test_get_config_exported(self):
        assert callable(get_config)

    def test_all_contains_rag_pipeline(self):
        from src.rag_system.sdk import __all__
        assert "RAGPipeline" in __all__
