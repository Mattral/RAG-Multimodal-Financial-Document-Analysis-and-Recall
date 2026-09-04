"""Tests for reranker/ — previously 0% covered."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from src.rag_system.components.base import RetrievedChunk
from src.rag_system.components.reranker import CohereReranker, CrossEncoderReranker, NoOpReranker, build_reranker

def _c(text, score=0.5, doc="f.pdf"):
    return RetrievedChunk(text=text, score=score, source_document=doc, page_number=1)

class TestNoOpReranker:
    def test_name(self): assert NoOpReranker().name == "noop"
    @pytest.mark.asyncio
    async def test_top_n(self):
        cs = [_c("a"),_c("b"),_c("c")]
        assert await NoOpReranker().rerank("q", cs, top_n=2) == cs[:2]
    @pytest.mark.asyncio
    async def test_empty(self): assert await NoOpReranker().rerank("q", [], top_n=5) == []
    @pytest.mark.asyncio
    async def test_top_n_larger(self):
        cs = [_c("a"),_c("b")]
        assert await NoOpReranker().rerank("q", cs, top_n=10) == cs

class TestCrossEncoderReranker:
    def test_name(self):
        r = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "cross_encoder" in r.name and "ms-marco" in r.name
    def test_default_model(self): assert "ms-marco" in CrossEncoderReranker()._model_name
    @pytest.mark.asyncio
    async def test_empty_no_model_load(self):
        r = CrossEncoderReranker()
        with patch.object(r, "_get_model") as m:
            assert await r.rerank("q", [], top_n=5) == []
            m.assert_not_called()
    @pytest.mark.asyncio
    async def test_fallback_when_model_none(self):
        r = CrossEncoderReranker()
        cs = [_c("a"),_c("b"),_c("c")]
        with patch.object(r, "_get_model", return_value=None):
            assert await r.rerank("q", cs, top_n=2) == cs[:2]
    def test_none_without_library(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            assert CrossEncoderReranker()._get_model() is None
    @pytest.mark.asyncio
    async def test_reorders_by_score(self):
        r = CrossEncoderReranker()
        cs = [_c("low", 0.9), _c("high", 0.1)]
        m = MagicMock(); m.predict = MagicMock(return_value=[0.2, 0.95])
        with patch.object(r, "_get_model", return_value=m):
            result = await r.rerank("q", cs, top_n=2)
        assert result[0].text == "high" and result[0].score == pytest.approx(0.95)
    @pytest.mark.asyncio
    async def test_respects_top_n(self):
        r = CrossEncoderReranker()
        cs = [_c(f"c{i}") for i in range(5)]
        m = MagicMock(); m.predict = MagicMock(return_value=[0.1,0.2,0.3,0.4,0.5])
        with patch.object(r, "_get_model", return_value=m):
            assert len(await r.rerank("q", cs, top_n=2)) == 2
    @pytest.mark.asyncio
    async def test_preserves_fields(self):
        r = CrossEncoderReranker()
        cs = [_c("text", doc="tesla.pdf")]
        m = MagicMock(); m.predict = MagicMock(return_value=[0.7])
        with patch.object(r, "_get_model", return_value=m):
            result = await r.rerank("q", cs, top_n=1)
        assert result[0].source_document == "tesla.pdf"

class TestCohereReranker:
    def test_name(self): assert CohereReranker(model="rerank-english-v3.0").name == "cohere/rerank-english-v3.0"
    def test_default_model(self): assert CohereReranker()._model == "rerank-english-v3.0"
    @pytest.mark.asyncio
    async def test_fallback_without_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        from src.rag_system.config import reset_config; reset_config()
        cs = [_c("a"),_c("b"),_c("c")]
        assert await CohereReranker().rerank("q", cs, top_n=2) == cs[:2]
        reset_config()
    def test_none_without_library(self):
        with patch.dict("sys.modules", {"cohere": None}):
            assert CohereReranker()._get_client() is None
    def test_none_on_exception(self):
        bad = MagicMock(); bad.Client = MagicMock(side_effect=RuntimeError("boom"))
        with patch.dict("sys.modules", {"cohere": bad}):
            assert CohereReranker()._get_client() is None
    @pytest.mark.asyncio
    async def test_maps_response(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        from src.rag_system.config import reset_config; reset_config()
        r = CohereReranker()
        cs = [_c("first"), _c("second"), _c("third")]
        r0 = MagicMock(index=2, relevance_score=0.95)
        r1 = MagicMock(index=0, relevance_score=0.42)
        mc = MagicMock(); mc.rerank = MagicMock(return_value=MagicMock(results=[r0,r1]))
        with patch.object(r, "_get_client", return_value=mc):
            result = await r.rerank("q", cs, top_n=2)
        assert result[0].text == "third" and result[0].score == pytest.approx(0.95)
        reset_config()

class TestBuildRerankerFactory:
    def test_cross_encoder(self): assert isinstance(build_reranker("cross_encoder"), CrossEncoderReranker)
    def test_cohere(self): assert isinstance(build_reranker("cohere"), CohereReranker)
    def test_none_noop(self): assert isinstance(build_reranker("none"), NoOpReranker)
    def test_unknown_cross_encoder(self): assert isinstance(build_reranker("xyz"), CrossEncoderReranker)
